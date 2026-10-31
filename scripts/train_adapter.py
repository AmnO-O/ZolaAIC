"""Offline QLoRA training. Requires prepared private train data and a CUDA GPU."""
import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.training import encode_example, pad_batch, assistant_loss, frozen_io_precision
from lifemate.prompts import INSTRUCTION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--epochs', type=float, default=1)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--smoke', action='store_true', help='Only 8 train / 2 validation examples and one optimizer step')
    args = parser.parse_args()
    if args.smoke:
        args.max_steps = 1
    if not 1 <= args.max_length <= 16384:
        parser.error('max-length must be 1..16384')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig,
                              Trainer, TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise RuntimeError('Adapter training requires CUDA; CPU checks do not establish GPU readiness')
    set_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    data = Path(args.data)
    manifest = json.loads((data/'manifest.json').read_text())
    prompt_hash = hashlib.sha256(INSTRUCTION.encode()).hexdigest()
    if manifest['prompt_sha256'] != prompt_hash:
        raise ValueError('Prepared data belongs to a different policy; prepare it again')
    datasets = {}
    skipped = {}
    for partition in ('train', 'validation'):
        datasets[partition] = []
        skipped[partition] = []
        lines = (data/(partition+'.jsonl')).read_text(encoding='utf-8').splitlines()
        if args.smoke:
            lines = lines[:8 if partition == 'train' else 2]
        for line in lines:
            example = json.loads(line)
            encoded = encode_example(example, tokenizer, args.max_length)
            if encoded is None:
                skipped[partition].append({'id': example['id'], 'step': example['step']})
            else:
                datasets[partition].append(encoded)
    (output/'skipped_length.json').write_text(json.dumps(skipped, indent=2))
    if not datasets['train'] or not datasets['validation']:
        raise ValueError('Need nonempty train and validation after length checks; no silent truncation')
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    loader = Qwen3_5ForConditionalGeneration if model_config.model_type == 'qwen3_5' else AutoModelForCausalLM
    model = loader.from_pretrained(args.model, local_files_only=True,
                quantization_config=quant, device_map={'': 0}, torch_dtype=torch.float16,
                attn_implementation='sdpa')
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant': False})
    frozen_io_precision(model, torch.float16)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                     target_modules=['q_proj','k_proj','v_proj','o_proj'], bias='none', task_type='CAUSAL_LM'))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(len(datasets['train'])/8)*args.epochs
    warmup_steps = 0 if args.smoke else math.ceil(total_steps*0.05)
    training = TrainingArguments(output_dir=str(output/'checkpoints'), num_train_epochs=args.epochs,
               max_steps=args.max_steps, per_device_train_batch_size=1, per_device_eval_batch_size=1,
               gradient_accumulation_steps=8, learning_rate=1e-4, warmup_steps=warmup_steps,
               fp16=True, bf16=False, gradient_checkpointing=True,
               gradient_checkpointing_kwargs={'use_reentrant': False}, optim='adamw_torch',
               logging_steps=1, eval_strategy='epoch', save_strategy='epoch',
               save_total_limit=1, report_to='none', seed=42, data_seed=42,
               full_determinism=True, remove_unused_columns=False,
               prediction_loss_only=True, dataloader_num_workers=0)
    class AssistantTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = assistant_loss(model, inputs)
            return (loss, outputs) if return_outputs else loss

    trainer = AssistantTrainer(model=model, args=training, train_dataset=datasets['train'],
                      eval_dataset=datasets['validation'], processing_class=tokenizer,
                      data_collator=partial(pad_batch, pad_token_id=tokenizer.pad_token_id))
    # The custom loss is a mean per sequence; let Trainer scale accumulation.
    trainer.model_accepts_loss_kwargs = False
    report = {'arguments': vars(args), 'data_manifest': manifest, 'prompt_sha256': prompt_hash,
              'retained_examples': {k: len(v) for k,v in datasets.items()},
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
              'frozen_vocabulary_dtype': 'float16',
              'packages': subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)}
    (output/'training_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    trainer.train()
    trainer.save_model(str(output/'adapter'))
    tokenizer.save_pretrained(str(output/'adapter'))
    (output/'adapter'/'lifemate_policy.json').write_text(json.dumps({
        'prompt_sha256': prompt_hash, 'base_model': args.model, 'seed': 42}, indent=2))
    # Validation loss is not task accuracy. The frozen adapter must run the live tool evaluator.
    metrics = trainer.evaluate()
    (output/'validation_loss.json').write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
