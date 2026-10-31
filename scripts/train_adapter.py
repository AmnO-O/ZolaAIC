"""Offline QLoRA training. Requires prepared private train data and a CUDA GPU."""
import argparse
from functools import partial
import hashlib
import json
import math
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.training import encode_example, pad_batch, assistant_loss, frozen_io_precision
from lifemate.prompts import INSTRUCTION
from lifemate.checkpointing import verify_resume, session_callback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--epochs', type=float, default=1)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--save-steps', type=int, default=10)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--max-run-seconds', type=float, default=0, help='0=unlimited; otherwise save and stop at a step boundary')
    parser.add_argument('--resume-from-checkpoint', help='Own complete checkpoint with matching training identity')
    parser.add_argument('--base-revision', default='', help='Pinned public base checkpoint revision for resume identity')
    parser.add_argument('--smoke', action='store_true', help='Only 8 train / 2 validation examples and one optimizer step')
    args = parser.parse_args()
    if args.smoke:
        args.max_steps = 1
    if not 1 <= args.max_length <= 16384:
        parser.error('max-length must be 1..16384')
    if (args.save_steps<1 or args.eval_steps<1 or not 0<args.learning_rate<float('inf')
            or not 0<=args.max_run_seconds<float('inf') or not 0<args.epochs<float('inf')):
        parser.error('Invalid training/session limits')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig,
                              Trainer, TrainingArguments, set_seed)
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
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
    targets=[name for name,_ in model.named_modules() if name.rsplit('.',1)[-1]
             in ('q_proj','k_proj','v_proj','o_proj') and '.visual.' not in name]
    if not targets: raise ValueError('No text attention LoRA targets')
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    identity={'prompt_sha256':prompt_hash,'train_sha256':sha(data/'train.jsonl'),
        'validation_sha256':sha(data/'validation.jsonl'),
        'base_config_sha256':sha(Path(args.model)/'config.json'),'base_revision':args.base_revision,
        'max_length':args.max_length,'epochs':args.epochs,'max_steps':args.max_steps,
        'smoke':args.smoke,'seed':42,'gradient_accumulation':8,'batch_size':1,
        'learning_rate':args.learning_rate,'lora_targets':targets,'rank':16,'alpha':32,
        'frozen_io_dtype':'float16','initial_grad_scale':256.,
        'trainer_script_sha256':sha(Path(__file__)),
        'encoding_script_sha256':sha(Path(__file__).resolve().parents[1]/'src/lifemate/training.py'),
        'package_versions':{name:importlib.metadata.version(name) for name in
                            ('torch','transformers','accelerate','peft','bitsandbytes')}}
    previous_state=None
    if args.resume_from_checkpoint:
        previous_state=verify_resume(args.resume_from_checkpoint,identity)
        model=PeftModel.from_pretrained(model,args.resume_from_checkpoint,is_trainable=True,local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                         target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    initial_adapters={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(len(datasets['train'])/8)*args.epochs
    warmup_steps = 0 if args.smoke else math.ceil(total_steps*0.05)
    training = TrainingArguments(output_dir=str(output/'checkpoints'), num_train_epochs=args.epochs,
               max_steps=args.max_steps, per_device_train_batch_size=1, per_device_eval_batch_size=1,
               gradient_accumulation_steps=8, learning_rate=args.learning_rate, warmup_steps=warmup_steps,
               fp16=True, bf16=False, gradient_checkpointing=True,
               gradient_checkpointing_kwargs={'use_reentrant': False}, optim='adamw_torch',
               logging_steps=1, eval_strategy='epoch' if args.smoke else 'steps',
               eval_steps=args.eval_steps, save_strategy='steps', save_steps=args.save_steps,
               save_total_limit=2, report_to='none', seed=42, data_seed=42,
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
    # Explicit low initial scale for the bounded FP16 smoke, not a silent retry.
    if trainer.accelerator.scaler is not None:
        trainer.accelerator.scaler = torch.amp.GradScaler('cuda',init_scale=256.)
    report = {'arguments': vars(args), 'data_manifest': manifest, 'prompt_sha256': prompt_hash,
              'retained_examples': {k: len(v) for k,v in datasets.items()},
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
              'frozen_vocabulary_dtype': 'float16',
              'lora_targets':targets, 'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
              'status':'started', 'initial_grad_scale':256.,'training_identity':identity,
              'resume_start_step':previous_state['global_step'] if previous_state else 0,
              'packages': subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)}
    (output/'training_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    callback=session_callback(identity,args.max_run_seconds)
    trainer.add_callback(callback)
    try:
        trained=trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        changes=[]
        for name,param in model.named_parameters():
            if name in initial_adapters:
                changes.append(float((param.detach().cpu().float()-initial_adapters[name].float()).abs().max()))
        report.update(global_step=trainer.state.global_step,train_metrics=trained.metrics,
                      changed_adapter_tensors=sum(x>0 for x in changes),max_adapter_delta=max(changes,default=0),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        if not report['changed_adapter_tensors']:
            raise RuntimeError('No adapter weights changed; optimizer-step count alone is not proof of learning')
        report['status']='checkpointed_time_budget' if callback.time_limit_reached else 'training_completed'
        report['epoch_completed']=bool(trainer.state.epoch is not None and trainer.state.epoch>=args.epochs-1e-6)
        report['latest_checkpoint']=str(Path(training.output_dir)/f'checkpoint-{trainer.state.global_step}')
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        (output/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise
    (output/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    trainer.save_model(str(output/'adapter'))
    tokenizer.save_pretrained(str(output/'adapter'))
    (output/'adapter'/'lifemate_policy.json').write_text(json.dumps({
        'prompt_sha256': prompt_hash, 'base_model': args.model, 'seed': 42,
        'training_status':report['status'],'epoch_completed':report['epoch_completed'],
        'retrieval':{k:manifest.get('retrieval',{}).get(k) for k in
                     ('top_k','query','train_only_target_schema_supervision')}}, indent=2))
    # Validation loss is not task accuracy. The frozen adapter must run the live tool evaluator.
    metrics=next((entry for entry in reversed(trainer.state.log_history) if 'eval_loss' in entry),None)
    if not callback.time_limit_reached and (metrics is None or metrics.get('step')!=trainer.state.global_step):
        metrics = trainer.evaluate()
        metrics['global_step']=trainer.state.global_step
    if metrics is None:
        metrics={'evaluation_deferred':True,'global_step':trainer.state.global_step}
    (output/'validation_loss.json').write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
