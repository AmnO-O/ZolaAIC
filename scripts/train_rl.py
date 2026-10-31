"""Experimental multi-turn, single-update GRPO pilot after SFT (private train only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.input_adapter import SplitInput
from lifemate.prompts import INSTRUCTION
from lifemate.rl import train_task, rollout, score_episode, group_advantages, policy_loss


def load_tasks(directory, prepared, limit):
    split = SplitInput(directory)
    if split.split != 'train':
        raise ValueError('Only train may be used for RL')
    source = split.directory/'train.jsonl'
    manifest = json.loads((Path(prepared)/'manifest.json').read_text(encoding='utf-8'))
    digest = lambda value: hashlib.sha256(value).hexdigest()
    if manifest['source_sha256'] != digest(source.read_bytes()):
        raise ValueError('Train source differs from SFT manifest')
    if manifest['catalog_sha256'] != digest(json.dumps(split.catalog, sort_keys=True).encode()):
        raise ValueError('Catalog differs from SFT manifest')
    if manifest['prompt_sha256'] != digest(INSTRUCTION.encode()):
        raise ValueError('Policy differs from SFT manifest')
    selected = set(manifest['train'])
    if selected & set(manifest['validation']):
        raise ValueError('Train/validation overlap')
    tasks = []
    with source.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['id'] in selected:
                tasks.append(train_task(row, split.tools(row), split.catalog))
            if len(tasks) >= limit:
                break
    if not tasks:
        raise ValueError('No RL train tasks')
    return split, tasks, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--sft-adapter', required=True)
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--data', required=True, help='Prepared SFT folder containing split manifest')
    parser.add_argument('--output', required=True)
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--group-size', type=int, default=2)
    parser.add_argument('--max-turns', type=int, default=8)
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--max-new-tokens', type=int, default=384)
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    args = parser.parse_args()
    if (args.groups < 1 or args.group_size < 2 or args.max_turns < 1
            or not 1 <= args.max_length <= 16384 or args.max_new_tokens < 1
            or not 0 < args.seconds < float('inf')
            or not 0 < args.learning_rate < float('inf')):
        parser.error('Invalid RL limits')
    split, tasks, manifest = load_tasks(args.train_dir, args.data, args.groups)
    metadata = json.loads((Path(args.sft_adapter)/'lifemate_policy.json').read_text())
    if metadata['prompt_sha256'] != manifest['prompt_sha256']:
        raise ValueError('SFT adapter policy mismatch')
    if Path(metadata['base_model']).resolve() != Path(args.model).resolve():
        raise ValueError('SFT base-model path differs; use the original pinned local base')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from transformers import (AutoTokenizer, AutoConfig, AutoModelForCausalLM,
                              Qwen3_5ForConditionalGeneration, BitsAndBytesConfig, set_seed)
    from peft import PeftModel, prepare_model_for_kbit_training
    from lifemate.rl_actor import RolloutActor
    if not torch.cuda.is_available():
        raise RuntimeError('RL pilot requires CUDA')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'started', 'arguments': vars(args), 'seed': 42,
              'algorithm': 'single-update on-policy group-relative policy gradient; no KL/reference model',
              'reward': 'diagnostic exact-call proxy, NOT verified state/task success',
              'train_ids': [task['id'] for task in tasks],
              'validation_ids': manifest['validation'],
              'prompt_sha256': manifest['prompt_sha256'],
              'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'packages': subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True),
              'gpu': torch.cuda.get_device_name(0), 'updates': 0, 'zero_signal_groups': 0}
    def save_report():
        (output/'training_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save_report()
    try:
        set_seed(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        if config.model_type not in ('qwen3', 'qwen3_5'):
            raise ValueError('Unsupported architecture')
        loader = Qwen3_5ForConditionalGeneration if config.model_type == 'qwen3_5' else AutoModelForCausalLM
        model = loader.from_pretrained(args.model, local_files_only=True,
                    quantization_config=quant, device_map={'':0}, torch_dtype=torch.float16,
                    attn_implementation='sdpa')
        model = prepare_model_for_kbit_training(model,
                    gradient_checkpointing_kwargs={'use_reentrant': False})
        model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True,
                                          local_files_only=True)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        actor = RolloutActor(model, tokenizer, args.max_new_tokens, args.max_length)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=args.learning_rate)
        scaler = torch.amp.GradScaler('cuda')
        with (output/'episodes.jsonl').open('x', encoding='utf-8') as logs:
            for group, task in enumerate(tasks):
                episodes = [rollout(task, actor, split.session, args.seconds, args.max_turns)
                            for _ in range(args.group_size)]
                if len({ep['initial_hash'] for ep in episodes}) != 1:
                    raise RuntimeError('ToolEnv reset does not reproduce identical starting state')
                rewards = [score_episode(ep, task['gold_calls'], task['tools']) for ep in episodes]
                advantages = group_advantages([reward['total'] for reward in rewards])
                for index, (ep, reward, advantage) in enumerate(zip(episodes, rewards, advantages)):
                    logs.write(json.dumps({'id': task['id'], 'group': group, 'rollout': index,
                        'reward': reward, 'advantage': advantage, **ep}, ensure_ascii=False)+'\n')
                logs.flush()
                optimizer.zero_grad(set_to_none=True)
                if not any(advantages):
                    report['zero_signal_groups'] += 1
                    save_report()
                    continue
                # All samples come from the same frozen weights. One update only,
                # so stop-gradient current logps are the on-policy old logps.
                # eval() disables dropout; checkpointing is explicitly activated
                # by train() during scoring (LoRA dropout is zero).
                model.train()
                for ep, advantage in zip(episodes, advantages):
                    count = sum(len(s['response_ids']) for s in ep['segments'])
                    if not count or advantage == 0:
                        continue
                    for segment in ep['segments']:
                        with torch.autocast('cuda', dtype=torch.float16):
                            logps = actor.logps(segment)
                            loss = policy_loss(logps, advantage, count, args.group_size)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite RL loss')
                        scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                report['updates'] += 1
                report['last_gradient_norm'] = float(norm)
                save_report()
        model.save_pretrained(output/'adapter')
        tokenizer.save_pretrained(output/'adapter')
        (output/'adapter'/'lifemate_policy.json').write_text(json.dumps({
            **metadata, 'stage':'rl_pilot', 'seed':42, 'updates':report['updates']}), encoding='utf-8')
        report['status'] = 'completed' if report['updates'] else 'completed_without_learning_signal'
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        save_report()
        print(json.dumps({k:report[k] for k in ('status','updates','zero_signal_groups')}, indent=2))
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        save_report()
        raise


if __name__ == '__main__':
    main()
