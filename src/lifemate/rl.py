"""Train-only on-policy rollout utilities. No label is passed to the actor."""
import copy
import math
import time

from .agent import run_agent
from .evaluation import call_matches, call_metrics
from .training import trajectory_examples


def train_task(row, tools, catalog):
    """Separate initial actor input from the scorer's private gold calls."""
    if not row['id'].startswith('train_'):
        raise ValueError('RL is restricted to train samples')
    examples = trajectory_examples(row, tools, catalog)
    return {'id': row['id'], 'messages': examples[0]['messages'],
            'tools': copy.deepcopy(tools),
            'gold_calls': copy.deepcopy(row['label']['tool_calls'])}


def rollout(task, actor, session_factory, seconds=90, max_turns=8):
    """Exactly one new ToolEnv per training episode; never inspect its files."""
    if not math.isfinite(seconds) or seconds <= 0 or max_turns <= 0:
        raise ValueError('Invalid rollout limits')
    actor.reset_episode()
    started = time.monotonic()
    env = session_factory(task['id'])
    initial_hash = env.get_state_hash()
    prediction, trace = run_agent(env, actor.decide, task['messages'], task['tools'],
                                  started + seconds, max_turns=max_turns)
    return {'prediction': prediction, 'trace': trace, 'initial_hash': initial_hash,
            'seconds': time.monotonic()-started, 'segments': copy.deepcopy(actor.segments)}


def score_episode(episode, gold, tools):
    """Diagnostic reward, NOT a state-success verifier or official score.

    Dense credit uses exact typed arguments (catalog fuzzy fields excepted).
    Unmatched writes are separately penalized. Efficiency never offsets an
    incorrect trajectory; only exact, clean call-set completion earns it.
    """
    specs = {tool['name']: tool for tool in tools}
    pred = episode['prediction']['tool_calls']
    trace = episode['trace']
    metrics = call_metrics(pred, gold, specs)
    # Match write multisets separately, so duplicate writes are not forgiven.
    remaining = list(gold)
    wrong_writes = 0
    for call in pred:
        match = next((i for i, target in enumerate(remaining)
                      if call_matches(call, target, specs)), None)
        if match is not None:
            remaining.pop(match)
        elif specs.get(call['name'], {}).get('kind') == 'write':
            wrong_writes += 1
    failed_writes = sum(entry['executed'] and entry['status'] != 'succeeded'
                        and specs.get(entry['call']['name'], {}).get('kind') == 'write'
                        for entry in trace['trace'])
    clean = (trace['termination'] in ('finished', 'clarification')
             and not trace['protocol_errors'] and not trace['validation_rejections']
             and not failed_writes)
    exact_proxy = metrics['matched'] == len(pred) == len(gold) and clean
    # Small bounded shaping; no name-only or per-call format bonus.
    dense = 0.5 * metrics['f1']
    penalties = (min(2, wrong_writes) + 0.5*min(2, failed_writes)
                 + 0.1*min(2, trace['protocol_errors'])
                 + 0.1*min(2, trace['validation_rejections']))
    efficiency = 0.1/(1+trace['decisions']) if exact_proxy else 0.0
    total = 2.0*exact_proxy + dense + efficiency - penalties
    return {'total': total, 'diagnostic_f1': metrics['f1'],
            'exact_call_proxy': exact_proxy, 'task_success_verified': False,
            'wrong_writes': wrong_writes, 'failed_writes': failed_writes,
            'efficiency': efficiency, 'penalties': penalties}


def group_advantages(rewards):
    """Population-std normalization; constant groups carry zero signal."""
    if len(rewards) < 2 or any(not math.isfinite(r) for r in rewards):
        raise ValueError('Need at least two finite rewards')
    mean = sum(rewards)/len(rewards)
    std = math.sqrt(sum((r-mean)**2 for r in rewards)/len(rewards))
    return [(r-mean)/std if std > 1e-8 else 0.0 for r in rewards]


def policy_loss(logps, advantage, token_count, group_size):
    """One on-policy update, no replay or PPO epochs, KL coefficient zero.

    exp(logp-stop_gradient(logp)) has value one and the correct score-function
    gradient at the sampling policy. Normalize over all assistant tokens of
    each episode; environment observations never receive gradient.
    """
    return -(logps-logps.detach()).exp().sum()*advantage/(token_count*group_size)
