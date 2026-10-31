"""Single-GPU SFT session boundaries and strict own-checkpoint resume identity."""
import json
from pathlib import Path
import time


def verify_resume(path, identity):
    path=Path(path)
    expected=json.loads((path/'training_identity.json').read_text(encoding='utf-8'))
    if expected != identity:
        raise ValueError('Checkpoint data/model/training identity differs; refuse incompatible resume')
    required=('adapter_config.json','adapter_model.safetensors','trainer_state.json',
              'optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth')
    missing=[name for name in required if not (path/name).is_file()]
    if missing:raise ValueError('Incomplete resumable checkpoint: '+', '.join(missing))
    return json.loads((path/'trainer_state.json').read_text(encoding='utf-8'))


def session_callback(identity, seconds=0, clock=time.monotonic, min_session_steps=2):
    from transformers import TrainerCallback

    class SessionCallback(TrainerCallback):
        def __init__(self):
            self.started=None
            self.time_limit_reached=False

        def on_train_begin(self,args,state,control,**kwargs):
            self.started=clock()
            self.start_step=state.global_step

        def on_step_end(self,args,state,control,**kwargs):
            elapsed=clock()-self.started
            if seconds and elapsed>=seconds and state.global_step-self.start_step>=min_session_steps:
                self.time_limit_reached=True
                control.should_training_stop=True
                control.should_save=True
            if getattr(state,'is_world_process_zero',True):
                progress={'global_step':state.global_step,'target_steps':state.max_steps,
                          'epoch':state.epoch,'session_seconds':elapsed,
                          'time_limit_reached':self.time_limit_reached}
                Path(args.output_dir).mkdir(parents=True,exist_ok=True)
                (Path(args.output_dir)/'progress.json').write_text(json.dumps(progress,indent=2))
            return control

        def on_save(self,args,state,control,**kwargs):
            if getattr(state,'is_world_process_zero',True):
                path=Path(args.output_dir)/f'checkpoint-{state.global_step}'
                path.mkdir(parents=True,exist_ok=True)
                (path/'training_identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
            return control

    return SessionCallback()
