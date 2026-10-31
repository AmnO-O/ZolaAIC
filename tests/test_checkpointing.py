import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.checkpointing import verify_resume,session_callback


class CheckpointTests(unittest.TestCase):
    def test_step_boundary_save_and_identity(self):
        now=[0.]
        callback=session_callback({'data':'hash'},seconds=1,clock=lambda:now[0])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            args=SimpleNamespace(output_dir=directory)
            state=SimpleNamespace(global_step=10,max_steps=100,epoch=.1,is_world_process_zero=True)
            control=SimpleNamespace(should_save=False,should_training_stop=False)
            callback.on_train_begin(args,state,control)
            now[0]=2;state.global_step=11
            callback.on_step_end(args,state,control)
            self.assertFalse(control.should_training_stop)
            state.global_step=12
            callback.on_step_end(args,state,control)
            self.assertTrue(control.should_training_stop);self.assertTrue(control.should_save)
            callback.on_save(args,state,control)
            path=Path(directory)/'checkpoint-12'
            self.assertEqual(json.loads((path/'training_identity.json').read_text()),{'data':'hash'})
            self.assertTrue(json.loads((Path(directory)/'progress.json').read_text())['time_limit_reached'])

    def test_resume_refuses_different_or_incomplete_state(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            root=Path(directory)
            (root/'training_identity.json').write_text(json.dumps({'data':'same'}))
            with self.assertRaises(ValueError):verify_resume(root,{'data':'changed'})
            with self.assertRaises(ValueError):verify_resume(root,{'data':'same'})
            for name in ('adapter_config.json','adapter_model.safetensors','optimizer.pt',
                         'scheduler.pt','scaler.pt','rng_state.pth'):
                (root/name).write_bytes(b'fixture')
            (root/'trainer_state.json').write_text(json.dumps({'global_step':12}))
            self.assertEqual(verify_resume(root,{'data':'same'})['global_step'],12)


if __name__=='__main__':unittest.main()
