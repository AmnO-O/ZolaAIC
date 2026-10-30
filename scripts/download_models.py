"""Development/build-time download only. Final inference never invokes this."""
import json
import argparse
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download
    parser=argparse.ArgumentParser()
    parser.add_argument('--component',choices=['all','llm','asr'],default='all')
    parser.add_argument('--config',default='configs/baseline.json')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    for component in (('llm','asr') if args.component=='all' else (args.component,)):
        spec=config[component]
        if spec.get('files'):
            snapshot_download(spec['id'],revision=spec['revision'],local_dir=spec.get('local_dir', 'models/'+component),
                              allow_patterns=spec['files'],max_workers=1)
            continue
        ignore=['*.onnx','*.h5','*.msgpack','*.ot']
        if component=='llm': ignore.append('*.bin')
        snapshot_download(spec['id'],revision=spec['revision'],local_dir=spec.get('local_dir', 'models/'+component),
                          ignore_patterns=ignore)


if __name__=='__main__': main()
