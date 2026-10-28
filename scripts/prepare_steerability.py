"""Vendor unmodified Python sources from the user-selected Steerability checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import io


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default=str(root.parent/'steerability'))
    args=parser.parse_args()
    source=Path(args.source).resolve()
    if not (source/'steerability/__init__.py').is_file():raise ValueError('Steerability source not found')
    destination=root/'vendor/steerability'
    assert destination.resolve().is_relative_to(root.resolve())
    archive=root/'artifacts/steerability_source.tar.xz'
    archive.parent.mkdir(exist_ok=True)
    paths=sorted((source/'steerability').rglob('*.py'))+[source/'LICENSE',source/'NOTICE']
    manifest={}
    with tarfile.open(archive,'w:xz') as bundle:
        for path in paths:
            if not path.resolve().is_relative_to(source):raise ValueError('Source link escapes checkout')
            name=path.relative_to(source).as_posix()
            data=path.read_bytes()
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
            bundle.addfile(info,io.BytesIO(data))
            target=destination/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
            manifest[name]=hashlib.sha256(data).hexdigest()
    (destination/'source-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Packaged {len(paths)} unmodified source/license files; archive {archive.stat().st_size:,} bytes')


if __name__=='__main__':main()
