"""Read public inputs only; environment loading belongs to zaic_tools."""
import importlib
import json
import sys
import tempfile
import os
from pathlib import Path


class SplitInput:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        catalogs = list(self.directory.glob('*_tools_catalog.json'))
        if len(catalogs) != 1:
            raise ValueError('Expected exactly one split tools catalog')
        self.split = catalogs[0].name.removesuffix('_tools_catalog.json')
        self.catalog = json.loads(catalogs[0].read_text(encoding='utf-8'))
        self._runtime_root = self.directory.parent
        self._mount_alias = None

    def records(self):
        with (self.directory / f'{self.split}.jsonl').open(encoding='utf-8') as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    # Deliberately exclude transcripts and labels from inference.
                    history = [{k: v for k, v in turn.items() if k != 'audio_text'}
                               for turn in row.get('history', [])]
                    yield {k: row[k] for k in ('id', 'audio', 'context')} | {'history': history}

    def tools(self, item):
        allowed = {name for app in item['context']['installed_apps'].values()
                   for name in app['tools']} | {'sys.ask_missing_info'}
        return [dict(tool)
                for app in self.catalog['apps'] for tool in app['tools']
                if tool['name'] in allowed]

    def audio_path(self, relative):
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory / 'audio'):
            raise ValueError('Audio must be inside the split audio directory')
        return path

    def runtime(self):
        existing = sys.modules.get('zaic_tools')
        expected = self.directory / 'zaic_tools' / '__init__.py'
        if existing and Path(existing.__file__).resolve() != expected:
            raise RuntimeError('Run each split in a separate process')
        sys.path.insert(0, str(self.directory))
        return importlib.import_module('zaic_tools')

    def session(self, sample_id):
        if self.directory.name != self.split:
            if self._mount_alias is None:
                # Alias the mount root only. Do not inspect/copy environment files.
                # The supplied ToolEnv alone follows this alias to load its data.
                self._mount_alias = tempfile.TemporaryDirectory(prefix='lifemate-mount-')
                self._runtime_root = Path(self._mount_alias.name)
                os.symlink(self.directory, self._runtime_root/self.split, target_is_directory=True)
        return self.runtime().ToolEnv(self.split, sample_id, root=self._runtime_root)
