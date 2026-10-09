"""Create downloadable core artifacts, dereferencing historical absolute links."""
import argparse
from hashlib import sha256
import json
import os
import subprocess
from pathlib import Path
import tarfile

from .longrun_engine import atomic_json
from .permworld_combinations import sha


def build(destination, only=None):
    destination.mkdir(parents=True, exist_ok=True)
    groups = {
        'mechanism-confirmations': [
            'results/operator_capacity_confirmation',
            'results/operator_mechanism_diagnostic',
            'results/final_mechanism_confirmation'],
        'directional-loss-confirmation': ['results/readout_null_confirmation'],
        'initial-cross-domain': ['results/algebra_relation_v3'],
        'reviewer-controls': ['results/null_space_review_controls',
                             'results/null_space_review_audit',
                             'results/null_space_review_reporting'],
        'reviewer-revision-diagnostics': [
            'results/reviewer_revision_diagnostics',
            'results/reviewer_revision_diagnostics_v2',
            'results/reviewer_revision_audit',
            'results/reviewer_revision_reporting'],
        'fresh-confirmation-f17': ['results/reviewer_fresh_confirmation',
                                   'results/reviewer_fresh_audit',
                                   'results/reviewer_fresh_reporting',
                                   'results/reviewer_fresh_reporting_v2',
                                   'results/reviewer_fresh_presentation_history'],
    }
    # Dataset aliases must be followed even when their targets are outside a
    # study's own directory. Keep them portable and preserve logical paths.
    tracked = subprocess.check_output(['git', 'ls-files', '--stage'], text=True)
    groups['historical-inputs'] = [line.split('\t', 1)[1]
        for line in tracked.splitlines() if line.startswith('120000 ')
        and Path(line.split('\t', 1)[1]).is_dir()]
    if only:
        unknown = set(only) - groups.keys()
        if unknown:
            raise ValueError(f'Unknown artifact groups: {sorted(unknown)}')
        groups = {name: folders for name, folders in groups.items() if name in only}
    # The six historical test shards are necessary for the final split audit.
    config = json.loads(Path('configs/final_mechanism_confirmation.json').read_text())
    extra = [Path(p) for p in config['excluded_test_datasets']]
    inventories = []
    for name, folders in groups.items():
        for folder in folders:
            if not Path(folder).is_dir():
                raise FileNotFoundError(folder)
        if name == 'fresh-confirmation-f17':
            result = json.loads(Path(folders[0], 'results.json').read_text())
            if result['status'] != 'complete':
                raise ValueError('Cannot publish an incomplete confirmation')
        entries = {Path(directory) / name for folder in folders
                   for directory, _, names in os.walk(folder, followlinks=True)
                   for name in names if (Path(directory) / name).is_file()
                   and '__pycache__' not in Path(directory).parts}
        if name == 'mechanism-confirmations':
            entries.update(extra)
        if name == 'historical-inputs':
            entries = {p for p in entries if p.suffix in ['.npz', '.json']}
        entries = sorted(entries)
        archive = destination / (name + '.tar.gz')
        with tarfile.open(archive, 'w:gz', compresslevel=3, dereference=True) as tf:
            for p in entries:
                tf.add(p, arcname=str(p), recursive=False)
        inventory = {'asset': archive.name, 'sha256': sha(archive),
                     'bytes': archive.stat().st_size,
                     'files': {str(p): sha(p) for p in entries},
                     'absolute_symlinks_dereferenced': True}
        assert inventory['bytes'] < 2 * 1024**3
        atomic_json(destination / (name + '.inventory.json'), inventory)
        inventories.append({k: v for k, v in inventory.items() if k != 'files'})
        print(json.dumps({'archive_created': archive.name,
                          'bytes': inventory['bytes'], 'files': len(entries)}), flush=True)
    atomic_json(destination / 'release_assets.json', {'assets': inventories})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('.publication/releases'))
    parser.add_argument('--only', nargs='+', help='Build only these archive groups.')
    args = parser.parse_args()
    build(args.output, args.only)
