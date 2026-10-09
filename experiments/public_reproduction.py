"""Portable integrity checks and read-only replay of the sealed confirmation."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile

from .permworld_combinations import sha


HISTORICAL_ROOT = Path('/home/yangx/ICML')


def portable_path(value):
    p = Path(value)
    if p.is_absolute() and p.is_relative_to(HISTORICAL_ROOT):
        return Path.cwd() / p.relative_to(HISTORICAL_ROOT)
    return p


def check_manifest(file, allow_missing=False):
    data = json.loads(Path(file).read_text())
    checked, missing = [], []
    for filename, expected in data['artifact_sha256'].items():
        p = portable_path(filename)
        if not p.exists():
            missing.append(filename)
        else:
            assert sha(p) == expected, f'Hash mismatch: {filename}'
            checked.append(filename)
    if missing and not allow_missing:
        raise FileNotFoundError(f'{len(missing)} archived files missing; download the listed release archives first.')
    return {'manifest': str(file), 'verified_files': len(checked),
            'missing_files': missing, 'hash_mismatches': 0}


def replay():
    # Copying and atomic replacement keep the original timestamped audit immutable.
    from . import final_mechanism_verify as verifier
    with tempfile.TemporaryDirectory(prefix='relation-confirmation-replay-') as folder:
        scratch = Path(folder) / 'final'
        shutil.copytree('results/final_mechanism_confirmation', scratch, symlinks=False)
        old_root, old_sha = verifier.ROOT, verifier.sha
        verifier.ROOT = scratch
        verifier.sha = lambda p: sha(portable_path(p))
        capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(capture):
                verifier.run()
            audit = json.loads((scratch / 'independent_verification.json').read_text())
        finally:
            verifier.ROOT, verifier.sha = old_root, old_sha
    return audit


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay-final', action='store_true')
    parser.add_argument('--allow-missing', action='store_true',
                        help='Inventory omitted large binaries without treating them as available.')
    args = parser.parse_args()
    manifests = ['results/' + name + '/delivery.json' for name in [
        'algebra_relation_review', 'readout_null_confirmation',
        'operator_capacity_confirmation', 'operator_mechanism_diagnostic',
        'final_mechanism_confirmation', 'null_space_review_controls',
        'reviewer_revision_diagnostics_v2', 'reviewer_revision_reporting',
        'reviewer_fresh_confirmation', 'reviewer_fresh_reporting',
        'reviewer_fresh_reporting_v2']]
    audits = [check_manifest(p, args.allow_missing) for p in manifests]
    output = {'status': 'passed' if not any(a['missing_files'] for a in audits)
              else 'available_files_verified_with_explicit_missing_inventory',
              'manifest_checks': audits, 'sealed_files_modified': False}
    if args.replay_final:
        output['independent_final_replay'] = replay()
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
