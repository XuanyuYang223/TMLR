"""Reproduce/resume the completed structural-assay pipeline.

The native frozen program recorded action_names as a JSON array but builds
them as a tuple. Normalize only that metadata constant for replay, preserving
the original source-file hashes and the exact action order/definitions.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .longrun_engine import atomic_json


def replay_sources():
    from . import native_algebra_structure, field_algebra_structure
    original = native_algebra_structure.ACTION_NAMES
    try:
        native_algebra_structure.ACTION_NAMES = list(original)
        native_algebra_structure.run()
    finally:
        native_algebra_structure.ACTION_NAMES = original
    field_algebra_structure.run()


def run(source_replay_only=False):
    replay_sources()
    if source_replay_only: return
    from . import (algebra_structure_controls, algebra_structure_direct, algebra_structure_rank,
        algebra_structure_new_groups, algebra_structure_analysis,
        algebra_structure_verify, algebra_structure_supplement_verify)
    for module in (algebra_structure_controls, algebra_structure_direct, algebra_structure_rank,
                   algebra_structure_new_groups, algebra_structure_analysis,
                   algebra_structure_verify, algebra_structure_supplement_verify):
        module.run()
    root=Path(json.loads(Path('configs/algebra_structure.json').read_text())['output'])
    atomic_json(root/'completion.json', {'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),
        'primary_verification':json.loads((root/'verification.json').read_text())['status'],
        'supplementary_verification':json.loads((root/'supplementary_verification.json').read_text())['status'],
        'research_scope':'one completed structural assay; further source-seed replication remains future work'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-replay-only',action='store_true')
    run(parser.parse_args().source_replay_only)
