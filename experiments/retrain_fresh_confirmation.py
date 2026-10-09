"""Repeat the fixed new-field training in a separate, resumable output root."""
import argparse
import json
from pathlib import Path

from .longrun_engine import atomic_json
from . import reviewer_fresh_confirmation as runner


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['register', 'run'], default='run')
    args = parser.parse_args()
    c = json.loads(runner.CONFIG.read_text())
    output = args.output
    sealed = [Path(c['output']), Path('results/reviewer_revision_diagnostics_v2'),
              Path('results/final_mechanism_confirmation')]
    if any(output.resolve() == p.resolve() or output.resolve().is_relative_to(p.resolve()) for p in sealed):
        raise ValueError('Choose a separate output root; preserve all sealed studies.')
    if (output / 'test_opened.json').exists():
        raise ValueError('This reproduction has already opened its test; choose another output root.')
    c['output'] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    config = output / 'reproduction_config.json'
    if config.exists():
        assert json.loads(config.read_text()) == c
    else:
        atomic_json(config, c)
    runner.CONFIG = config
    c, root = runner.setup()
    if args.stage == 'run':
        for i in range(c['independent_units']):
            if not (root / f'u{i:02d}/unit_complete.json').exists():
                runner.train_unit(c, root, i)
        runner.evaluate(c, root)


if __name__ == '__main__':
    run()
