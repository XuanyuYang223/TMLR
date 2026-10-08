"""Run the archived training stages in a separate, explicitly chosen output."""
import argparse
import json
from pathlib import Path

from .longrun_engine import atomic_json


def run(output, stage):
    from . import final_mechanism_train as driver
    base = Path.cwd().resolve()
    output = output.resolve()
    for name in ['algebra_relation_review', 'readout_null_confirmation',
                 'operator_capacity_confirmation', 'operator_mechanism_diagnostic',
                 'final_mechanism_confirmation', 'null_space_review_controls',
                 'null_space_review_audit', 'null_space_review_reporting']:
        protected = base/'results'/name
        if output==protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('Choose a separate output outside the archived study directories.')
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path('configs/final_mechanism_confirmation.json').read_text())
    config['output'] = str(output)
    file = output/'reproduction_config.json'
    if file.exists():
        assert json.loads(file.read_text())==config, 'Existing reproduction config differs.'
    else:
        assert not any(output.iterdir()), 'The new output must be empty.'
        atomic_json(file, config)
    driver.CONFIG = file
    driver.run(stage)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'sources', 'fits', 'evaluate', 'run'], default='prepare')
    args = parser.parse_args()
    run(args.output, args.stage)
