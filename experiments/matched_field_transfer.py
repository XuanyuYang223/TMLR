"""Apply the frozen behavior protocol using the audited constrained bases."""
import json
from pathlib import Path

from . import field_symmetry_transfer
from .matched_field import controlled_world
from .permworld_combinations import sha


def run():
    config_path = 'configs/field_matched_support_transfer.json'
    config = json.loads(Path(config_path).read_text())
    source = json.loads(Path(config['source_config']).read_text())
    assert source['world_provider_sha256'] == sha('experiments/matched_field.py')
    field_symmetry_transfer.world = controlled_world
    field_symmetry_transfer.run(config_path)


if __name__ == '__main__': run()
