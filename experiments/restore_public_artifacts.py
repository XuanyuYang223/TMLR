"""Restore checksum-verified release archives without overwriting local changes."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from .permworld_combinations import sha


def restore(directory):
    expected_assets = json.loads(Path('docs/artifact_inventory.json').read_text())['assets']
    root = Path.cwd().resolve()
    restored = 0
    for asset in expected_assets:
        archive = directory / asset['asset']
        assert sha(archive) == asset['sha256'], f'Archive checksum mismatch: {archive}'
        inventory_file = directory / asset['asset'].replace('.tar.gz', '.inventory.json')
        assert sha(inventory_file) == asset['inventory_sha256']
        expected = json.loads(inventory_file.read_text())['files']
        seen = set()
        with tarfile.open(archive, 'r:gz') as tf:
            for member in tf:
                target = (root / member.name).resolve()
                assert target.is_relative_to(root) and member.name.startswith('results/')
                assert member.isfile() and member.name in expected
                assert member.name not in seen
                seen.add(member.name)
                if target.exists():
                    if sha(target) != expected[member.name]:
                        raise FileExistsError(f'Local file differs from archived snapshot: {member.name}')
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
                assert sha(target) == expected[member.name]
                restored += 1
        assert seen == set(expected)
        print(json.dumps({'verified_archive': archive.name, 'files': len(seen)}), flush=True)
    print(json.dumps({'status': 'restored', 'new_files': restored}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('downloads'))
    restore(parser.parse_args().directory)
