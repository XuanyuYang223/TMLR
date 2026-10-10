"""Verify and restore this diagnostic's Release assets without replacing changed files."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def restore(directory,destination,verify_only=False):
    manifest=json.loads((ROOT/'publication_manifest.json').read_text())
    destination=destination.resolve();file_count=0
    for asset in manifest['assets']:
        archive=directory/asset['asset'];side=directory/asset['inventory_asset']
        assert sha(archive)==asset['sha256'],archive
        assert sha(side)==asset['inventory_sha256'],side
        inventory=json.loads(side.read_text());expected=inventory['files'];seen=set()
        with tarfile.open(archive,'r:') as tf:
            for member in tf:
                name=PurePosixPath(member.name)
                assert member.isfile() and not name.is_absolute() and '..' not in name.parts,member.name
                assert member.name in expected and member.name not in seen,member.name
                target=(destination/Path(*name.parts)).resolve()
                assert target.is_relative_to(destination),target
                existing=target.exists();same=sha(target)==expected[member.name] if existing else False
                if existing and not same:raise FileExistsError(f'Refusing to replace changed local artifact: {target}')
                write=not verify_only and not existing
                if write:target.parent.mkdir(parents=True,exist_ok=True)
                stream=tf.extractfile(member);digest=hashlib.sha256()
                output=target.with_name(target.name+'.restore-part') if write else None
                f=output.open('xb') if write else None
                try:
                    for b in iter(lambda:stream.read(1024*1024),b''):
                        digest.update(b)
                        if f:f.write(b)
                    assert digest.hexdigest()==expected[member.name],member.name
                    if f:f.close();f=None;output.rename(target)
                finally:
                    if f:f.close()
                    if output and output.exists():output.unlink()
                seen.add(member.name);file_count+=1
        assert seen==set(expected),archive
        print(json.dumps({'verified_asset':asset['asset'],'files':len(seen),'verify_only':verify_only}),flush=True)
    return file_count


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archives',type=Path,required=True)
    p.add_argument('--destination',type=Path,default=ROOT.parents[1]);p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();print('Verified/restored',restore(a.archives,a.destination,a.verify_only),'files')
