"""Package existing read-only analysis artifacts; never trains a source model."""
from pathlib import Path
import csv
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
UPSTREAM=Path('/home/yangx/neurips')
TAG='permworld-existing-checkpoints-2026-10-09'
OUT=ROOT/'.release'
PREFIX=ROOT.relative_to(REPO)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def pack(name,files):
    path=OUT/name;inventory={};sizes={}
    with tarfile.open(path,'w',format=tarfile.PAX_FORMAT) as tf:
        for archive_name,source in sorted(files.items()):
            info=tarfile.TarInfo(archive_name);info.size=source.stat().st_size;info.mode=0o644;info.mtime=0
            with source.open('rb') as f:tf.addfile(info,f)
            inventory[archive_name]=sha(source);sizes[archive_name]=source.stat().st_size
    assert path.stat().st_size<1_950_000_000,(name,path.stat().st_size)
    data=dict(asset=name,sha256=sha(path),bytes=path.stat().st_size,files=inventory,file_bytes=sizes)
    side=OUT/(name+'.inventory.json');side.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print('Packaged',name,path.stat().st_size,'bytes',len(files),'files',flush=True)
    return dict(asset=name,sha256=data['sha256'],bytes=data['bytes'],files=len(files),
        inventory_asset=side.name,inventory_sha256=sha(side),inventory_bytes=side.stat().st_size,
        download_url=f'https://github.com/XuanyuYang223/2026TMLR/releases/download/{TAG}/{name}',
        inventory_url=f'https://github.com/XuanyuYang223/2026TMLR/releases/download/{TAG}/{side.name}')


def run():
    OUT.mkdir(exist_ok=True);assets=[];group={};size=0
    for p in sorted((ROOT/'features').glob('*.npz')):
        meta=json.loads(p.with_suffix('.json').read_text());assert sha(p)==meta['archive_sha256']
        if size+p.stat().st_size>1_850_000_000:
            assets.append(pack(f'permworld-feature-cache-{len(assets)+1:02d}.tar',group));group={};size=0
        group[str(p.relative_to(REPO))]=p;size+=p.stat().st_size
    if group:assets.append(pack(f'permworld-feature-cache-{len(assets)+1:02d}.tar',group))
    support={str(PREFIX/'evaluation_inputs.npz'):ROOT/'evaluation_inputs.npz'}
    old=REPO/'diagnostics/permworld_cka_20261009'
    for p in old.iterdir():
        if p.is_file():support[str(p.relative_to(REPO))]=p
    rows=list(csv.DictReader((old/'model_inventory.csv').open()))
    for row in rows:
        if row['family']!='k_series':continue
        p=Path(row['activation_cache']);support[str(PREFIX/'upstream_k_caches'/p.relative_to(UPSTREAM))]=p
    field=REPO/'results/field_symmetry'
    for p in field.glob('*_step*_features.npy'):support[str(p.relative_to(REPO))]=p
    for p in (field/'checkpoints').glob('*.pt'):support[str(p.relative_to(REPO))]=p
    for p in field.glob('*.json'):support[str(p.relative_to(REPO))]=p
    # Original input archive permits provenance checks of the mapping-fit/test inputs.
    p=REPO/'results/specialist_cka_controls/dataset.npz';support[str(p.relative_to(REPO))]=p
    assets.append(pack('permworld-analysis-support.tar',support))
    data=dict(format='permworld-existing-model-publication/v1',repository='XuanyuYang223/2026TMLR',release_tag=TAG,
        base_repository_commit='1073dcb3be4db90ec23476bab8f503c9e00f364b',
        upstream_commit='74f0de2017115f06f11b7285366b66e7e8431d30',
        new_source_training=0,scientific_audit_sha256=sha(ROOT/'audit.json'),
        scientific_protocol_sha256=sha(ROOT/'metric_protocol.json'),assets=assets,
        scope='Git stores exact readable results/code/metadata; new separate Release stores all164 feature archives, inputs and metric-replay prerequisites. No PermWorld source checkpoints duplicated; inventory/source hash evidence retained.',
        old_F17_release='reviewer-revision-2026-10-09 remains unchanged and latest',
        interpretation='Exploratory existing-checkpoint results; C reconstructed PermWorld initialization, B historical F5 encoder snapshots. Not independent confirmation.')
    (ROOT/'publication_manifest.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Total binary assets',len(assets),'bytes',sum(x['bytes'] for x in assets),flush=True)


if __name__=='__main__':run()
