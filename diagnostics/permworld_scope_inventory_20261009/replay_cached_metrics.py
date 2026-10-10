"""Replay frozen metrics from archived features in a fresh output directory.

The binding shim replaces machine-specific inventory paths only. The original
hash-frozen evaluator/metric code executes unchanged; no source optimizer or
training entry point is invoked. PermWorld/k replay requires no source weights.
F5 replay checks ordinary saved source encoders and historical activations.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types

SOURCE=Path(__file__).resolve().parent
REPO=SOURCE.parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def run(output,upstream,stages):
    output=output.resolve();upstream=upstream.resolve()
    if output.exists():raise FileExistsError('Use a new output directory; archived results are never overwritten.')
    assert (upstream/'src/neurips_permutations/math_ops.py').exists(),'Clone the pinned neurips upstream first.'
    expected=json.loads((SOURCE/'protocol_freeze.json').read_text())
    assert sha(SOURCE/'evaluate.py')==expected['evaluation_code_sha256']
    assert sha(SOURCE/'metrics.py')==expected['metrics_code_sha256']
    assert sha(SOURCE/'relations.py')==expected['relation_code_sha256']
    assert sha(upstream/'src/neurips_permutations/math_ops.py')==expected['model_task_definitions_sha256']
    output.mkdir(parents=True)
    for name in ['metric_protocol.json','protocol_freeze.json','metrics.py','relations.py',
                 'evaluation_inputs.npz','evaluation_pairs.json','training_inventory.csv','summary_protocol.json']:
        shutil.copyfile(SOURCE/name,output/name)
    (output/'features').symlink_to(SOURCE/'features',target_is_directory=True)
    sys.path[:0]=[str(SOURCE),str(REPO),str(upstream/'src')]
    # Do not import the historical inventory constructor: its absolute repository
    # roots are provenance of the original run, not requirements on a new machine.
    bindings=types.ModuleType('inventory');bindings.ROOT=output;bindings.WS=REPO;bindings.NR=upstream
    bindings.sha=sha;bindings.js=lambda p:json.loads(Path(p).read_text())
    def read(path):
        rows=list(csv.DictReader(Path(path).open()))
        if Path(path).name=='model_inventory.csv':
            for r in rows:
                if r.get('family')!='k_series':continue
                relative=Path(r['activation_cache']).relative_to('/home/yangx/neurips')
                r['activation_cache']=str(SOURCE/'upstream_k_caches'/relative)
        return rows
    def write(name,rows,fields=None):
        fields=fields or list(dict.fromkeys(k for r in rows for k in r))
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    def save(name,obj):(output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    bindings.csv_read=read;bindings.csv_write=write;bindings.save=save;sys.modules['inventory']=bindings
    spec=importlib.util.spec_from_file_location('evaluate',SOURCE/'evaluate.py');evaluator=importlib.util.module_from_spec(spec)
    sys.modules['evaluate']=evaluator;spec.loader.exec_module(evaluator)
    if 'permworld' in stages:evaluator.evaluate_permutations()
    if 'f5' in stages:evaluator.evaluate_field()
    if 'k' in stages:
        spec=importlib.util.spec_from_file_location('cached_k_replay',SOURCE/'evaluate_k.py');module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module);module.run()
    # Compare row identities and numeric scientific outputs, not wall-clock audit fields.
    files={'permworld':'permutation_metric_results.csv','f5':'field_metric_results.csv','k':'k_metric_results.csv'}
    verification={}
    import math
    for stage in stages:
        name=files[stage];old=list(csv.DictReader((SOURCE/name).open()));new=list(csv.DictReader((output/name).open()))
        assert len(old)==len(new);max_error=0.;numeric_cells=0
        tolerance=2e-5 if stage=='f5' else 1e-10
        for a,b in zip(old,new):
            assert a.keys()==b.keys()
            for key in a:
                if a[key]==b[key]:continue
                try:aa=float(a[key]);bb=float(b[key])
                except ValueError:raise AssertionError((name,key,a[key],b[key]))
                assert math.isfinite(aa) and math.isfinite(bb)
                error=abs(aa-bb);assert error<tolerance,(name,key,error)
                max_error=max(max_error,error);numeric_cells+=1
        verification[stage]=dict(rows=len(old),maximum_numeric_difference=max_error,tolerance=tolerance,differing_numeric_cells=numeric_cells)
    save('publication_replay_audit.json',dict(status='passed',new_source_training=0,archived_results_changed=False,
        evaluator_sha256=sha(SOURCE/'evaluate.py'),metrics_sha256=sha(SOURCE/'metrics.py'),comparisons=verification))
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=REPO/'external/neurips')
    p.add_argument('--stage',choices=['all','permworld','f5','k'],default='all')
    a=p.parse_args();run(a.output,a.upstream,['permworld','f5','k'] if a.stage=='all' else [a.stage])
