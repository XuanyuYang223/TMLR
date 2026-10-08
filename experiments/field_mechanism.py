"""Exploratory finite-field diagnostics: nuisance, spectrum and unseen states.

The generic categorical decoder never receives source or target coefficients.
Algebraic minimum subset size is an external prediction, not its input. The
explicit field-affine decoder remains a separately labeled privileged control.
"""
import hashlib
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import torch

from .algebra import composition_size, scenarios, world
from .analysis import write_csv
from .models import SourceModel
from .readout_followup import fit_field, lookup_predict


def categorical_decoder(train_codes, train_labels, test_codes, classes):
    """Choose the smallest consistent subset; ties favor support repetition."""
    selected = None
    for size in range(1, train_codes.shape[1]+1):
        valid = []
        for subset in combinations(range(train_codes.shape[1]), size):
            buckets = {}
            for code, label in zip(train_codes[:,subset], train_labels):
                buckets.setdefault(tuple(code), []).append(int(label))
            if all(len(set(labels))==1 for labels in buckets.values()):
                repeat_rows = sum(len(labels) for labels in buckets.values() if len(labels)>1)
                valid.append((repeat_rows, subset))
        if valid:
            selected = max(valid, key=lambda r: (r[0], tuple(-i for i in r[1])))[1]
            break
    if selected is None:
        selected = tuple(range(train_codes.shape[1]))
    predicted, seen = lookup_predict(train_codes[:,selected], train_labels, test_codes[:,selected], classes)
    return predicted, seen, selected


def spectral_decomposition(representation, latent, sources, p):
    width = representation.shape[1]
    means = np.zeros((p,p,p,width), dtype=np.float64)
    counts = np.zeros((p,p,p), dtype=np.int64)
    np.add.at(means, tuple(latent.T), representation)
    np.add.at(counts, tuple(latent.T), 1)
    assert np.all(counts==counts.flat[0])
    means /= counts[...,None]
    coefficients = np.fft.fftn(means, axes=(0,1,2))/p**3
    power = np.square(np.abs(coefficients)).sum(-1)
    power[0,0,0] = 0
    nuisance = float(np.square(representation-means[tuple(latent.T)]).sum(1).mean())
    total = float(np.square(representation-representation.mean(0)).sum(1).mean())
    assert np.isclose(power.sum()+nuisance, total, rtol=1e-10, atol=1e-10)
    fractions = {r: 0. for r in (1,2,3)}
    for frequency in product(range(p), repeat=3):
        if any(frequency):
            fractions[composition_size(sources, frequency, p)] += float(power[frequency])/total
    return power, {'total_centered_energy': total, 'nuisance_fraction': nuisance/total,
                   **{f'order_{r}_energy_fraction': v for r,v in fractions.items()}}, means


def state_splits(latent, labels, p, seed):
    """Target support/test latent states are disjoint, not only input rows."""
    states = np.array(list(product(range(p), repeat=3)))
    rows = {tuple(state): np.flatnonzero(np.all(latent==state, axis=1)) for state in states}
    first = np.array([rows[tuple(state)][0] for state in states])
    rng = np.random.default_rng(seed)
    test, available = [], []
    for category in range(p):
        choices = first[labels[first]==category].copy()
        rng.shuffle(choices)
        test.extend(choices[:5])
        available.append(choices[5:])
    test_ids = np.concatenate([rows[tuple(latent[i])] for i in test])
    supports = {b: np.concatenate([pool[:b//p] for pool in available]) for b in (10,25,50,100)}
    for support in supports.values():
        assert not {tuple(v) for v in latent[support]} & {tuple(v) for v in latent[test_ids]}
    return supports, test_ids


def run(output='results/field_mechanism'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    parents=[Path('results/pilot'),Path('results/replication')]
    metadata=[json.loads((p/'metadata.json').read_text()) for p in parents]
    config=metadata[0]['config']
    targets=np.array(json.loads((parents[0]/'algebra_audit.json').read_text())['targets'])
    parent_by_seed={s:p for p,m in zip(parents,metadata) for s in m['config']['model_seeds']}
    spectral_rows,target_spectral,behavior=[],[],[]
    for world_seed in config['world_seeds']:
        _,latent,encoded,_=world(config['p'],config['dimension'],world_seed)
        target_labels=latent@targets.T%config['p']
        for model_seed,parent in parent_by_seed.items():
            for group in scenarios():
                run_id=f"{group['id']}_w{world_seed}_m{model_seed}"
                representation=np.load(parent/f'{run_id}_features.npy').astype(np.float64)
                power,statistics,means=spectral_decomposition(representation,latent,group['sources'],config['p'])
                spectral_rows.append({'scenario':group['id'],'family':group['family'],'world_seed':world_seed,'model_seed':model_seed,**statistics})
                model=SourceModel(encoded.shape[1],config['hidden'],config['features'],config['p'])
                model.load_state_dict(torch.load(parent/'checkpoints'/f'{run_id}.pt',weights_only=True,map_location='cpu'))
                with torch.no_grad():codes=model(torch.tensor(encoded)).argmax(-1).numpy()
                order=np.random.default_rng(world_seed+40000).permutation(4)
                assert np.array_equal(codes,(latent@group['sources'].T%config['p'])[:,order])
                for target_id,target in enumerate(targets):
                    minimum=composition_size(group['sources'],target,config['p'])
                    target_power=sum(power[tuple(k*target%config['p'])] for k in range(1,config['p']))
                    common={'scenario':group['id'],'family':group['family'],'world_seed':world_seed,'model_seed':model_seed,
                            'target_id':target_id,'minimum_source_subset':minimum}
                    target_spectral.append({**common,'target_direction_energy_fraction':float(target_power/statistics['total_centered_energy'])})
                    supports,test=state_splits(latent,target_labels[:,target_id],config['p'],202610050+world_seed*100+target_id)
                    for budget,train in supports.items():
                        predicted,seen,selected=categorical_decoder(codes[train],target_labels[train,target_id],codes[test],config['p'])
                        lookup,full_seen=lookup_predict(codes[train],target_labels[train,target_id],codes[test],config['p'])
                        design=np.column_stack([codes,np.ones(len(codes),dtype=np.int64)])
                        coefficients,rank=fit_field(design[train],target_labels[train,target_id],config['p'])
                        field=np.full(len(test),0) if coefficients is None else design[test]@coefficients%config['p']
                        truth=target_labels[test,target_id]
                        behavior.append({**common,'budget':budget,'categorical_accuracy':float(np.mean(predicted==truth)),
                                         'chosen_subset_size':len(selected),'partial_tuple_seen_fraction':float(seen.mean()),
                                         'full_tuple_lookup_accuracy':float(np.mean(lookup==truth)),'full_tuple_seen_fraction':float(full_seen.mean()),
                                         'field_affine_accuracy':float(np.mean(field==truth)),'field_fit_rank':rank})
    write_csv(output/'spectral_models.csv',spectral_rows)
    write_csv(output/'target_spectra.csv',target_spectral)
    write_csv(output/'state_holdout_endpoints.csv',behavior)
    summary=[]
    for minimum in sorted({r['minimum_source_subset'] for r in behavior}):
        for budget in (10,25,50,100):
            rows=[r for r in behavior if r['minimum_source_subset']==minimum and r['budget']==budget]
            summary.append({'minimum_source_subset':minimum,'budget':budget,'cells':len(rows),
                            **{key:float(np.mean([r[key] for r in rows])) for key in ('categorical_accuracy','chosen_subset_size','partial_tuple_seen_fraction','full_tuple_lookup_accuracy','field_affine_accuracy')}})
    result={'status':'exploratory mechanistic diagnostics','source_models':len(spectral_rows),'state_holdout_endpoints':len(behavior),
            'summary':summary,'mean_nuisance_fraction':float(np.mean([r['nuisance_fraction'] for r in spectral_rows])),
            'policy':'generic categorical subset decoder uses support codes and labels only; algebra predicts minimum subset size externally',
            'limitations':['Source pretraining exposed every input and latent state.','State-heldout target labels differ from original input-row-heldout protocol.','F5 affine fitting is a privileged mathematical-prior control.','No new training algorithm or causal CKA claim.'],
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'parent_fingerprints':[m['fingerprint'] for m in metadata]}
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
