"""Frozen-candidate replication with new source seeds and a disjoint data world."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .longrun_engine import atomic_json, train_job
from .longrun_transfer import make_model
from .native_length_baseline import smooth_prediction, majority_predictions
from .native_target_repeat import fit_weights, apply_weights
from .native_transfer_prediction import label_statistics, ridge_predict
from .native_prediction_sensitivity import corrected_math
from .native_transform_audit import certificates, OPERATORS, token_transform
from .analysis import linear_cka
from .permworld_combinations_report import within_length_center
from .permutation_audit import IDENTITIES
from .permworld_combinations import answer_logits, select_groups, sha
from .six_hour_report import write_rows


VARIANTS = ('source_learning', 'source_learning_plus_relations',
            'source_learning_and_label_stats', 'source_learning_label_stats_plus_relations')
MODES = ('candidate_shared', 'shared', 'condition')
CORE_PATHS = ('experiments/native_confirmation.py', 'experiments/longrun_engine.py',
              'experiments/native_confirmation_analysis.py',
              'experiments/longrun_transfer.py', 'experiments/longrun_attention.py',
              'experiments/permworld_combinations.py', 'experiments/native_target_repeat.py',
              'experiments/native_transfer_prediction.py', 'experiments/native_transform_audit.py',
              'experiments/native_prediction_sensitivity.py', 'experiments/native_length_baseline.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def paths():
    plan = json.loads(Path('configs/native_confirmation.json').read_text())
    config = json.loads(Path(plan['base_config']).read_text())
    config.update(data_seed=plan['data_seed'], examples_per_length=plan['examples_per_length'],
                  pretrain_steps=plan['source_steps'], record_every=plan['record_every'],
                  examples_per_task_per_step=plan['examples_per_task_per_step'],
                  learning_rate=plan['source_learning_rate'])
    return plan, config, Path(plan['output'])


def setup(config):
    sys.path.insert(0, str(Path(config['repository'])/'src'))
    from neurips_permutations.math_ops import PROPERTY32_TASK_NAMES, PROPERTY_FUNCTIONS
    from neurips_permutations.passage import TOKEN_TO_ID, one_line_tokens
    assert all(TOKEN_TO_ID[f'{i:02d}'] == i for i in range(31))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    return list(PROPERTY32_TASK_NAMES), PROPERTY_FUNCTIONS, TOKEN_TO_ID, one_line_tokens


def jobs(plan, groups):
    candidate = next(g for g in groups if g['id'] == plan['candidate_group'])
    result = []
    for seed in plan['model_seeds']:
        result.append({'phase': 'multi', 'id': candidate['id'], 'seed': seed,
                       'tasks': candidate['tasks'], 'steps': plan['source_steps']})
        result.extend({'phase': 'single', 'id': t, 'seed': seed, 'tasks': [t],
                       'steps': plan['source_steps']} for t in plan['single_tasks'])
    result.extend({'phase': 'multi', 'id': g['id'], 'seed': seed,
                   'tasks': g['tasks'], 'steps': plan['source_steps']}
                  for seed in plan['prediction_seeds'] for g in groups
                  if g['id'] != candidate['id'])
    assert len(result) == 46
    assert len({(r['phase'], r['id'], r['seed']) for r in result}) == len(result)
    return result


def permutation_row(row, n):
    return tuple(map(int, row[4+2*np.arange(n)]))


def feature_vector(row, variant):
    baseline = row['baseline']
    # Old baseline positions 7:10 are the three target indicators. Old seed
    # indicators 10:12 cannot apply to new seeds and are excluded everywhere.
    values = baseline[:2]+baseline[7:10]
    if variant in ('source_learning_and_label_stats', 'source_learning_label_stats_plus_relations'):
        values += baseline[2:7]
    if variant in ('source_learning_plus_relations', 'source_learning_label_stats_plus_relations'):
        values += corrected_math(row)
    return values


def prepare_weights(plan, root):
    old = Path('results/six_hour_session')
    features = json.loads((old/'transfer_prediction/features.json').read_text())
    gains_by_pool = []
    provenance = {str(old/'transfer_prediction/features.json'): sha(old/'transfer_prediction/features.json')}
    for pool in (old, Path('results/native_target_repeat')):
        records = [json.loads(p.read_text()) for p in (pool/'transfer').glob('*.json')]
        assert len(records) == 27 and all(r['status'] == 'complete' for r in records)
        rows = [{**r, 'group': record['group'], 'seed': record['seed']} for record in records for r in record['rows']]
        random = {(r['seed'], r['target'], r['budget'], r['mode']): r['test_accuracy'] for r in rows if r['group'] == 'random'}
        gains_by_pool.append({(r['group'], r['seed'], r['target'], r['budget'], r['mode']):
                              r['test_accuracy']-random[r['seed'], r['target'], r['budget'], r['mode']]
                              for r in rows if r['group'] != 'random'})
        provenance.update({str(p): sha(p) for p in (pool/'transfer').glob('*.json')})
    fitted, old_scores = [], []
    for mode, budget in product(('shared', 'condition'), (64, 256)):
        old_mode = 'finetune_shared' if mode == 'shared' else 'finetune'
        y = np.array([np.mean([pool[r['group'], r['seed'], r['target'], budget, old_mode]
                              for pool in gains_by_pool]) for r in features])
        for variant in VARIANTS:
            x = np.array([feature_vector(r, variant) for r in features])
            weights = fit_weights(x, y)
            fitted.append({'mode': mode, 'budget': budget, 'variant': variant, 'weights': weights})
            predicted = np.zeros(len(y))
            for group in sorted({r['group'] for r in features}):
                mask = np.array([r['group'] == group for r in features])
                predicted[mask] = ridge_predict(x[~mask], y[~mask], x[mask], plan['ridge_alpha'])
            old_scores.append({'mode': mode, 'budget': budget, 'variant': variant,
                               'r2': float(1-np.sum((y-predicted)**2)/np.sum((y-y.mean())**2)),
                               'mse': float(np.mean((y-predicted)**2)), 'mae': float(np.mean(abs(y-predicted))),
                               'unit': 'eight old whole-group folds; exploratory, not new observations'})
    atomic_json(root/'weights.json', {'frozen_utc': now(), 'new_data_exists_at_freeze': False,
                                      'training_provenance': provenance, 'weights': fitted})
    atomic_json(root/'old_group_holdout.json', {'scores': old_scores, 'rows': 96,
                                              'old_pool_gains_averaged_not_counted_as_independent': True})


def prepare_data(plan, config, root, names, functions, tokens, one_line):
    dataset = root/'dataset'; dataset.mkdir(exist_ok=True)
    seen = set()
    for path in plan['excluded_datasets']:
        with np.load(path) as archive:
            for key in archive.files:
                if key.endswith('_input'):
                    split = key[:-6]
                    seen.update(permutation_row(row, int(n)) for row, n in zip(archive[key], archive[split+'_lengths']))
    excluded_count = len(seen)
    rng = np.random.default_rng(plan['data_seed']); data = {}
    for split, per_length in plan['examples_per_length'].items():
        inputs, labels, lengths = [], [], []
        for n in range(config['lengths'][0], config['lengths'][1]+1):
            for _ in range(per_length):
                while True:
                    permutation = tuple(map(int, rng.permutation(n)+1))
                    if permutation not in seen:
                        seen.add(permutation); break
                prefix = [tokens['<BOS>'], tokens['<SIZE>'], n]+[tokens[t] for t in one_line(permutation)]
                inputs.append(prefix+[tokens['<PAD>']]*(2*config['lengths'][1]+4-len(prefix)))
                labels.append([functions[t](permutation) for t in names]); lengths.append(n)
        data[split+'_input'] = np.array(inputs, dtype=np.int64)
        data[split+'_labels'] = np.array(labels, dtype=np.int64)
        data[split+'_lengths'] = np.array(lengths, dtype=np.int64)
        for identity in IDENTITIES:
            lhs = sum(coefficient*data[split+'_labels'][:, names.index(task)] for task, coefficient in identity['terms'].items())
            assert np.array_equal(lhs, identity['n']*data[split+'_lengths']+identity['constant'])
        print(f'confirmation data {split}: {len(inputs)}', flush=True)
    np.savez_compressed(dataset/'data.npz', **data)
    order = np.random.default_rng(plan['data_seed']+50000).permutation(len(data['support_pool_input']))
    atomic_json(dataset/'support_indices.json', {str(b): order[:b].tolist() for b in config['target_budgets']})
    atomic_json(dataset/'metadata.json', {'names': names, 'data_sha256': sha(dataset/'data.npz'),
                'generator_sha256': sha(__file__), 'data_seed': plan['data_seed'],
                'split_counts': {s: len(data[s+'_input']) for s in plan['examples_per_length']},
                'excluded_distinct_prior_permutations': excluded_count,
                'excluded_sha256': {p: sha(p) for p in plan['excluded_datasets']},
                'all_splits_and_previous_native_inputs_globally_disjoint': True})


def prepare():
    plan, config, root = paths(); root.mkdir(parents=True, exist_ok=True)
    signature = {'plan': plan, 'config_sha256': sha('configs/native_confirmation.json'),
                 'core_sha256': {p: sha(p) for p in CORE_PATHS},
                 'upstream_sha256': {name: sha(Path(config['repository'])/'src/neurips_permutations'/name)
                                    for name in ('models.py', 'math_ops.py', 'passage.py')},
                 'target_tuning_sha256': sha('results/six_hour_session/target_tuning.json')}
    protocol = root/'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())['signature'] == signature
        if (root/'dataset/metadata.json').exists() and (root/'weights.json').exists():
            assert (root/'dataset/data.npz').exists()
            return
    else:
        assert not (root/'dataset/data.npz').exists() and not list((root/'transfer').glob('*.json'))
        atomic_json(protocol, {'registered_utc': now(), 'signature': signature,
                              'new_source_records_at_freeze': 0, 'new_target_records_at_freeze': 0})
    names, functions, tokens, one_line = setup(config)
    groups = select_groups(config)
    assert next(g for g in groups if g['id'] == plan['candidate_group'])['tasks'] == plan['single_tasks']
    assert not set(plan['model_seeds']) & {17, 42, 101}
    atomic_json(root/'jobs.json', jobs(plan, groups))
    if not (root/'weights.json').exists():
        assert not (root/'dataset/data.npz').exists()
        prepare_weights(plan, root)
    prepare_data(plan, config, root, names, functions, tokens, one_line)
    atomic_json(root/'state.json', {'status': 'prepared', 'updated_utc': now(), 'planned_source_models': 46})


def load():
    plan, config, root = paths()
    protocol = json.loads((root/'protocol.json').read_text())
    assert protocol['signature']['config_sha256'] == sha('configs/native_confirmation.json')
    for p, expected in protocol['signature']['core_sha256'].items(): assert sha(p) == expected, p
    for p, expected in protocol['signature']['upstream_sha256'].items():
        assert sha(Path(config['repository'])/'src/neurips_permutations'/p) == expected
    names, functions, tokens, one_line = setup(config)
    assert sha(root/'dataset/data.npz') == json.loads((root/'dataset/metadata.json').read_text())['data_sha256']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with np.load(root/'dataset/data.npz') as archive:
        raw = {k: archive[k] for k in archive.files}
    data = {k: torch.tensor(v, device=device) for k, v in raw.items()}
    return plan, config, root, raw, data, names, functions, tokens, device


def train():
    plan, config, root, raw, data, names, functions, tokens, device = load()
    catalog = json.loads((root/'jobs.json').read_text())
    for index, job in enumerate(catalog):
        atomic_json(root/'state.json', {'status': 'source_training', 'job': job,
                    'completed_queue_entries': index, 'planned_source_models': len(catalog), 'updated_utc': now()})
        model, record = train_job(plan, config, plan['architecture'], job, data, names, tokens, device, float('inf'))
        assert record['status'] == 'complete'
        del model
        if device == 'cuda': torch.cuda.empty_cache()
    atomic_json(root/'state.json', {'status': 'all_sources_complete', 'completed_source_models': len(catalog), 'updated_utc': now()})


def forecast():
    plan, config, root = paths()
    if (root/'forecasts.json').exists():
        saved=json.loads((root/'forecasts.json').read_text())
        assert saved['weights_sha256']==sha(root/'weights.json')
        assert saved['features_sha256']==sha(root/'forecast_features.json')
        return
    assert not list((root/'transfer').glob('*.json'))
    names, functions, tokens, one_line = setup(config)
    groups = select_groups(config)
    with np.load(root/'dataset/data.npz') as archive:
        # No new target-test outcomes or labels enter forecast construction.
        source = archive['train_labels']; lengths = archive['train_lengths']
        x, n, y = archive['representation_input'], archive['representation_lengths'], archive['representation_labels']
    transformed = {}
    for operator in OPERATORS:
        changed = token_transform(x, n, operator)
        transformed[operator] = np.array([[functions[t](permutation_row(row, int(length))) for t in names]
                                         for row, length in zip(changed, n)])
    rows = []
    for group in groups:
        ids = [names.index(t) for t in group['tasks']]
        code = np.eye(31)[y[:, ids]].reshape(len(y), -1)
        centered = within_length_center(code, n)
        code_cka = []
        actions = certificates(group)
        for operator in OPERATORS:
            other = transformed[operator][:, ids]
            if operator in actions: assert np.array_equal(y[:, ids]@actions[operator][0].T, other)
            code_cka.append(linear_cka(centered, within_length_center(np.eye(31)[other].reshape(len(y), -1), n)))
        unary = sum(bool(np.isin(matrix,[0,1]).all() and (matrix.sum(0)==1).all() and (matrix.sum(1)==1).all())
                    for matrix,_ in actions.values())
        minimum = group['minimum_constraint_size']
        math = [minimum or 5, float(not minimum), float(np.mean(code_cka)), len(actions)/4, unary/4]
        statistics={target:label_statistics(source[:,ids],source[:,names.index(target)],lengths)
                    for target in config['target_tasks']}
        seeds = plan['model_seeds'] if group['id'] == plan['candidate_group'] else plan['prediction_seeds']
        for seed in seeds:
            rid = f"{plan['architecture']['id']}_{group['id']}_s{seed}"
            record = json.loads((root/'multi'/f'{rid}.json').read_text()); assert record['status']=='complete'
            audit = float(np.mean([r['accuracy'] for r in record['source_audit']]))
            curve = float(np.mean([r['accuracy'] for r in record['curve']]))
            for target in config['target_tasks']:
                stats = statistics[target]
                baseline = [audit, curve]+stats+[float(target==t) for t in config['target_tasks'][1:]]
                rows.append({'group':group['id'],'seed':seed,'target':target,'baseline':baseline,'math':math})
    atomic_json(root/'forecast_features.json', rows)
    fitted = json.loads((root/'weights.json').read_text())
    forecasts = []
    for weights in fitted['weights']:
        predictions = apply_weights(np.array([feature_vector(r, weights['variant']) for r in rows]), weights['weights'])
        forecasts.extend({**{k:r[k] for k in ('group','seed','target')},
                          **{k:weights[k] for k in ('mode','budget','variant')},
                          'predicted_paired_gain':float(p)} for r,p in zip(rows,predictions))
    atomic_json(root/'forecasts.json', {'frozen_utc':now(),'new_transfer_records_at_freeze':0,
                'weights_sha256':sha(root/'weights.json'),'features_sha256':sha(root/'forecast_features.json'),
                'source_records_sha256':{str(p):sha(p) for p in (root/'multi').glob('*.json')},'rows':forecasts})
    atomic_json(root/'state.json', {'status':'forecasts_saved_before_new_adaptation','forecast_rows':len(forecasts),'updated_utc':now()})


def policies(plan, group):
    condition = group if group in plan['condition_policies'] else plan['candidate_group']
    return {**plan['adaptation_policies'], 'condition':plan['condition_policies'][condition]}


def adapt_predictions(model, config, data, target, indices, seed, policy, names, tokens):
    learner = copy.deepcopy(model)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(learner.parameters(), lr=policy['learning_rate'], weight_decay=config['weight_decay'])
    rng = np.random.default_rng(seed+70000)
    for _ in range(policy['steps']):
        chosen = rng.choice(indices, min(config['adapt_batch_size'],len(indices)), replace=False)
        learner.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            logits = answer_logits(learner,data['support_pool_input'][chosen],data['support_pool_lengths'][chosen],[target],tokens)
            loss = F.cross_entropy(logits.float(),data['support_pool_labels'][chosen,names.index(target)])
        loss.backward();torch.nn.utils.clip_grad_norm_(learner.parameters(),1.);optimizer.step()
    learner.eval();predictions=[]
    with torch.no_grad():
        for start in range(0,len(data['target_test_input']),64):
            logits=answer_logits(learner,data['target_test_input'][start:start+64],data['target_test_lengths'][start:start+64],[target],tokens)
            predictions.append(logits.argmax(-1).cpu().numpy())
    del learner
    return np.concatenate(predictions).astype(np.int16)


def evaluate():
    plan, config, root, raw, data, names, functions, tokens, device = load()
    forecasts=json.loads((root/'forecasts.json').read_text())
    assert forecasts['weights_sha256']==sha(root/'weights.json')
    support=json.loads((root/'dataset/support_indices.json').read_text())
    catalog=[{'phase':'random','id':'random','seed':s,'tasks':[]} for s in plan['model_seeds']]+json.loads((root/'jobs.json').read_text())
    directory=root/'transfer';directory.mkdir(exist_ok=True)
    for index,job in enumerate(catalog):
        rid=f"{plan['architecture']['id']}_{job['id']}_s{job['seed']}"
        path=directory/f'{rid}.json';prediction_path=directory/f'{rid}_predictions.npz'
        signature={'job':job,'protocol_sha256':sha(root/'protocol.json'),'forecasts_sha256':sha(root/'forecasts.json'),'data_sha256':sha(root/'dataset/data.npz')}
        checkpoint=None
        if job['phase']!='random':
            record=json.loads((root/job['phase']/f'{rid}.json').read_text())
            checkpoint=root/job['phase']/'checkpoints'/f'{rid}.pt'
            assert record['status']=='complete' and sha(checkpoint)==record['checkpoint_sha256']
            signature['source_checkpoint_sha256']=record['checkpoint_sha256']
        if path.exists() and json.loads(path.read_text())['status']=='complete':
            saved=json.loads(path.read_text());assert saved['signature']==signature
            assert saved['predictions_sha256']==sha(prediction_path);continue
        atomic_json(root/'state.json',{'status':'target_evaluation','job':job,'completed_queue_entries':index,'planned_transfer_models':len(catalog),'updated_utc':now()})
        model=make_model(config,plan['architecture'],job['seed'],device)
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location=device)['model'])
        rows,archived=[],{}
        for target,budget in product(config['target_tasks'],config['target_budgets']):
            cache={}
            truth=raw['target_test_labels'][:,names.index(target)]
            for mode,policy in policies(plan,job['id']).items():
                key=policy['learning_rate'],policy['steps']
                if key not in cache: cache[key]=adapt_predictions(model,config,data,target,support[str(budget)],job['seed'],policy,names,tokens)
                prediction=cache[key]
                archived[f'{target}_{budget}_{mode}']=prediction
                rows.append({'target':target,'budget':budget,'mode':mode,'policy':policy,'test_accuracy':float(np.mean(prediction==truth))})
            atomic_json(path,{'status':'running','signature':signature,'group':job['id'],'seed':job['seed'],'phase':job['phase'],'rows':rows})
        np.savez_compressed(prediction_path,**archived)
        atomic_json(path,{'status':'complete','signature':signature,'group':job['id'],'seed':job['seed'],'phase':job['phase'],'rows':rows,
                          'predictions_sha256':sha(prediction_path),'evaluated_utc':now()})
        print(json.dumps({'transfer_complete':index+1,'total':len(catalog),'group':job['id'],'seed':job['seed']}),flush=True)
        del model
        if device=='cuda':torch.cuda.empty_cache()
    length_rows=[];length_predictions={}
    for target,budget in product(config['target_tasks'],config['target_budgets']):
        ids=support[str(budget)];index=names.index(target)
        train_n=raw['support_pool_lengths'][ids];train_y=raw['support_pool_labels'][ids,index]
        test_n=raw['target_test_lengths'];truth=raw['target_test_labels'][:,index]
        predicted=list(majority_predictions(train_n,train_y,test_n))+[smooth_prediction(train_n,train_y,test_n,plan['length_baseline_bandwidth'],31)]
        for mode,prediction in zip(('global_majority','exact_length_majority','smooth_length'),predicted):
            length_rows.append({'target':target,'budget':budget,'mode':mode,'test_accuracy':float(np.mean(prediction==truth))})
            length_predictions[f'{target}_{budget}_{mode}']=prediction.astype(np.int16)
    atomic_json(root/'length_baselines.json',{'bandwidth':plan['length_baseline_bandwidth'],'selected_using_previous_data_only':True,'rows':length_rows})
    np.savez_compressed(root/'length_predictions.npz',**length_predictions)
    atomic_json(root/'state.json',{'status':'all_target_evaluations_complete','transfer_models':len(catalog),'endpoints':len(catalog)*24,'updated_utc':now()})


def run():
    prepare();train();forecast();evaluate()
    from .native_confirmation_analysis import analyze, verify
    analyze();verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','train','forecast','evaluate','run'),default='run')
    args=parser.parse_args()
    globals()[args.phase]()
