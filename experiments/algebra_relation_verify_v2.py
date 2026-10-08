"""Independent exact-label, exposure, readout, prediction, and statistical audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch

from . import algebra_relation_v3 as runner
from . import algebra_relation_campaign as campaign
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .two_step_relation_factorial import now
from .overnight_inverse_statistics_verify import bootstrap_interval


def independent_action(x, letter, domain, p):
    a,b,c,d=map(int,x)
    if domain=='matrix':
        return (c,d,a,b) if letter=='a' else ((-c)%p,(-d)%p,(a-c)%p,(b-d)%p)
    if letter=='b':return (b,2*c%p,3*d%p,0)
    return ((a+b+c+d)%p,(b+2*c+3*d)%p,(c+3*d)%p,d)


def independent_word(x,word,domain,p):
    for letter in word:x=independent_action(x,letter,domain,p)
    return tuple(map(int,x))


def split_key(x,domain,p):
    if domain=='matrix':
        return min(independent_word(x,w,domain,p) for w in ['','a','b','bb','ab','ba'])
    a,b,c,d=map(int,x)
    if d:return (d,(4*c*c-12*b*d)%p)
    assert c
    return (c*pow(3,-1,p)%p,(b*b-4*a*c)%p)


def collision_components(data,domain,p):
    parent={}
    def find(k):
        parent.setdefault(k,k)
        if parent[k]!=k:parent[k]=find(parent[k])
        return parent[k]
    for pair in np.unique(data['pair_ids'][data['split']==1]):
        ids=np.flatnonzero(data['pair_ids']==pair)
        left,right=[split_key(data['x'][i,0],domain,p) for i in ids]
        a,b=find(left),find(right)
        if a!=b:parent[a]=b
    sizes=Counter(find(k) for k in parent)
    return {'split_blocks':len(parent),'connected_components':len(sizes),
            'component_block_sizes':sorted(sizes.values(),reverse=True),
            'interpretation':'Collision pairs linking the same split blocks are dependent; intervals below resample sources only and condition on the fixed test inputs.'}


def run(domain):
    runner.configure();torch.set_num_threads(1)
    root=runner.ROOT/domain
    assert json.loads((root/'state.json').read_text())['status']=='complete'
    done=json.loads((root/'completion.json').read_text())
    for file,digest in done['artifact_sha256'].items():assert sha(file)==digest
    protocol=json.loads((root/'protocol.json').read_text())
    plan=protocol['signature']['plan'];p=plan['field_prime'];words=plan['words']
    data={name:dict(np.load(root/'dataset'/f'{name}.npz')) for name in
          ['source','validation','pilot_validation','test']+[f'support{i}' for i in range(7)]}
    labels_checked=0
    for name,d in data.items():
        local=words if name=='test' else ['','a','b']
        for row,y in zip(d['x'],d['labels']):
            for j,word in enumerate(local):
                truth=independent_word(row[0],word,domain,p)
                assert truth==tuple(map(int,row[j]))
                assert int(y[j])==((truth[0]+truth[3])%p if domain=='matrix' else truth[0])
                labels_checked+=1
        if name.startswith('support'):
            pc,pi=d['pc'],d['pi'];index=np.arange(len(pc))
            assert np.array_equal(np.sort(pc),index) and np.array_equal(np.sort(pi),index)
            assert np.all(pc!=index) and np.all(pi!=index)
            assert np.all(pi[pc]!=index) and np.all(pc[pi]!=index)
            assert np.array_equal(d['labels'],d['labels'][pc]) and np.array_equal(d['labels'],d['labels'][pi])
    train=set(tuple(x) for name,d in data.items() if name=='source' or name.startswith('support') for x in d['x'].reshape(-1,4))
    val=set(tuple(x) for name in ['validation','pilot_validation'] for x in data[name]['x'].reshape(-1,4))
    test=set(tuple(x) for x in data['test']['x'].reshape(-1,4))
    assert not(train&val or train&test or val&test)
    train_blocks={split_key(x,domain,p) for x in train}
    val_blocks={split_key(x,domain,p) for x in val}
    test_blocks={split_key(x,domain,p) for x in data['test']['x'][:,0]}
    assert not(train_blocks&val_blocks or train_blocks&test_blocks or val_blocks&test_blocks)
    actual=data['test'];use=actual['split']==1
    for pair in np.unique(actual['pair_ids'][use]):
        ids=np.flatnonzero(actual['pair_ids']==pair)
        assert len(ids)==2 and np.array_equal(actual['labels'][ids[0],:3],actual['labels'][ids[1],:3])
        assert actual['labels'][ids[0],3]!=actual['labels'][ids[1],3]
    summary=json.loads((root/'summary.json').read_text());records=summary['records']
    checks=0;max_native_error=0.;roundoff_ties=0;primitive=[];scores={};relations=[]
    with torch.no_grad():
        features=torch.tensor((actual['x'].astype(np.float32)-(p-1)/2)*(2*np.pi/p),device='cuda')
        for i in range(6):
            schedule=None
            source=torch.load(root/'sources'/f's{i%3}.pt',map_location='cpu',weights_only=True)['model']
            for condition in plan['conditions']:
                name=f'n{i}_{condition}';fit=json.loads((root/'fits'/f'{name}.json').read_text())
                assert fit['epochs']==900 and fit['updates']==900*16
                assert fit['anchor_exposures']==900*1024 and fit['native_forward_rows']==5*900*1024
                if schedule is None:schedule=fit['schedule_sha256']
                assert fit['schedule_sha256']==schedule
                cp=torch.load(root/'models'/f'{name}.pt',map_location='cpu',weights_only=True)
                for key in ['readout.weight','readout.bias']:
                    assert torch.equal(cp['model'][key],source[key])
                model=campaign.Encoder(p,plan['hidden_width']).to('cuda');model.load_state_dict(cp['model']);model.eval()
                h=model(features);native=model.readout(h);v=native if cp['output_space'] else h
                saved=dict(np.load(root/'evaluations'/f'{name}.npz'))
                error=float(np.abs(v.cpu().numpy()-saved['native']).max());max_native_error=max(max_native_error,error)
                np.testing.assert_allclose(v.cpu().numpy(),saved['native'],atol=1e-4,rtol=5e-5)
                native_grade=(native[:,:3].argmax(-1).cpu().numpy()==actual['labels'][:,:3]).mean(0)
                gen=[]
                for word in words[1:]:
                    pred=saved[word+'_pred'].astype(float)
                    independent=saved['native'][:,0].astype(float)
                    for letter in word:
                        j=int(letter=='b')
                        independent=independent@cp['operators'][f'maps.{j}.weight'].numpy().T+cp['operators'][f'maps.{j}.bias'].numpy()
                    np.testing.assert_allclose(pred,independent,atol=2e-3,rtol=2e-4)
                    logits=pred if cp['output_space'] else pred@source['readout.weight'].numpy().T+source['readout.bias'].numpy()
                    answer=saved[word+'_answers'];mismatch=logits.argmax(-1)!=answer
                    if mismatch.any():
                        margins=logits.max(-1)-logits[np.arange(len(answer)),answer]
                        assert np.all(margins[mismatch]<2e-4)
                        roundoff_ties+=int(mismatch.sum())
                    hits=answer==actual['labels'][:,words.index(word)]
                    if word in ['a','b']:gen.append(float(hits.mean()))
                    for sid,split in [(0,'iid'),(1,'collisions')]:
                        use=actual['split']==sid
                        r=next(r for r in records if (r['replicate'],r['condition'],r['word'],r['split'])==(i,condition,word,split))
                        assert abs(hits[use].mean()-r['accuracy'])<1e-12
                        truth=saved['native'][use,words.index(word)].astype(float)
                        base=saved['native'][use,0].astype(float)
                        nmse=float(np.square(pred[use]-truth).sum()/max(np.square(truth-base).sum(),1e-30))
                        assert abs(nmse-r['prediction_nmse'])<1e-9*max(1,nmse)
                        if sid==1:
                            pairhits=[hits[actual['pair_ids']==pair].all() for pair in np.unique(actual['pair_ids'][use])]
                            assert abs(np.mean(pairhits)-r['pair_both_correct'])<1e-12
                        checks+=1
                    if word=='ab':scores[(i,condition)]=hits
                primitive.append({'replicate':i,'source_index':i%3,'condition':condition,
                                  'test_native':native_grade.tolist(),'test_generators':gen,
                                  'validation_grade':fit['curve'][-1]})
                # Auxiliary laws are evaluated using learned operators only, without refitting.
                def apply_word(word):
                    result=saved['native'][:,0].astype(float)
                    for letter in word:
                        j=int(letter=='b')
                        result=result@cp['operators'][f'maps.{j}.weight'].numpy().T+cp['operators'][f'maps.{j}.bias'].numpy()
                    return result
                if domain=='matrix':
                    laws=[('A_squared_identity',apply_word('aa'),saved['native'][:,0]),
                          ('B_cubed_identity',apply_word('bbb'),saved['native'][:,0]),
                          ('ABA_equals_B_squared',apply_word('aba'),apply_word('bb'))]
                else:
                    laws=[('T_D_commute',apply_word('ab'),apply_word('ba')),
                          ('D_fourth_zero_state',apply_word('bbbb'),saved['native'][:,words.index('bbbb')]),
                          ('T_thirteenth_identity',apply_word('a'*13),saved['native'][:,0])]
                for sid,split in [(0,'iid'),(1,'collisions')]:
                    use=actual['split']==sid;base=saved['native'][use,0].astype(float)
                    variance=float(np.square(base-base.mean(0)).sum())
                    relations.append({'replicate':i,'condition':condition,'split':split,
                        'normalization':'Squared norm divided by centered native-state variance, not zero identity displacement.',
                        **{name:float(np.square(left[use]-right[use]).sum()/max(variance,1e-30)) for name,left,right in laws},
                        'A_operator_spectral_norm':float(np.linalg.svd(cp['operators']['maps.0.weight'].numpy(),compute_uv=False)[0]),
                        'B_operator_spectral_norm':float(np.linalg.svd(cp['operators']['maps.1.weight'].numpy(),compute_uv=False)[0])})
    contrasts=[]
    for condition in plan['conditions'][1:]:
        for sid,split in [(0,'iid'),(1,'collisions')]:
            use=actual['split']==sid
            values=[100*(scores[(i,'both_correct')][use].mean()-scores[(i,condition)][use].mean()) for i in range(6)]
            sources=[float(np.mean([values[j],values[j+3]])) for j in range(3)]
            original=next(c for c in summary['contrasts'] if (c['condition'],c['split'])==(condition,split))
            np.testing.assert_allclose(values,original['paired_values'],atol=1e-12)
            interval=bootstrap_interval(sources)
            np.testing.assert_allclose(interval,original['three_source_bootstrap_95_pp'],atol=1e-12)
            contrasts.append({'comparison':'both_correct-'+condition,'split':split,'mean_pp':float(np.mean(values)),
                              'source_means_pp':sources,'source_bootstrap_95_pp':interval})
    output=root.parent/f'{domain}_verification';output.mkdir(exist_ok=True)
    atomic_json(output/'verification.json',{'status':'complete','completed_utc':now(),
        'independent_exact_transform_and_label_checks':labels_checked,'independent_endpoint_counts':checks,
        'matched900_epochs_all36':True,'all36_fixed_source_readouts_preserved':True,
        'maximum_original_native_forward_error':max_native_error,'original_native_replay_device':'cuda; independent affine calculations remain NumPy float64','float_roundoff_argmax_ties':roundoff_ties,
        'actual_known_and_test_paths_disjoint':True,'training_validation_test_blocks_disjoint':True,
        'collision_block_dependence':collision_components(actual,domain,p),
        'three_gold_visible_answer_ceiling':.5,'full_probability_output_ceiling_claimed':False,
        'primitive_grades':primitive,'contrasts':contrasts,'auxiliary_learned_relation_checks':relations,
        'zero_displacement_scope':'Registered primary AB prediction NMSE is well-defined. Identity words such as matrixAA andBBB have zero true displacement: their stored displacement NMSE is undefined in meaning and must not be interpreted. Supplemental law checks normalize by centered native-state variance.',
        'original_completion_sha256':sha(root/'completion.json'),'code_sha256':sha(__file__),
        'scope':'Source-cluster intervals conditional on fixed inputs; three sources reused in six fits. Scalar architecture selected by known-only matrix validation; supplied operators and order, not spontaneous discovery.'})
    print(json.dumps({'status':'complete','domain':domain,'checks':checks,'primitive_records':len(primitive)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('domain',choices=['matrix','polynomial']);args=parser.parse_args()
    run(args.domain)
