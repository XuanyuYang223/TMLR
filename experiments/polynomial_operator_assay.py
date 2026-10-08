"""Reuse field encoders for D P1 = P0 D on quadratics plus fixed context.

The four latent coordinates are (constant, linear, quadratic, context).
All transformations preserve the context. Probe splits hold out complete
contexts, keeping source/input/image points disjoint between splits. Encoders
were previously trained on all 625 points, so this is probe generalization,
not new source training or an independent mathematical-domain replication.
"""
from datetime import datetime,timezone
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from .algebra_noninvertible import polynomial_operators
from .field_algebra_structure import affine_fit
from .field_symmetry import group_sources
from .matched_field import controlled_world
from .longrun_engine import atomic_json
from .permworld_combinations import sha
from .representation_algebra import pca_basis,shuffled_rows


def operators():
    d,ps=polynomial_operators(2);result={}
    for name,small in [('d',d),('p1',ps[1]),('p0',ps[0])]:
        full=np.eye(4,dtype=np.int64);full[:3,:3]=small;result[name]=full
    assert np.array_equal(result['d'] @ result['p1'],result['p0'] @ result['d'])
    assert not np.array_equal(result['d'] @ result['p1'],result['p1'] @ result['d'])
    return result


def input_word(values,word,p=5):
    result=values.copy();ops=operators()
    for letter in word:result=result @ ops[letter].T % p
    return result


def context_split(latent,seed):
    values=np.random.default_rng(seed).permutation(5)
    assignment={int(z):0 if i<3 else 1 if i==3 else 2 for i,z in enumerate(values)}
    return np.array([assignment[int(z)] for z in latent[:,3]],dtype=np.int64),values.tolist()


def affine_word(x,word,maps):
    result=x
    for letter in word:
        rho,bias=maps[letter];result=result @ rho+bias
    return result


def probe(hidden,images,split,contexts,dimension,seed):
    h=np.asarray(hidden,dtype=np.float64);fit=split==0;val=split==1;test=split==2
    mean=h[fit].mean(0);q=pca_basis(h[fit]-mean,dimension);z=(h-mean) @ q
    grid=(1e-6,1e-4,.01,1.);maps={};wrong_maps={};rng=np.random.default_rng(seed)
    result={'probe_dimension':q.shape[1],'generators':[],'composites':[],'laws':[]};arrays={'basis':q.astype(np.float32),'mean':mean,'test_latent':z[test]}
    def evaluate(predicted,word,wrong=None):
        ids=images['_'.join(word)];delta=h[ids[test]]-h[test];denom=np.square(delta).sum()
        row={'word':'_'.join(word),'identity_displacement_nmse':1.,
            'displacement_nmse':float(np.square((predicted-z[test]) @ q.T-delta).sum()/denom),
            'displacement_energy':float(denom/np.square(h[test]-mean).sum()),'shuffled_fit_displacement_nmse':None}
        if wrong is not None:row['shuffled_fit_displacement_nmse']=float(np.square((wrong-z[test]) @ q.T-delta).sum()/denom)
        return row
    for name in ('d','p1','p0'):
        y=z[images[name]];candidates=[affine_fit(z[fit],y[fit]-z[fit],a) for a in grid]
        affine=[(np.eye(q.shape[1])+a[:-1],a[-1]) for a in candidates]
        selected=int(np.argmin([np.square(affine_word(z[val],[name],{name:m})-y[val]).sum() for m in affine]));maps[name]=affine[selected]
        sy=y[fit][shuffled_rows(contexts[fit],rng)];sv=y[val][shuffled_rows(contexts[val],rng)]
        candidates=[affine_fit(z[fit],sy-z[fit],a) for a in grid];wrong=[(np.eye(q.shape[1])+a[:-1],a[-1]) for a in candidates]
        selected_wrong=int(np.argmin([np.square(affine_word(z[val],[name],{name:m})-sv).sum() for m in wrong]));wrong_maps[name]=wrong[selected_wrong]
        row=evaluate(affine_word(z[test],[name],maps),[name],affine_word(z[test],[name],wrong_maps));row['alpha']=grid[selected];row['shuffled_alpha']=grid[selected_wrong]
        result['generators'].append(row);arrays['rho_'+name]=maps[name][0];arrays['bias_'+name]=maps[name][1]
        arrays['shuffled_rho_'+name]=wrong_maps[name][0];arrays['shuffled_bias_'+name]=wrong_maps[name][1]
    words=[('p1','d'),('d','p0'),('p1','p1'),('p0','p0')]
    for word in words:result['composites'].append(evaluate(affine_word(z[test],word,maps),word,affine_word(z[test],word,wrong_maps)))
    for left,right in [(('p1','d'),('d','p0')),(('p1','p1'),('p1',)),(('p0','p0'),('p0',))]:
        assert np.array_equal(images['_'.join(left)],images['_'.join(right)])
        one,two=affine_word(z[test],left,maps),affine_word(z[test],right,maps)
        delta=h[images['_'.join(left)][test]]-h[test]
        result['laws'].append({'left':'_'.join(left),'right':'_'.join(right),
            'full_space_consistency_nmse':float(np.square((one-two) @ q.T).sum()/np.square(delta).sum()),
            'left_prediction_nmse':evaluate(one,left)['displacement_nmse'],
            'right_prediction_nmse':evaluate(two,right)['displacement_nmse']})
    prediction=h[test]+(affine_word(z[test],('p1','d'),maps)-z[test]) @ q.T
    correct=h[images['p1_d'][test]];wrong=h[images['d_p1'][test]];denom=np.square(correct-h[test]).sum()
    result['wrong_order_gap']=float((np.square(prediction-wrong).sum()-np.square(prediction-correct).sum())/denom)
    return result,arrays


def run():
    root=Path('results/algebra_structure_replication/polynomial');root.mkdir(exist_ok=True,parents=True)
    for name in ('probes','arrays'):(root/name).mkdir(exist_ok=True)
    config=json.loads(Path('configs/field_matched_support.json').read_text());source=Path(config['output'])
    signature={'code_sha256':sha(__file__),'operator_code_sha256':sha('experiments/algebra_noninvertible.py'),
        'dependency_sha256':{p:sha(p) for p in ['experiments/field_algebra_structure.py','experiments/representation_algebra.py','experiments/matched_field.py','experiments/field_symmetry.py']},
        'source_protocol_sha256':sha(source/'metadata.json'),'context_split_seed':2026100642,
        'operators':{k:v.tolist() for k,v in operators().items()},'probe_dimensions':[32,64],
        'scope':'Reuse field encoders; polynomial coordinates with independent preserved context; source encoders saw all points; probe-held-out contexts only.',
        'predictions':['P has lower mean generator and unfitted-cross prediction errors than each mixed source group.',
                       'Correct cross law prediction beats wrong D/P1 order in P.',
                       'Actual prediction error accompanies consistency; no law enforced during fitting.']}
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['signature']==signature
    else:atomic_json(path,{'registered_utc':datetime.now(timezone.utc).isoformat(),'signature':signature,'new_results':0})
    rows=[];words=[('d',),('p1',),('p0',),('p1','d'),('d','p0'),('p1','p1'),('p0','p0'),('d','p1')]
    for w in config['world_seeds']:
        inputs,_,_,basis=controlled_world(5,4,w);latent=inputs @ basis.T % 5;lookup={tuple(z):i for i,z in enumerate(latent)}
        images={'_'.join(word):np.array([lookup[tuple(z)] for z in input_word(latent,word)]) for word in words}
        split,ctx=context_split(latent,signature['context_split_seed'])
        for ids in images.values():
            assert np.array_equal(split[ids],split)
        for g,m,status in product(config['groups'],config['model_seeds'],('random','trained')):
            name=f'{g}_w{w}_s{m}_{status}';dest=root/'probes'/f'{name}.json'
            if dest.exists():rows.append(json.loads(dest.read_text()));continue
            step=0 if status=='random' else config['steps'];fp=source/f'{g}_w{w}_m{m}_step{step}_features.npy';h=np.load(fp)
            views={'hidden':h}
            if status=='trained':
                state=torch.load(source/'checkpoints'/f'{g}_w{w}_m{m}.pt',weights_only=True,map_location='cpu')
                weights=state['heads.weight'].numpy().astype(np.float64).reshape(4,5,64);contrasts=(weights-weights.mean(1,keepdims=True)).reshape(20,64)
                _,s,v=np.linalg.svd(contrasts,full_matrices=False);rank=int(np.sum(s>s[0]*1e-10));b=v[:rank].T
                projected=(h @ b) @ b.T;views['readout_contrast']=projected;views['readout_null']=h-projected
                labels=latent @ group_sources(g,5).T % 5;views['exact_onehot']=np.eye(5)[labels].reshape(len(labels),-1)
            results={};hashes={}
            for view,features in views.items():
                for dim in [32]+([64] if view=='hidden' else []):
                    key=view if dim==32 else view+'_full_width';result,arrays=probe(features,images,split,latent[:,3],dim,m+9300)
                    path=root/'arrays'/f'{name}_{key}.npz';np.savez_compressed(path,**arrays);hashes[path.name]=sha(path);results[key]=result
            record={'group':g,'world_seed':w,'seed':m,'model_status':status,'results':results,
                'feature_sha256':sha(fp),'array_sha256':hashes,'heldout_context_values':ctx[3:],
                'source_encoders_previously_saw_all_inputs':True}
            atomic_json(dest,record);rows.append(record)
    summaries=[]
    for g in config['groups']:
        selected=[r['results']['hidden'] for r in rows if r['group']==g and r['model_status']=='trained']
        random=[r['results']['hidden'] for r in rows if r['group']==g and r['model_status']=='random']
        score=lambda r:float(np.mean([a['displacement_nmse'] for a in r['generators']]))
        summaries.append({'group':g,'generator_nmse':float(np.mean([score(r) for r in selected])),
            'random_generator_nmse':float(np.mean([score(r) for r in random])),
            'cross_prediction_nmse':float(np.mean([r['laws'][0]['left_prediction_nmse'] for r in selected])),
            'cross_consistency_nmse':float(np.mean([r['laws'][0]['full_space_consistency_nmse'] for r in selected])),
            'wrong_order_gap':float(np.mean([r['wrong_order_gap'] for r in selected]))})
    atomic_json(root/'summary.json',{'rows':summaries,'scope':signature['scope'],'heldout_contexts':ctx[3:],
        'source_inputs_per_model':625,'fit_validation_test_rows':[375,125,125],
        'source_label_statistics_matched':True,'source_accuracy_all_tasks':1.,'new_source_training':False})
    atomic_json(root/'state.json',{'status':'complete','conditions':54})
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':run()
