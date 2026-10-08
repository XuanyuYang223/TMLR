"""Exploratory margin-aware nullspace error analysis; immutable old models."""
import json
from pathlib import Path
import numpy as np
from .permworld_combinations import sha
from .two_step_relation_factorial import now
from .relation_error_localization import row_projection
from .longrun_engine import atomic_json

ROOT=Path('results/downstream_harm_diagnostic')
AMPLITUDES=[0.,.05,.1,.25,.5,1.,1.5,2.]
NATURAL=[0.,.25,.5,1.,1.5,2.]
CONDITIONS=['both_correct','first_correct','second_correct','both_wrong']


def metrics(z,d,y):
    correct=z.argmax(1)==y
    rivals=z.copy();rivals[np.arange(len(y)),y]=-np.inf
    gap=z[np.arange(len(y)),y]-rivals.max(1)
    changes=d-d[np.arange(len(y)),y,None]
    changes[np.arange(len(y)),y]=-np.inf
    harmful=changes.max(1)
    # Worst rival with its own margin, rather than mixing rival and minimum gap.
    allgap=z[np.arange(len(y)),y,None]-z
    allgap[np.arange(len(y)),y]=np.inf
    ratio=np.max(changes/np.maximum(allgap,1e-8),axis=1)
    center=d-d.mean(1,keepdims=True)
    return {'forced_correct':correct,'margin':gap,'worst_harm':harmful,'worst_margin_ratio':ratio,
            'centered_logit_sq':np.square(center).sum(1),
            'perturbed_correct':(z+d).argmax(1)==y}


def load(domain,i,c):
    if domain=='permworld':
        root=Path('results/two_step_relation_dose_confirmation')
        old=['both_correct','c_correct_i_wrong','c_wrong_i_correct','both_wrong'][CONDITIONS.index(c)]
        name=f'n{i}_hidden_{old}'
        hp=root/'features'/f'{name}_test.npz';mp=root/'maps'/f'{name}.npz';dp=root/'dataset/test/dataset.npz'
        h=np.load(hp)['hidden'].astype(float);m=dict(np.load(mp));data=dict(np.load(dp))
        a,b=m['rho_c'].astype(float),m['rho_i'].astype(float)
        ba,bb=m['bias_c'].astype(float),m['bias_i'].astype(float)
        w,bw=m['readout_weight'].astype(float),m['readout_bias'].astype(float)
        k=5;cp=root/'fits'/f'{name}.json'
    else:
        import torch
        root=Path('results/algebra_relation_v3/matrix')
        old=['both_correct','a_correct_b_wrong','a_wrong_b_correct','both_wrong'][CONDITIONS.index(c)]
        name=f'n{i}_{old}';hp=root/'evaluations'/f'{name}.npz';mp=root/'models'/f'{name}.pt';dp=root/'dataset/test.npz'
        h=np.load(hp)['native'].astype(float);s=torch.load(mp,map_location='cpu',weights_only=True)
        a=s['operators']['maps.0.weight'].numpy().T.astype(float);b=s['operators']['maps.1.weight'].numpy().T.astype(float)
        ba=s['operators']['maps.0.bias'].numpy().astype(float);bb=s['operators']['maps.1.bias'].numpy().astype(float)
        w=s['model']['readout.weight'].numpy().astype(float);bw=s['model']['readout.bias'].numpy().astype(float)
        data=dict(np.load(dp));k=3;cp=root/'fits'/f'{name}.json'
    return h,a,b,ba,bb,w,bw,data,k,{'hidden':str(hp),'operators':str(mp),'data':str(dp),'fit':str(cp)}


def run():
    import torch
    torch.set_num_threads(1)
    ROOT.mkdir(exist_ok=True,parents=True)
    protocol={'analysis':'posthoc; all original outcomes previously observed',
              'code_sha256':sha(__file__),'domains':['permworld','matrix'],'conditions':CONDITIONS,
              'amplitudes_hidden_rms':AMPLITUDES,'natural_error_multipliers':NATURAL,
              'random_directions_per_row':8,'random_seed':261079001,
              'normalization':'Fixed test hidden RMS around true-intermediate mean; diagnostics only. Margin ratios restricted to oracle-correct inputs.',
              'primary_contrast':'both_correct minus both_wrong; use source means, not six independent sources',
              'scope':'Real intermediates and labels used only for diagnosis, not inference or hyperparameter choice; nullspace preserves linear logits, not all answer information.'}
    pp=ROOT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())['signature']==protocol
    else:atomic_json(pp,{'registered_utc':now(),'signature':protocol})
    records=[];curves=[];inputs={};max_preserved=0.;max_identity=0.
    for domain in protocol['domains']:
        for i in range(6):
            for c in CONDITIONS:
                h,a,b,ba,bb,w,bw,data,k,paths=load(domain,i,c)
                for path in paths.values():inputs[path]=sha(path)
                p=row_projection(w);q=np.eye(len(p))-p
                first=h[:,0]@a+ba;err=first-h[:,1];en=err@q;er=err@p
                z=(h[:,1]@b+bb)@w.T+bw;y=data['labels'][:,k]
                dn=en@b@w.T;dr=er@b@w.T
                max_preserved=max(max_preserved,float(abs(en@w.T).max()))
                max_identity=max(max_identity,float(abs(err@b@w.T-dn-dr).max()))
                rms=float(np.sqrt(np.square(h[:,1]-h[:,1].mean(0)).sum(1).mean()))
                dnunit=np.divide(dn,np.linalg.norm(en,axis=1)[:,None],out=np.zeros_like(dn),where=np.linalg.norm(en,axis=1)[:,None]>1e-12)*rms
                natural=metrics(z,dn,y);mrow=metrics(z,dr,y)
                orig=(first@b+bb)@w.T+bw
                random_logits=[];rng=np.random.default_rng(261079001+i)
                for r in range(8):
                    v=rng.normal(size=en.shape)@q
                    v=v/np.linalg.norm(v,axis=1)[:,None]*rms
                    max_preserved=max(max_preserved,float(abs(v@w.T).max()))
                    random_logits.append(v@b@w.T)
                saved={'null_error_sq':np.square(en).sum(1),'row_error_sq':np.square(er).sum(1),
                       'total_error_sq':np.square(err).sum(1),'composed_correct':orig.argmax(1)==y,
                       **natural,'row_centered_logit_sq':mrow['centered_logit_sq'],
                       'split':data['split'],'pair_ids':data['pair_ids']}
                np.savez_compressed(ROOT/f'{domain}_n{i}_{c}.npz',**saved)
                for sid,split in [(0,'iid'),(1,'collisions')]:
                    use=data['split']==sid;positive=use & natural['forced_correct']
                    ratio=natural['worst_margin_ratio'][positive]
                    centered_scale=float(np.square(z[use]-z[use].mean(1,keepdims=True)).sum(1).mean())
                    records.append({'domain':domain,'replicate':i,'source':i%3,'condition':c,'split':split,
                        'first_hidden_nmse':float(np.square(err[use]).sum(1).mean()/rms**2),
                        'null_hidden_nmse':float(np.square(en[use]).sum(1).mean()/rms**2),
                        'row_hidden_nmse':float(np.square(er[use]).sum(1).mean()/rms**2),
                        'null_logit_nmse':float(natural['centered_logit_sq'][use].mean()/max(centered_scale,1e-12)),
                        'row_logit_nmse':float(mrow['centered_logit_sq'][use].mean()/max(centered_scale,1e-12)),
                        'forced_accuracy':float(natural['forced_correct'][use].mean()),
                        'composed_accuracy':float(saved['composed_correct'][use].mean()),
                        'natural_null_only_accuracy':float(natural['perturbed_correct'][use].mean()),
                        'oracle_correct_count':int(positive.sum()),
                        'null_harm_margin_ratio_median':float(np.median(ratio)) if len(ratio) else None,
                        'null_harm_margin_ratio_p90':float(np.quantile(ratio,.9)) if len(ratio) else None,
                        'null_margin_crossing_fraction':float((ratio>=1).mean()) if len(ratio) else None})
                    for direction,ds in [('actual_null',[dnunit]),('random_null',random_logits),('actual_natural',[dn])]:
                        for alpha in NATURAL if direction=='actual_natural' else AMPLITUDES:
                            hits=[(z+alpha*d).argmax(1)==y for d in ds]
                            accuracy=float(np.mean([hh[use].mean() for hh in hits]))
                            lost=float(np.mean([(natural['forced_correct'][use]&~hh[use]).mean() for hh in hits]))
                            curves.append({'domain':domain,'replicate':i,'source':i%3,'condition':c,'split':split,
                                'direction':direction,'amplitude':alpha,'accuracy':accuracy,'forced_correct_lost_fraction':lost})
                print(json.dumps({'domain':domain,'replicate':i,'condition':c,'status':'diagnosed'}),flush=True)
    assert max_preserved<1e-9 and max_identity<1e-8
    def aggregate(rows,keys):
        result=[]
        for key in sorted(set(tuple(r[k] for k in keys) for r in rows)):
            selected=[r for r in rows if tuple(r[k] for k in keys)==key]
            numeric=[k for k,v in selected[0].items() if k not in keys+['replicate','source'] and isinstance(v,(float,int))]
            result.append(dict(zip(keys,key),**{k:float(np.mean([r[k] for r in selected])) for k in numeric}))
        return result
    means=aggregate(records,['domain','condition','split']);cm=aggregate(curves,['domain','condition','split','direction','amplitude'])
    source_means=aggregate(records,['domain','condition','split','source'])
    contrasts=[]
    for domain in protocol['domains']:
        for split in ['iid','collisions']:
            x=[r for r in source_means if r['domain']==domain and r['split']==split]
            for metric in ['first_hidden_nmse','null_logit_nmse','null_margin_crossing_fraction','composed_accuracy']:
                ds=[next(r[metric] for r in x if r['condition']=='both_correct' and r['source']==s)-next(r[metric] for r in x if r['condition']=='both_wrong' and r['source']==s) for s in range(3)]
                rng=np.random.default_rng(261079111);boots=np.asarray(ds)[rng.integers(0,3,(10000,3))].mean(1)
                contrasts.append({'domain':domain,'split':split,'metric':metric,'correct_minus_wrong':float(np.mean(ds)),
                                  'source_differences':ds,'source_bootstrap_interval':np.quantile(boots,[.025,.975]).tolist()})
    atomic_json(ROOT/'results.json',{'status':'complete','completed_utc':now(),'records':records,'means':means,
        'source_means':source_means,'curves':curves,'curve_means':cm,'contrasts':contrasts,
        'input_sha256':inputs,'maximum_first_logit_change':max_preserved,'maximum_error_identity_residual':max_identity,
        'uncertainty_scope':'Only three source clusters; diagnostics are exploratory and conditional on fixed test inputs.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,domain in zip(axs,protocol['domains']):
        for condition,color in [('both_correct','tab:blue'),('both_wrong','tab:orange')]:
            for direction,style in [('actual_null','-'),('random_null','--')]:
                rs=sorted([r for r in cm if (r['domain'],r['condition'],r['split'],r['direction'])==(domain,condition,'collisions',direction)],key=lambda r:r['amplitude'])
                ax.plot([r['amplitude'] for r in rs],[100*r['accuracy'] for r in rs],style,color=color,label=condition+'/'+direction)
        ax.set_title(domain);ax.set_xlabel('Null perturbation norm / hidden RMS');ax.set_ylabel('Compound accuracy (%)');ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(ROOT/f'perturbations.{ext}',dpi=160)
    print(json.dumps({'status':'complete','contrasts':contrasts,'maximum_first_logit_change':max_preserved}),flush=True)

if __name__=='__main__':run()
