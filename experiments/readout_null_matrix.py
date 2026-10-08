"""Prospective matrix 2x2 directional-loss study with new sources and orbits."""
import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import algebra_relation_campaign as old
from . import algebra_relation_worlds as worlds
from .algebra_relation_feasibility import apply_encoding
from .relation_error_localization import row_projection
from .longrun_engine import atomic_json,atomic_torch
from .permworld_combinations import sha
from .two_step_relation_factorial import factorial_pairings,now,gram_cka
from .algebra_relation_campaign import Operators,kd,arrays,known_grade
CONFIG=Path('configs/readout_null_confirmation.json')
FLAGS={'full_correct':(1,1),'full_wrong':(0,0),'null_correct':(1,1),'null_wrong':(0,0)}

def projector(w):
    p=row_projection(w.astype(float));rank=int(round(np.trace(p)))
    return np.eye(len(p))-p,rank

def source_model(plan,root,index,device):return old.source_model(plan,root,index,device)

def setup():
    global Encoder
    torch.set_num_threads(1);apply_encoding('numeric_gelu');Encoder=old.Encoder
    common=json.loads(CONFIG.read_text());plan=json.loads(Path('configs/algebra_relation_common_v3.json').read_text())
    plan.update(source_seeds=common['source_seeds'],conditions=common['conditions'],fit_repeats=3)
    root=Path(common['output'])/'matrix'
    for name in ['', 'dataset','sources','fits','models','evaluations']:(root/name).mkdir(parents=True,exist_ok=True)
    signature={'plan':plan,'common_config_sha256':sha(CONFIG),'code_sha256':sha(__file__),
        'old_campaign_sha256':sha(old.__file__),'world_sha256':sha(worlds.__file__),'prediction':common['predictions'],
        'test_opening':'All12 formal fits at900 epochs before test inference; no outcome tuning.'}
    pp=root/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())['signature']==signature
    else:atomic_json(pp,{'registered_utc':now(),'new_compound_outcomes_observed':False,'signature':signature})
    return common,plan,root

def prepare(common,plan,root):
    if (root/'data_audit.json').exists():return
    domain='matrix';p=13;rng=np.random.default_rng(common['matrix']['data_seed'])
    previous=np.load('results/algebra_relation_v3/matrix/dataset/test.npz')['x'][:,0]
    excluded={worlds.key(tuple(x),domain,p) for x in previous}
    rows=[tuple(x) for x in product(range(p),repeat=4) if (x[0]*x[3]-x[1]*x[2])%p and worlds.key(tuple(x),domain,p) not in excluded]
    used=set();collision=worlds.choose_collision_pairs(rows,128,rng,domain,p,used)
    rng.shuffle(rows);iid=[]
    for x in rows:
        k=worlds.key(x,domain,p)
        if k not in used:
            iid.append(x);used.add(k)
            if len(iid)==256:break
    available=defaultdict(list)
    for x in rows:
        if worlds.key(x,domain,p) not in used:available[worlds.key(x,domain,p)].append(x)
    keys=list(available);rng.shuffle(keys)
    valkeys=set(keys[:192]);sourcekeys=set(keys[192:704])
    source=worlds.encode([available[k][0] for k in keys[192:704]],domain,p)
    validation=worlds.encode([available[k][0] for k in keys[:128]],domain,p)
    pilotval=worlds.encode([available[k][0] for k in keys[128:192]],domain,p)
    pool=[x for x in rows if worlds.key(x,domain,p) not in used|valkeys|sourcekeys]
    supports=[]
    for i in range(3):
        d=worlds.encode(worlds.paired_fit(pool,1024,np.random.default_rng(common['matrix']['data_seed']+101+i),domain,p),domain,p)
        pc,pi,eligible=factorial_pairings(np.ones(1024,int),d['labels'],261081501+i,3)
        assert eligible.all();d.update(pc=pc,pi=pi,eligible=eligible);supports.append(d)
    words=plan['words'];test=worlds.encode(iid+collision,domain,p,words)
    test.update(split=np.asarray([0]*256+[1]*256),pair_ids=np.asarray([-1]*256+[j for j in range(128) for _ in range(2)]))
    partitions=[source,validation,pilotval,*supports,test]
    sets=[{tuple(x) for x in d['x'].reshape(-1,4)} for d in partitions]
    train=set.union(sets[0],*sets[3:6]);val=sets[1]|sets[2];ts=sets[-1]
    assert not train&val and not train&ts and not val&ts
    assert not ({worlds.key(tuple(x),domain,p) for d in partitions for x in d['x'][:,0]} & excluded)
    for name,d in [('source',source),('validation',validation),('pilot_validation',pilotval),('test',test)]+[(f'support{i}',d) for i,d in enumerate(supports)]:
        np.savez_compressed(root/'dataset'/f'{name}.npz',**d)
    atomic_json(root/'data_audit.json',{'status':'complete','created_utc':now(),'excluded_old_test_orbits':len(excluded),
        'new_source_val_fit_test_inputs_disjoint':True,'all_wrong_pairings_answer_matched':True,
        'dataset_sha256':{str(p):sha(p) for p in (root/'dataset').glob('*.npz')}})

def fit(plan,root,index,condition,device,pilot=False):
    name='pilot' if pilot else f'n{index}_{condition}';record=root/'fits'/f'{name}.json'
    if record.exists():return json.loads(record.read_text())
    source_index=0 if pilot else index%3
    model=source_model(plan,root,source_index,device)
    with torch.no_grad():head=model.readout.weight.cpu().clone();head_bias=model.readout.bias.cpu().clone()
    for param in model.readout.parameters():param.requires_grad_(False)
    data,x,y=arrays(root/'dataset'/f'support{6 if pilot else index}.npz',plan['field_prime'],device)
    _,vx,vy=arrays(root/'dataset'/('pilot_validation.npz' if pilot else 'validation.npz'),plan['field_prime'],device)
    with torch.no_grad():model.eval();teacher_h=model(x);teacher=model.readout(teacher_h).detach()
    output_space=condition=='output_space';width=plan['field_prime'] if output_space else plan['hidden_width']
    projection,rank=projector(model.readout.weight.detach().cpu().numpy())
    q=torch.as_tensor(projection,device=device,dtype=torch.float32)
    nullmode=condition.startswith('null_')
    scale=float((teacher_h@q if nullmode else teacher_h).var(dim=(0,1)).mean());del teacher_h
    ops=Operators(width).to(device)
    opt=torch.optim.AdamW([{'params':list(model.encoder.parameters()),'lr':plan['source_learning_rate']},
                          {'params':list(ops.parameters()),'lr':plan['operator_learning_rate']}],weight_decay=1e-4)
    rng=np.random.default_rng(261078101+index*997+(9001 if pilot else 0));digest=sha256();curve=[]
    count=np.zeros(len(x),int);total_steps=plan['joint_epochs']*((len(x)+plan['batch_size']-1)//plan['batch_size']);step=0
    ca,cb=FLAGS[condition];pa=np.arange(len(x)) if ca else data['pc'];pb=np.arange(len(x)) if cb else data['pi']
    started=time.monotonic()
    for epoch in range(1,plan['joint_epochs']+1):
        order=rng.permutation(len(x));model.train()
        for start in range(0,len(x),plan['batch_size']):
            ids=order[start:start+plan['batch_size']];count[ids]+=1;digest.update(np.asarray([epoch],np.int64).tobytes()+ids.tobytes());step+=1
            ix=torch.as_tensor(ids,device=device);ia=torch.as_tensor(pa[ids],device=device);ib=torch.as_tensor(pb[ids],device=device)
            inputs=torch.stack([x[ix,0],x[ix,1],x[ix,2],x[ia,1],x[ib,2]],1)
            h=model(inputs);native=model.readout(h);v=native if output_space else h
            predictions=[ops(v[:,0],j) for j in range(2)]
            logits=[pred if output_space else model.readout(pred) for pred in predictions]
            ce=F.cross_entropy(native[:,:3].reshape(-1,plan['field_prime']),y[ix].reshape(-1))
            native_kd=kd(native[:,:3].reshape(-1,plan['field_prime']),teacher[ix].reshape(-1,plan['field_prime']),plan['temperature'])
            op_kd=sum(kd(logits[j],teacher[ix,j+1],plan['temperature']) for j in range(2))/2
            geom=sum(((predictions[j]-v[:,j+3])@q if nullmode else predictions[j]-v[:,j+3]).square().mean() for j in range(2))/(2*max(scale,1e-6))
            weight=0 if condition=='no_geometry' else plan['geometry_weight']
            loss=plan['native_ce_weight']*ce+plan['native_kd_weight']*native_kd+plan['operator_kd_weight']*op_kd+weight*geom
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters())+list(ops.parameters()),1.)
            factor=.2+.8*(1+np.cos(np.pi*step/total_steps))/2
            opt.param_groups[0]['lr']=plan['source_learning_rate']*factor;opt.param_groups[1]['lr']=plan['operator_learning_rate']*factor;opt.step()
        if epoch in [100,300,900]:
            grade=known_grade(model,ops,vx,vy,output_space);curve.append({'epoch':epoch,**grade})
            print(json.dumps({'domain':root.name,'fit':name,'epoch':epoch,**grade,'elapsed':time.monotonic()-started}),flush=True)
            atomic_json(root/'current_job.json',{'fit':name,'epoch':epoch,'updated_utc':now()})
    assert np.all(count==plan['joint_epochs'])
    torch.testing.assert_close(head,model.readout.weight.cpu(),atol=0,rtol=0)
    torch.testing.assert_close(head_bias,model.readout.bias.cpu(),atol=0,rtol=0)
    file=root/'models'/f'{name}.pt';atomic_torch(file,{'model':model.state_dict(),'operators':ops.state_dict(),'output_space':output_space})
    result={'status':'complete','completed_utc':now(),'replicate':index,'condition':condition,'source_index':source_index,
            'epochs':plan['joint_epochs'],'updates':step,'anchor_exposures':int(count.sum()),'native_forward_rows':5*int(count.sum()),
            'schedule_sha256':digest.hexdigest(),'checkpoint_sha256':sha(file),'curve':curve,'frozen_numeric_readout_verified':True,
            'geometry_space':'null' if nullmode else 'full','normalization_scale':scale,'readout_rank':rank,'output_space':output_space,'operator_parameters':sum(p.numel() for p in ops.parameters()),
            'source_checkpoint_sha256':sha(root/'sources'/f's{source_index}.pt')}
    atomic_json(record,result);return result


@torch.no_grad()
def evaluate(plan,root):
    files=[root/'fits'/f'n{i}_{c}.json' for i in range(3) for c in FLAGS]
    assert all(p.exists() for p in files)
    atomic_json(root/'test_opened.json',{'opened_utc':now(),'all12_fixed_budget_fits_complete':True,'fit_record_sha256':{str(p):sha(p) for p in files}})
    data,x,y=arrays(root/'dataset/test.npz',13,'cuda');rows=[]
    for i in range(3):
        for c in FLAGS:
            name=f'n{i}_{c}';state=torch.load(root/'models'/f'{name}.pt',map_location='cpu',weights_only=True)
            model=Encoder(13,128).cuda();model.load_state_dict(state['model']);model.eval();ops=Operators(128).cuda();ops.load_state_dict(state['operators'])
            h=model(x);w=model.readout.weight.cpu().numpy().astype(float);bw=model.readout.bias.cpu().numpy().astype(float)
            hn=h.cpu().numpy().astype(float);a=ops.maps[0].weight.cpu().numpy().T.astype(float);b=ops.maps[1].weight.cpu().numpy().T.astype(float)
            ba=ops.maps[0].bias.cpu().numpy().astype(float);bb=ops.maps[1].bias.cpu().numpy().astype(float)
            first=hn[:,0]@a+ba;pred=first@b+bb;forced=hn[:,1]@b+bb
            ans=(pred@w.T+bw).argmax(1);truth=data['labels'][:,3];hit=ans==truth
            np.savez_compressed(root/'evaluations'/f'{name}.npz',native=hn.astype(np.float32),ab_pred=pred,ab_answers=ans,
                rho_a=a,rho_b=b,bias_a=ba,bias_b=bb,readout_weight=w,readout_bias=bw)
            for sid,split in [(0,'iid'),(1,'collisions')]:
                use=data['split']==sid;pairs=data['pair_ids'][use]
                rows.append({'replicate':i,'source':i,'condition':c,'split':split,'word':'ab','accuracy':float(hit[use].mean()),
                    'pair_both_correct':float(np.mean([hit[use][pairs==p].all() for p in np.unique(pairs)])) if sid else None,
                    'prediction_nmse':float(np.square(pred[use]-hn[use,3]).sum()/np.square(hn[use,3]-hn[use,0]).sum()),
                    'cka':gram_cka(pred[use],hn[use,3]),
                    'true_intermediate_accuracy':float(((forced@w.T+bw).argmax(1)[use]==truth[use]).mean()),
                    'direct_native_accuracy':float(((hn[:,3]@w.T+bw).argmax(1)[use]==truth[use]).mean())})
    atomic_json(root/'evaluation_records.json',{'status':'complete','completed_utc':now(),'records':rows})
    atomic_json(root/'completion.json',{'status':'trained_and_evaluated','completed_utc':now(),'fits':12,'source_models':3})

def run():
    common,plan,root=setup();prepare(common,plan,root)
    for i in range(3):source_model(plan,root,i,'cuda')
    for i in range(3):
        for c in FLAGS:fit(plan,root,i,c,'cuda')
    evaluate(plan,root)

if __name__=='__main__':run()
