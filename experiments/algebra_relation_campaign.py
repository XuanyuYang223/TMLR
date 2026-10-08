"""Forward-only, budget-matched intervention assays in two exact worlds."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import algebra_relation_worlds as worlds
from .longrun_engine import atomic_json, atomic_torch
from .permworld_combinations import sha
from .two_step_relation_factorial import factorial_pairings, now, gram_cka
from .two_step_relation_final_report import table
from .overnight_inverse_statistics_verify import bootstrap_interval

CONFIG=Path('configs/algebra_relation_common.json')
ROOT=Path('results/algebra_relation_followup')
FLAGS={'both_correct':(1,1),'a_correct_b_wrong':(1,0),'a_wrong_b_correct':(0,1),'both_wrong':(0,0),
       'no_geometry':(1,1),'output_space':(1,1)}


class Encoder(nn.Module):
    def __init__(self, p, width):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(4*p+4,width),nn.GELU(),nn.Linear(width,width),nn.LayerNorm(width))
        self.readout=nn.Linear(width,p)

    def forward(self,x):
        return self.encoder(x)


class Operators(nn.Module):
    def __init__(self,width):
        super().__init__();self.maps=nn.ModuleList([nn.Linear(width,width) for _ in range(2)])
        with torch.no_grad():
            for layer in self.maps:layer.weight.copy_(torch.eye(width));layer.bias.zero_()

    def forward(self,x,index):return self.maps[index](x)


def kd(x,y,t):
    return F.kl_div(F.log_softmax(x/t,-1),F.softmax(y/t,-1),reduction='batchmean')*t*t


def register(domain):
    plan=json.loads(CONFIG.read_text());root=ROOT/domain;root.mkdir(parents=True,exist_ok=True)
    signature={'plan':plan,'config_sha256':sha(CONFIG),'code_sha256':sha(__file__),
               'world_code_sha256':sha(worlds.__file__),'domain':domain}
    file=root/'protocol.json'
    if file.exists():assert json.loads(file.read_text())['signature']==signature
    else:
        atomic_json(file,{'registered_utc':now(),'compound_test_outcomes_observed':False,'signature':signature})
        (root/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (root/'world_source_snapshot.py').write_bytes(Path(worlds.__file__).read_bytes())
    for name in ['dataset','sources','fits','models','evaluations']:(root/name).mkdir(exist_ok=True)
    return plan,root


def prepare(plan,root,domain):
    if (root/'data_audit.json').exists():return
    data=worlds.create_world(domain,plan['field_prime'],261077001+(domain=='polynomial')*1000,
                            plan['fit_anchors'],plan['source_anchors'],plan['validation_anchors'],
                            plan['iid_examples'],plan['collision_pairs'])
    for name in ['source','validation','pilot_validation','test']:
        np.savez_compressed(root/'dataset'/f'{name}.npz',**data[name])
    for i,support in enumerate(data['supports']):
        pc,pi,eligible=factorial_pairings(np.ones(len(support['labels']),int),support['labels'],261077501+i,3)
        assert eligible.all()
        support.update(pc=pc,pi=pi,eligible=eligible)
        np.savez_compressed(root/'dataset'/f'support{i}.npz',**support)
    atomic_json(root/'data_audit.json',{'created_utc':now(),**data['audit'],'words':list(data['words']),
                'every_fit_anchor_pairing_eligible':True,'full_epoch_wrong_marginals_identical':True,
                'discrete_double_wrong_return_to_base_excluded':True,
                'dataset_sha256':{str(p):sha(p) for p in (root/'dataset').glob('*.npz')}})


def arrays(path,p,device):
    d=dict(np.load(path));f=torch.as_tensor(worlds.feature(d['x'],p),device=device)
    y=torch.as_tensor(d['labels'],device=device)
    return d,f,y


@torch.no_grad()
def known_grade(model,ops,features,labels,output_space=False):
    model.eval();h=model(features);native=model.readout(h)
    vectors=native if output_space else h
    scores=[]
    for j in range(2):
        pred=ops(vectors[:,0],j)
        logits=pred if output_space else model.readout(pred)
        scores.append(float((logits.argmax(-1)==labels[:,j+1]).float().mean()))
    return {'native':(native.argmax(-1)==labels).float().mean(0).cpu().tolist(),'generators':scores}


def source_model(plan,root,index,device):
    torch.manual_seed(plan['source_seeds'][index])
    model=Encoder(plan['field_prime'],plan['hidden_width']).to(device)
    file=root/'sources'/f's{index}.pt'
    if file.exists():model.load_state_dict(torch.load(file,map_location='cpu',weights_only=True)['model']);return model
    _,x,y=arrays(root/'dataset/source.npz',plan['field_prime'],device)
    x,y=x.reshape(-1,x.shape[-1]),y.reshape(-1)
    _,vx,vy=arrays(root/'dataset/pilot_validation.npz',plan['field_prime'],device)
    opt=torch.optim.AdamW(model.parameters(),lr=plan['source_pretrain_lr'],weight_decay=1e-4)
    rng=np.random.default_rng(plan['source_seeds'][index]+991);curve=[]
    for epoch in range(1,plan['source_epochs']+1):
        model.train()
        for start in range(0,len(x),plan['source_batch_size']):
            if start==0:order=rng.permutation(len(x))
            ids=torch.as_tensor(order[start:start+plan['source_batch_size']],device=device)
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model.readout(model(x[ids])),y[ids]);loss.backward();opt.step()
        if epoch%100==0:
            with torch.no_grad():model.eval();native=(model.readout(model(vx)).argmax(-1)==vy).float().mean(0).cpu().tolist()
            curve.append({'epoch':epoch,'native':native})
            print(json.dumps({'domain':root.name,'source':index,'epoch':epoch,'native':native}),flush=True)
    atomic_torch(file,{'model':model.state_dict(),'source_seed':plan['source_seeds'][index]})
    atomic_json(root/'sources'/f's{index}.json',{'status':'complete','completed_utc':now(),'source_seed':plan['source_seeds'][index],
                'epochs':plan['source_epochs'],'curve':curve,'checkpoint_sha256':sha(file)})
    return model


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
    scale=float((teacher if output_space else teacher_h).var(dim=(0,1)).mean());del teacher_h
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
            geom=sum((predictions[j]-v[:,j+3]).square().mean() for j in range(2))/(2*max(scale,1e-6))
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
            'output_space':output_space,'operator_parameters':sum(p.numel() for p in ops.parameters()),
            'source_checkpoint_sha256':sha(root/'sources'/f's{source_index}.pt')}
    atomic_json(record,result);return result


@torch.no_grad()
def evaluate(plan,root,device):
    files=[root/'fits'/f'n{i}_{c}.json' for i in range(6) for c in plan['conditions']]
    assert all(p.exists() for p in files)
    atomic_json(root/'test_opened.json',{'opened_utc':now(),'all36_fixed_budget_fits_complete':True,
                'fit_record_sha256':{p.name:sha(p) for p in files}})
    data,x,y=arrays(root/'dataset/test.npz',plan['field_prime'],device);rows=[];replay_checks=0
    word_indices={w:i for i,w in enumerate(plan['words'])}
    for i in range(6):
        for c in plan['conditions']:
            name=f'n{i}_{c}';state=torch.load(root/'models'/f'{name}.pt',map_location='cpu',weights_only=True)
            model=Encoder(plan['field_prime'],plan['hidden_width']).to(device);model.load_state_dict(state['model']);model.eval()
            space=state['output_space'];width=plan['field_prime'] if space else plan['hidden_width']
            ops=Operators(width).to(device);ops.load_state_dict(state['operators']);h=model(x);native=model.readout(h);v=native if space else h
            h_numpy=v.cpu().numpy().astype(float);predictions={};saved={'native':h_numpy.astype(np.float32)}
            npmaps=[(layer.weight.cpu().numpy().T.astype(float),layer.bias.cpu().numpy().astype(float)) for layer in ops.maps]
            for word in plan['words'][1:]:
                pred=v[:,0]
                independent=h_numpy[:,0].copy()
                for letter in word:
                    j=letter=='b';pred=ops(pred,j);independent=independent@npmaps[j][0]+npmaps[j][1]
                np.testing.assert_allclose(pred.cpu().numpy(),independent,atol=2e-3,rtol=2e-4);replay_checks+=1
                logits=pred if space else model.readout(pred)
                ans=logits.argmax(-1).cpu().numpy();target=data['labels'][:,word_indices[word]];hit=ans==target
                forced=None
                if len(word)==2:
                    mid=word_indices[word[0]];forced=ops(v[:,mid],word[1]=='b')
                    forced=(forced if space else model.readout(forced)).argmax(-1).cpu().numpy()
                saved[word+'_pred']=pred.cpu().numpy();saved[word+'_answers']=ans
                prediction=pred.cpu().numpy().astype(float);truth=h_numpy[:,word_indices[word]]
                for sid,label in [(0,'iid'),(1,'collisions')]:
                    use=data['split']==sid;denom=np.square(truth[use]-h_numpy[use,0]).sum()
                    row={'replicate':i,'source_index':i%3,'condition':c,'word':word,'split':label,'representation_space':'output' if space else 'hidden',
                         'accuracy':float(hit[use].mean()),'pair_both_correct':None,
                         'prediction_nmse':float(np.square(prediction[use]-truth[use]).sum()/max(denom,1e-30)),
                         'cka':gram_cka(prediction[use],truth[use]),
                         'true_intermediate_accuracy':float((forced[use]==target[use]).mean()) if forced is not None else None,
                         'direct_native_accuracy':float((native[:,word_indices[word]].argmax(-1).cpu().numpy()[use]==target[use]).mean())}
                    if sid==1:
                        pairs=data['pair_ids'][use];row['pair_both_correct']=float(np.mean([hit[use][pairs==p].all() for p in np.unique(pairs)]))
                    rows.append(row)
            # Compare both operation orders against exactly the same AB target.
            for sid,label in [(0,'iid'),(1,'collisions')]:
                use=data['split']==sid;answers=saved['ba_answers'];truth=data['labels'][:,word_indices['ab']]
                mask=use&(data['labels'][:,word_indices['ab']]!=data['labels'][:,word_indices['ba']])
                rows.append({'replicate':i,'condition':c,'split':label,'word':'wrong_order_against_ab',
                             'accuracy':float((answers[use]==truth[use]).mean()),
                             'order_discriminating_examples':int(mask.sum()),
                             'order_discriminating_accuracy':float((answers[mask]==truth[mask]).mean()) if mask.any() else None})
            np.savez_compressed(root/'evaluations'/f'{name}.npz',**saved)
    means=[];contrasts=[]
    for c in plan['conditions']:
        for split in ['iid','collisions']:
            selected=[r for r in rows if r['condition']==c and r['word']=='ab' and r['split']==split]
            means.append({'condition':c,'split':split,**{k:float(np.mean([r[k] for r in selected])) for k in
                         ['accuracy','prediction_nmse','cka','true_intermediate_accuracy','direct_native_accuracy']},
                         'pair_both_correct':None if split=='iid' else float(np.mean([r['pair_both_correct'] for r in selected]))})
            if c!='both_correct':
                values=[]
                for i in range(6):
                    a=next(r['accuracy'] for r in rows if (r['replicate'],r['condition'],r['word'],r['split'])==(i,'both_correct','ab',split))
                    b=next(r['accuracy'] for r in rows if (r['replicate'],r['condition'],r['word'],r['split'])==(i,c,'ab',split))
                    values.append(100*(a-b))
                source=[float(np.mean([values[j],values[j+3]])) for j in range(3)]
                contrasts.append({'condition':c,'split':split,'mean_pp':float(np.mean(values)),
                                  'paired_values':values,'three_source_bootstrap_95_pp':bootstrap_interval(source)})
    grades=[json.loads(p.read_text())['curve'][-1] for p in files if 'both_correct' in p.name]
    source_grades=[[float(np.mean([g['generators'][j] for g in [grades[k],grades[k+3]]])) for j in range(2)] for k in range(3)]
    summary={'status':'complete','completed_utc':now(),'domain':root.name,'records':rows,'means':means,'contrasts':contrasts,
             'source_correct_generator_scores':source_grades,'all_correct_source_generators_pass_gate':all(min(g)>=.9 for g in source_grades),
             'independent_affine_prediction_replay_checks':replay_checks,'formal_fits':36,'source_models':3,'same_900_epoch_budget':True}
    atomic_json(root/'summary.json',summary)
    selected=[r for r in means if r['split']=='collisions']
    (root/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:16px/1.7 system-ui;max-width:1150px;margin:30px auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}</style><h1>'+root.name+'：正确配对与留出组合</h1><p>预测只接收原输入表征和学到的两个算子；程序只提供操作顺序。36组全部固定900轮，三个来源复用为六个拟合重复。输出空间组是参数量不同的信息诊断。</p>'+table(['条件','碰撞AB准确率','配对双正确','预测误差','CKA','真实中间诊断'],[[r['condition'],f'{100*r["accuracy"]:.2f}%',f'{100*r["pair_both_correct"]:.2f}%',f'{r["prediction_nmse"]:.4f}',f'{r["cka"]:.4f}',f'{100*r["true_intermediate_accuracy"]:.2f}%'] for r in selected])+'<p><a href="summary.json">逐模型结果</a> · <a href="data_audit.json">输入划分</a> · <a href="protocol.json">预先登记</a></p>')


def run(domain):
    torch.set_num_threads(1);plan,root=register(domain);device='cuda' if torch.cuda.is_available() else 'cpu'
    try:
        atomic_json(root/'state.json',{'status':'running','stage':'data_and_known_pilot','updated_utc':now()})
        prepare(plan,root,domain)
        pilot=fit(plan,root,0,'both_correct',device,True);grade=pilot['curve'][-1]
        feasible=min(grade['native'])>=plan['pilot_known_gate']['native'] and min(grade['generators'])>=plan['pilot_known_gate']['generators']
        atomic_json(root/'pilot_gate.json',{'status':'complete','feasible':feasible,'grade':grade,'compound_test_not_opened':True})
        if not feasible:
            atomic_json(root/'state.json',{'status':'known_pilot_failed','updated_utc':now(),'compound_test_closed':True});return
        for i in range(6):
            for condition in plan['conditions']:
                atomic_json(root/'state.json',{'status':'running','stage':'fixed_formal_fit','replicate':i,'condition':condition,'updated_utc':now()})
                fit(plan,root,i,condition,device)
        evaluate(plan,root,device)
        atomic_json(root/'completion.json',{'status':'complete','completed_utc':now(),
                    'artifact_sha256':{str(p):sha(p) for p in root.rglob('*') if p.is_file() and p.name not in ['state.json','current_job.json','completion.json']}})
        atomic_json(root/'state.json',{'status':'complete','updated_utc':now()})
    except BaseException as e:
        atomic_json(root/'state.json',{'status':'failed','updated_utc':now(),'reason':repr(e)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('domain',choices=['matrix','polynomial','both']);args=parser.parse_args()
    if args.domain=='both':
        run('matrix')
        if json.loads((ROOT/'matrix/state.json').read_text())['status']=='complete':run('polynomial')
    else:run(args.domain)
