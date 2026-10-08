"""Train a source predictor with only e/C/I orbit states and known edges.

The learned relation condition is an explicitly supervised operator assay,
not a spontaneous-equivariance claim. Compound endpoints never enter source
training, validation, or the training loss. Source labels and affine maps
are different sources of supervision and are reported separately.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .algebra_structure_replication import collect_prior_inputs
from .longrun_engine import atomic_json,atomic_torch,learning_rate
from .longrun_transfer import make_model
from .native_confirmation import setup
from .permworld_combinations import prompts,sha
from .permutation_audit import transform
from .representation_algebra import ACTION_NAMES


CONFIG='configs/algebra_hidden_relations.json'


def now():return datetime.now(timezone.utc).isoformat()


def load_plan():
    plan=json.loads(Path(CONFIG).read_text());config=json.loads(Path(plan['base_config']).read_text())
    config.update(dropout=0.,weight_decay=plan['weight_decay'])
    return plan,config,Path(plan['output'])


def visible_key(orbit,function):
    return tuple(function(orbit[j]) for j in [0,1,4])


def collision_pair(rng,n,seen,pending,function):
    while True:
        base=tuple(map(int,rng.permutation(n)+1));orbit=[transform(base,a) for a in ACTION_NAMES]
        if len(set(orbit))!=8 or any(p in seen for p in orbit):continue
        key=visible_key(orbit,function);value=function(orbit[5]);old=pending.get(key)
        if old is not None and any(p in seen for p in old):pending.pop(key);old=None
        if old is None:pending[key]=orbit;continue
        if function(old[5])==value or set(old)&set(orbit):continue
        pending.pop(key);seen.update(old);seen.update(orbit)
        return old,orbit


def create_data(plan,config,root,tokens,one_line,functions):
    previous=json.loads(Path('configs/algebra_structure_replication.json').read_text())
    excludes=previous['excluded_datasets']+[str(Path(plan['previous_replication'])/'probe_dataset.npz')]
    seen=collect_prior_inputs(excludes);prior_count=len(seen);rng=np.random.default_rng(plan['data_seed'])
    f=functions[plan['source_task']];width=2*plan['lengths'][1]+4
    def encode(orbit,n):
        rows=[]
        for p in orbit:
            row=[tokens['<BOS>'],tokens['<SIZE>'],n]+[tokens[t] for t in one_line(p)]
            rows.append(row+[tokens['<PAD>']]*(width-len(row)))
        return rows
    def fresh(n):
        while True:
            base=tuple(map(int,rng.permutation(n)+1));orbit=[transform(base,a) for a in ACTION_NAMES]
            if len(set(orbit))==8 and not any(p in seen for p in orbit):seen.update(orbit);return orbit
    train_x=[];train_y=[];train_n=[];train_raw=[]
    probe_x=[];probe_y=[];probe_n=[];probe_raw=[];split=[];pair=[]
    for n in range(plan['lengths'][0],plan['lengths'][1]+1):
        for _ in range(plan['training_orbits_per_length']):
            orbit=fresh(n);encoded=encode(orbit,n)
            train_x.append([encoded[j] for j in plan['observed_actions']]);train_y.append([f(orbit[j]) for j in plan['observed_actions']]);train_n.append(n)
            train_raw.append([list(p)+[0]*(plan['lengths'][1]-n) for p in orbit])
    for k,(_,count) in enumerate(plan['probe_orbits_per_length'].items()):
        for n in range(plan['lengths'][0],plan['lengths'][1]+1):
            for _ in range(count):
                orbit=fresh(n);probe_x.append(encode(orbit,n));probe_y.append([f(p) for p in orbit]);probe_n.append(n)
                probe_raw.append([list(p)+[0]*(plan['lengths'][1]-n) for p in orbit]);split.append(k);pair.append(-1)
    pair_id=0
    for n in range(plan['lengths'][0],plan['lengths'][1]+1):
        pending={}
        for _ in range(plan['collision_pairs_per_length']):
            both=collision_pair(rng,n,seen,pending,f)
            for orbit in both:
                probe_x.append(encode(orbit,n));probe_y.append([f(p) for p in orbit]);probe_n.append(n)
                probe_raw.append([list(p)+[0]*(plan['lengths'][1]-n) for p in orbit]);split.append(3);pair.append(pair_id)
            pair_id+=1
    source={'train_input':np.asarray(train_x,dtype=np.int64),'train_labels':np.asarray(train_y,dtype=np.int64),'train_lengths':np.asarray(train_n,dtype=np.int64)}
    probe={'input':np.asarray(probe_x,dtype=np.int64),'labels':np.asarray(probe_y,dtype=np.int64),'lengths':np.asarray(probe_n,dtype=np.int64),
        'permutations':np.asarray(probe_raw,dtype=np.int64),'split':np.asarray(split,dtype=np.int64),'pair_ids':np.asarray(pair,dtype=np.int64)}
    # This is the only validation data loaded by the source trainer.
    val=probe['split']==1
    source.update(validation_input=probe['input'][val][:,plan['observed_actions']],validation_labels=probe['labels'][val][:,plan['observed_actions']],validation_lengths=probe['lengths'][val])
    np.savez_compressed(root/'source_data.npz',**source)
    np.savez_compressed(root/'training_orbit_audit.npz',permutations=np.asarray(train_raw,dtype=np.int64),lengths=source['train_lengths'])
    np.savez_compressed(root/'probe_dataset.npz',**probe)
    audit={'excluded_distinct_old_inputs':prior_count,'excluded_datasets':{p:sha(p) for p in excludes},
        'source_anchors':len(train_x),'source_observed_input_states':len(train_x)*3,
        'withheld_source_orbit_states':len(train_x)*5,'probe_split_orbits':[int(np.sum(probe['split']==k)) for k in range(4)],
        'collision_pairs':pair_id,'collision_visible_answers_equal':True,'collision_CI_and_ICI_answers_differ':True,
        'answer_length_task_ceiling_accuracy':.5,'answer_only_pair_both_correct_ceiling':0.,
        'scope':'Hidden endpoint states are excluded from source input and label archives; exact mathematical generator definitions imply composition. Collision test is selected by answer equality and hidden-answer difference.'}
    atomic_json(root/'dataset_audit.json',audit)


def initialize():
    plan,config,root=load_plan();root.mkdir(exist_ok=True,parents=True)
    for name in ['source','checkpoints','features','evaluations','arrays']:(root/name).mkdir(exist_ok=True)
    _,functions,tokens,one_line=setup(config)
    core=['experiments/hidden_relation_train.py','experiments/longrun_transfer.py','experiments/longrun_attention.py','experiments/permworld_combinations.py','experiments/permutation_audit.py','experiments/representation_algebra.py']
    signature={'plan_sha256':sha(CONFIG),'core_sha256':{p:sha(p) for p in core},
        'previous_protocol_sha256':sha(Path(plan['previous_replication'])/'protocol.json'),
        'upstream_sha256':{f:sha(Path(config['repository'])/'src/neurips_permutations'/f) for f in ['models.py','math_ops.py','passage.py']},'plan':plan}
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['signature']==signature
    else:atomic_json(path,{'registered_utc':now(),'signature':signature,'new_source_models':0})
    if not (root/'source_data.npz').exists():create_data(plan,config,root,tokens,one_line,functions)
    digest={p:sha(root/p) for p in ['source_data.npz','training_orbit_audit.npz','probe_dataset.npz']}
    path=root/'data_hashes.json'
    if path.exists():assert json.loads(path.read_text())==digest
    else:atomic_json(path,digest)
    return plan,config,root,tokens


class AffineOperators(nn.Module):
    def __init__(self,width):
        super().__init__();self.offset=nn.Parameter(torch.zeros(2,width,width));self.bias=nn.Parameter(torch.zeros(2,width))
    def forward(self,x,index):return x+x @ self.offset[index]+self.bias[index]


def query_hidden(model,inputs,lengths,task,tokens):
    ids,mask,position=prompts(inputs,lengths,[task],tokens);h,valid=model._embed_inputs(ids,mask)
    for block in model.blocks:h=block(h,valid)
    h=model.final_norm(h)
    return h[torch.arange(len(ids),device=ids.device),position]


def relation_loss(h,operators,order):
    base=h[:,0];c=h[order,1];i=h[order,2]
    losses=[F.mse_loss(operators(base,0).float(),c.float()),F.mse_loss(operators(c,0).float(),base.float()),
        F.mse_loss(operators(base,1).float(),i.float()),F.mse_loss(operators(i,1).float(),base.float())]
    scale=h.float().detach().var(dim=(0,1),unbiased=False).mean().clamp_min(.01)
    return torch.stack(losses).mean()/scale


@torch.no_grad()
def validate(model,data,plan,tokens):
    model.eval();x=data['validation_input'];n=data['validation_lengths'];y=data['validation_labels'];pred=[]
    for start in range(0,len(x),128):
        b=x[start:start+128];ns=n[start:start+128]
        h=query_hidden(model,b.reshape(-1,b.shape[-1]),ns.repeat_interleave(3),plan['source_task'],tokens)
        pred.append(model.lm_head(h)[:,:31].argmax(-1).reshape(-1,3))
    return (torch.cat(pred)==y).float().mean(0).cpu().tolist()


def train_one(plan,config,root,tokens,data,raw,seed,condition,device):
    name=f'{condition}_s{seed}';record=root/'source'/f'{name}.json';cp=root/'checkpoints'/f'{name}.pt'
    if record.exists():
        r=json.loads(record.read_text());assert r['status']=='complete' and sha(cp)==r['checkpoint_sha256'];return
    model=make_model(config,plan['architecture'],seed,device);ops=AffineOperators(plan['architecture']['d_model']).to(device)
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.parameters())).hexdigest()
    optimizer=torch.optim.AdamW(list(model.parameters())+list(ops.parameters()),lr=plan['source_learning_rate'],weight_decay=plan['weight_decay'])
    rng=np.random.default_rng(seed+2026100662);wrong_rng=np.random.default_rng(seed+2026100663)
    buckets={n:np.flatnonzero(raw['train_lengths']==n) for n in range(plan['lengths'][0],plan['lengths'][1]+1)}
    fingerprint={'protocol_sha256':sha(root/'protocol.json'),'data_sha256':sha(root/'source_data.npz'),'condition':condition,'seed':seed}
    step=0;curve=[];digest=hashlib.sha256();seen=np.zeros(len(raw['train_lengths']),bool)
    if cp.exists():
        state=torch.load(cp,weights_only=True,map_location=device);assert state['fingerprint']==fingerprint
        model.load_state_dict(state['model']);ops.load_state_dict(state['operators']);optimizer.load_state_dict(state['optimizer']);step=state['step'];curve=state['curve']
        for _ in range(step):
            n=int(rng.integers(plan['lengths'][0],plan['lengths'][1]+1));ids=rng.choice(buckets[n],plan['anchors_per_update'],replace=False)
            wrong_rng.integers(1,plan['anchors_per_update']);digest.update(np.array([n],dtype=np.int64).tobytes()+ids.tobytes());seen[ids]=True
        assert rng.bit_generator.state==state['core_rng'] and wrong_rng.bit_generator.state==state['wrong_rng'] and digest.hexdigest()==state['sample_sha256']
    started=time.monotonic()
    while step<plan['source_steps']:
        step+=1;n=int(rng.integers(plan['lengths'][0],plan['lengths'][1]+1));ids=rng.choice(buckets[n],plan['anchors_per_update'],replace=False)
        shift=int(wrong_rng.integers(1,plan['anchors_per_update']));digest.update(np.array([n],dtype=np.int64).tobytes()+ids.tobytes());seen[ids]=True
        x=data['train_input'][ids];ns=data['train_lengths'][ids].repeat_interleave(3);labels=data['train_labels'][ids].reshape(-1)
        model.train();optimizer.zero_grad(set_to_none=True)
        for g in optimizer.param_groups:g['lr']=learning_rate(step,plan['source_steps'],plan)
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
            h=query_hidden(model,x.reshape(-1,x.shape[-1]),ns,plan['source_task'],tokens)
            ce=F.cross_entropy(model.lm_head(h)[:,:31].float(),labels)
            order=torch.arange(len(x),device=device)
            if condition=='shuffled_relations':order=order.roll(shift)
            geom=relation_loss(h.reshape(len(x),3,-1),ops,order)
            weight=0. if condition=='ordinary' else plan['relation_loss_weight']*min(1.,step/plan['relation_loss_ramp_steps'])
            loss=ce+weight*geom
        loss.backward();torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(ops.parameters()),1.);optimizer.step()
        if step%plan['record_every']==0 or step==plan['source_steps']:
            acc=validate(model,data,plan,tokens);progress={'job':name,'step':step,'planned_steps':plan['source_steps'],'observed_validation_accuracy':acc,
                'source_ce':float(ce.detach().cpu()),'relation_loss':float(geom.detach().cpu())};curve.append(progress)
            atomic_torch(cp,{'model':model.state_dict(),'operators':ops.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                'curve':curve,'core_rng':rng.bit_generator.state,'wrong_rng':wrong_rng.bit_generator.state,'sample_sha256':digest.hexdigest(),'fingerprint':fingerprint})
            atomic_json(root/'current_job.json',progress);print(json.dumps(progress),flush=True)
    atomic_json(record,{'status':'complete','seed':seed,'condition':condition,'steps':step,'source_labels_per_update':96,
        'source_label_exposures':step*96,'known_directed_edges_per_update':0 if condition=='ordinary' else 4*plan['anchors_per_update'],
        'source_input_actions':plan['observed_actions'],'observed_validation_accuracy':acc,'unique_source_anchors_seen':int(seen.sum()),
        'initial_parameter_sha256':initial,'sample_sha256':digest.hexdigest(),'checkpoint_sha256':sha(cp),
        'core_rng':rng.bit_generator.state,'wrong_rng':wrong_rng.bit_generator.state,'curve':curve,'completed_utc':now(),'fingerprint':fingerprint})
    del model,ops,optimizer
    if device=='cuda':torch.cuda.empty_cache()


def run():
    plan,config,root,tokens=initialize();torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    with np.load(root/'source_data.npz') as a:raw={k:a[k] for k in a.files}
    data={k:torch.tensor(v,device=device) for k,v in raw.items()};done=0
    for seed in plan['source_seeds']:
        for condition in plan['conditions']:
            atomic_json(root/'state.json',{'status':'source_training','completed':done,'planned':9,'seed':seed,'condition':condition,'updated_utc':now()})
            train_one(plan,config,root,tokens,data,raw,seed,condition,device);done+=1
    atomic_json(root/'state.json',{'status':'sources_complete','source_models':done,'updated_utc':now()})


if __name__=='__main__':run()
