"""Independent closed-form replay of the new transported-error diagnosis."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import time

import numpy as np

from .until_10_verify import ROOT,DEADLINE,read,archive,digest,save,homogeneous,action_order


def verify(path):
    rec=read(path)
    h=archive(rec['feature_path'])['source_query_concat'][:,:,-1].astype(np.float64)
    data=archive(rec['data_path'])
    maps=archive(rec['map_path'])
    assert digest(rec['map_path'])==rec['map_sha256']
    assert digest(rec['feature_path'])==rec['feature_sha256']
    assert maps['mean_vectors'].size==0
    saved_path=path.parent.parent/'arrays'/(rec['source']+'.npz')
    assert digest(saved_path)==rec['prediction_archive_sha256']
    saved=archive(saved_path)
    width=h.shape[-1]
    rc,ri=maps['rho_c'],maps['rho_i']
    bc,bi=maps['bias_c'],maps['bias_i']
    total=0
    for split,tag in [(2,'iid'),(3,'answer_collisions')]:
        use=data['split']==split
        x=h[use]
        for word in ['ci','ici']:
            destination=int(action_order(word)[0])
            product=homogeneous(maps,word,width)
            prediction=np.column_stack([x[:,0],np.ones(len(x))])@product
            prediction=prediction[:,:width]
            target=x[:,destination]
            if word=='ci':
                first=x[:,0]@rc+bc-x[:,1]
                second=x[:,1]@ri+bi-x[:,5]
                components=[first@ri,second]
            else:
                first=x[:,0]@ri+bi-x[:,4]
                second=x[:,4]@rc+bc-x[:,6]
                third=x[:,6]@ri+bi-x[:,2]
                components=[first@(rc@ri),second@ri,third]
            np.testing.assert_allclose(sum(components),prediction-target,atol=1e-7,rtol=1e-7)
            denom=np.square(target-x[:,0]).sum()
            row=next(r for r in rec['rows'] if r['kind']=='composed_from_base' and r['split']==tag and r['word']==word)
            energies=[np.square(c).sum()/denom for c in components]
            np.testing.assert_allclose(energies,row['transported_local_error_energy_over_final_displacement'],atol=1e-10,rtol=1e-9)
            cross=2*sum((components[i]*components[j]).sum() for i in range(len(components)) for j in range(i+1,len(components)))/denom
            np.testing.assert_allclose(cross,row['cross_terms_over_final_displacement'],atol=1e-10,rtol=1e-8)
            np.testing.assert_allclose(np.square(prediction-target).sum()/denom,row['hidden_displacement_nmse'],atol=1e-10,rtol=1e-9)
            labels=data['labels'][use,destination]
            answers=(prediction@maps['readout_weight'].T+maps['readout_bias']).argmax(-1)
            np.testing.assert_allclose(np.mean(answers==labels),row['answer_accuracy'],atol=1e-12)
            np.testing.assert_allclose(prediction,saved[tag+'_'+word+'_composed'],atol=2e-5,rtol=1e-6)
            total+=1
        if split==3:
            error=x[:,0]@rc+bc-x[:,1]
            gain=np.square(error@ri).sum()/np.square(error).sum()
            np.testing.assert_allclose(gain,rec['C_error_effective_I_gain_squared'],atol=1e-10,rtol=1e-10)
    return {'source':rec['source'],'status':'passed','independently_replayed_endpoints':total,
        'exact_cross_terms_directly_calculated':True,'effective_error_gain_independently_recomputed':True,
        'scope':'CI/ICI base predictions and transported local-error energies, cross terms, answer scores and effective C-error gain. Other teacher-forced diagnostic transitions are saved but not independently replayed by this verifier.'}


def run():
    cp=ROOT/'transport_verification_cache.json'
    cache=read(cp) if cp.exists() else {}
    code=digest(__file__)+digest('experiments/until_10_verify.py')
    results,failures=[],[]
    for path in Path('results/relation_error_transport_confirmation/evaluations').glob('*.json'):
        fingerprint=code+digest(path)
        key=str(path)
        if key in cache and cache[key]['fingerprint']==fingerprint:
            results.append(cache[key]['result']);continue
        try:
            result=verify(path)
            cache[key]={'fingerprint':fingerprint,'result':result}
            results.append(result);save(cp,cache)
        except Exception as error:
            failures.append({'check':key,'error':repr(error)})
    result={'updated_utc':datetime.now(timezone.utc).isoformat(),'status':'passed_available_completed_records' if not failures else 'failed',
        'checks_completed':len(results),'independently_replayed_endpoints':sum(r['independently_replayed_endpoints'] for r in results),
        'checks':results,'failures':failures,'verifier_code_sha256':digest(__file__)}
    save(ROOT/'transport_verification.json',result)
    print({'transport_verification':result['status'],'sources':len(results),'failures':failures},flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        run()
        if not args.watch or time.time()>=DEADLINE:break
        time.sleep(min(45,max(0,DEADLINE-time.time())))
