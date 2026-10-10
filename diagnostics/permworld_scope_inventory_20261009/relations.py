"""Exact relationship certificates and within-cohort single-task coverage."""
from itertools import permutations
import json
import numpy as np
import tomllib
from inventory import ROOT,WS,NR,js,save,sha,csv_read,csv_write
from neurips_permutations.math_ops import PROPERTY_FUNCTIONS as F
from experiments.permutation_audit import transform


def relations():
    old=tomllib.loads((NR/'configs/property_task_geometry.toml').read_text())['relation_pairs']
    family={'descent_inverse':'descent_recoil','peak_complement':'interior','fixed_complement':'fixed',
        'exceedance_inverse':'positions','subsequence_complement':'subsequence','run_complement':'run',
        'record_complement':'records','decomposition_complement':'decomposition'}
    out=[dict(id=r['pair_id'],left=r['left'],right=r['right'],correct=r['input_transform'],
        wrong='complement' if r['input_transform']=='inverse' else 'inverse',offset=r['right_label_offset'],
        kind='cross_task',family=family[r['pair_id']],origin='original8') for r in old]
    for tag,l,r,fam in [('subsequence','lis_length','lds_length','subsequence'),
            ('run','longest_increasing_run','longest_decreasing_run','run'),
            ('fixed','fixed_points','anti_fixed_points','fixed'),
            ('max_record','left_to_right_maxima','right_to_left_maxima','records'),
            ('min_record','left_to_right_minima','right_to_left_minima','records'),
            ('double','double_ascents','double_descents','interior')]:
        out.append(dict(id=tag+'_reverse',left=l,right=r,correct='reverse',wrong='inverse',offset=0,
                        kind='cross_task',family=fam,origin='expanded'))
    out += [dict(id='max_min_record_inverse',left='left_to_right_maxima',right='right_to_left_minima',
        correct='inverse',wrong='complement',offset=0,kind='cross_task',family='records',origin='expanded'),
        dict(id='double_complement',left='double_ascents',right='double_descents',correct='complement',
        wrong='inverse',offset=0,kind='cross_task',family='interior',origin='expanded')]
    for task,fam in [('lis_length','subsequence'),('lds_length','subsequence'),('fixed_points','fixed')]:
        out.append(dict(id=task+'_inverse_self',left=task,right=task,correct='inverse',wrong='reverse',
            offset=0,kind='self_invariance',family=fam,origin='expanded'))
    return out


def run():
    rs=relations();stats={r['id']:{'checks':0,'wrong_counterexamples':0,'degenerate_equal_labels':0} for r in rs}
    rng=np.random.default_rng(2026100940)
    examples=(p for n in range(1,8) for p in permutations(range(1,n+1)))
    values=list(examples)+[tuple(map(int,rng.permutation(n)+1)) for n in range(10,31) for _ in range(128)]
    for p in values:
        for r in rs:
            y=int(F[r['left']](p));z=int(F[r['right']](transform(p,r['correct'])))
            assert y+r['offset']==z,(r,p,y,z)
            wrong=int(F[r['right']](transform(p,r['wrong'])))
            st=stats[r['id']];st['checks']+=1;st['wrong_counterexamples']+=wrong!=y+r['offset'];st['degenerate_equal_labels']+=wrong==y+r['offset']
    assert all(s['wrong_counterexamples']>0 for s in stats.values())
    inventory=csv_read(ROOT/'training_inventory.csv')
    groups={}
    for row in inventory:
        if row['record_status'] not in ['complete','completed'] or row['checkpoint_exists']!='True' or row['single_multi']!='single':continue
        groups.setdefault(row['cohort'],{}).setdefault(row['tasks'],{})[str(row['seed'])]=row
    graph=[];eval_pairs=[]
    for r in rs:
        matches={}
        for cohort,tasks in groups.items():
            seeds=sorted(set(tasks.get(r['left'],{}))&set(tasks.get(r['right'],{})),key=lambda x:int(x))
            if seeds:matches[cohort]=seeds
            for seed in seeds:
                if cohort not in ['specialist16','six_hour_session_single','native_confirmation_single']:continue
                a=tasks[r['left']][seed];b=tasks[r['right']][seed]
                assert a['data_sha256']==b['data_sha256']
                eval_pairs.append(dict(**r,cohort=cohort,seed=int(seed),left_id=a['logical_position'],right_id=b['logical_position'],
                    left_checkpoint=a['checkpoint'],right_checkpoint=b['checkpoint'],left_sha256=a['checkpoint_sha256'],right_sha256=b['checkpoint_sha256'],
                    data_sha256=a['data_sha256'],initialization_grade='C'))
        graph.append(dict(**r,formula=f'{r["right"]}({r["correct"]}(pi)) = {r["left"]}(pi) + {r["offset"]}',
            wrong_design='Original8 retain original wrong action; new R uses I because C also correctly swaps LIS/run/fixed/double. New I uses C. Frozen before neural metrics.',
            dependency_label=f'family:{r["family"]};shared_tasks_and_group_actions;fixed_domain',
            available_matched_single_cohorts=json.dumps(matches,sort_keys=True),
            gap='' if matches else 'No complete within-cohort single-task pair; cross-cohort models do not establish matched effect',
            exact_checks=stats[r['id']]['checks'],wrong_counterexamples=stats[r['id']]['wrong_counterexamples'],
            equal_answers_under_wrong=stats[r['id']]['degenerate_equal_labels'],
            identity_note='Same-model identity CKA=1 is a diagnostic ceiling, not an invariance null to beat' if r['kind']=='self_invariance' else ''))
    csv_write('relation_task_graph.csv',graph);save('relationship_certificates.json',{'permutations':len(values),'relations':len(rs),
        'verified':stats,'formula_source_hash':sha(NR/'src/neurips_permutations/math_ops.py'),'new_source_training':0})
    save('evaluation_pairs.json',eval_pairs)
    print({'relations':len(rs),'cross_task':sum(r['kind']=='cross_task' for r in rs),'matched_relation_source_pairs':len(eval_pairs)},flush=True)


if __name__=='__main__':run()
