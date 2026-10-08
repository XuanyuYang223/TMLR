"""Cross-study input exclusion and immutable scientific-code provenance."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .until_10_verify import ROOT, read, archive, digest, save


def inputs(path):
    data = archive(path)
    result = set()
    if 'permutations' in data:
        for raw, n in zip(data['permutations'], data['lengths']):
            raw = np.asarray(raw)
            rows = raw.reshape(-1, raw.shape[-1])
            result.update(tuple(map(int, p[:n])) for p in rows)
    else:
        for key, rows in data.items():
            if not key.endswith('_input'):
                continue
            lengths = data[key[:-6] + '_lengths']
            for values, n in zip(rows, lengths):
                for row in values.reshape(-1, values.shape[-1]):
                    result.add(tuple(map(int, row[4 + 2*np.arange(n)])))
    return result


def check_excluded(signature, key):
    seen = set()
    hashes = signature[key]
    for path, expected in hashes.items():
        assert digest(path) == expected, path
        seen.update(inputs(path))
    return seen


def run():
    checks, failures = [], []
    try:
        root = Path('results/relation_world_confirmation')
        signature = read(root/'protocol.json')['signature']
        seen = check_excluded(signature,'excluded_data_sha256')
        parent_train = Path('results/algebra_hidden_relations/training_orbit_audit.npz')
        assert digest(parent_train) == signature['parent_training_orbit_sha256']
        seen.update(inputs(parent_train))
        prior = len(seen)
        for world in sorted(root.glob('world*')):
            for tag in ['training_orbit_audit.npz','probe_dataset.npz']:
                fresh=inputs(world/tag)
                assert not (fresh & seen), (str(world/tag), len(fresh & seen))
                seen.update(fresh)
                checks.append({'study':str(world/tag),'distinct_inputs':len(fresh),'excluded_previous_inputs':len(seen)-len(fresh)})
        checks.append({'study':'new_world_cross_study_exclusion','prior_distinct_inputs':prior,'status':'passed'})
        root=Path('results/joint_answer_matched_geometry')
        signature=read(root/'protocol.json')['signature']
        seen=check_excluded(signature,'excluded_data_sha256')
        fresh=inputs(root/'dataset.npz')
        assert not (seen & fresh)
        checks.append({'study':'joint_answer_matching_cross_study_exclusion','distinct_inputs':len(fresh),'excluded_previous_inputs':len(seen),'status':'passed'})
    except Exception as error:
        failures.append({'check':'cross_study_input_exclusion','error':repr(error)})
    scientific_modules=['frozen_relation_followup','relation_seed_extension','relation_operator_stability','relation_observed_start',
        'paired_relation_geometry','answer_matched_geometry','relation_world_confirmation','joint_answer_matched_geometry',
        'ordinary_relation_seed_confirmation','relation_readout_diagnostics','task_block_geometry_control',
        'conditional_relation_specificity','ordinary_initialization_control','relation_frozen_seed_confirmation',
        'relation_error_transport_confirmation','known_route_readout_control','matched_prediction_conditioning','ordinary_prefix_confirmation',
        'confidence_residual_geometry','confidence_proxy_geometry']
    for name in scientific_modules:
        root=Path('results')/name
        try:
            signature=read(root/'protocol.json')['signature']
            assert digest(Path('experiments')/(name+'.py')) == signature['code_sha256'], name
            config=Path('configs')/(name+'.json')
            if 'config_sha256' in signature:
                assert digest(config)==signature['config_sha256'],str(config)
            for field in ['core_sha256','excluded_data_sha256','excluded_dataset_sha256','parent_data_sha256']:
                for path,expected in signature.get(field,{}).items():
                    actual = Path(path)
                    if not actual.exists() and field=='parent_data_sha256':
                        actual=Path('results/algebra_hidden_relations')/path
                    assert digest(actual)==expected,str(actual)
            if 'fit_helper_sha256' in signature:
                assert digest('experiments/frozen_relation_followup.py')==signature['fit_helper_sha256']
            if 'score_helper_sha256' in signature:
                assert digest('experiments/hidden_relation_evaluate.py')==signature['score_helper_sha256']
            if 'prediction_helper_sha256' in signature:
                assert digest('experiments/relation_observed_start.py')==signature['prediction_helper_sha256']
            if 'diagnostic_helper_sha256' in signature:
                assert digest('experiments/relation_composition_diagnostics.py')==signature['diagnostic_helper_sha256']
            if 'test_data_sha256' in signature:
                assert digest('results/joint_answer_matched_geometry/dataset.npz')==signature['test_data_sha256']
            if name in ['ordinary_prefix_confirmation', 'confidence_residual_geometry', 'confidence_proxy_geometry']:
                assert digest('results/algebra_structure_replication/probe_dataset.npz') == signature['calibration_data_sha256']
            checks.append({'study':name,'registered_scientific_code_and_config_unchanged':True,'status':'passed'})
        except Exception as error:
            failures.append({'check':name+'_scientific_provenance','error':repr(error)})
    paused=read('results/native_ablation/state.json')['status']
    if paused!='paused_for_research_focus_change':
        failures.append({'check':'LIS_branch_remains_paused','actual':paused})
    result={'updated_utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if not failures else 'failed',
        'checks_completed':len(checks),'checks':checks,'failures':failures,'old_LIS_branch_status':paused,
        'verifier_code_sha256':digest(__file__),
        'scope':'Cross-study disjointness uses manually decoded permutation values. Code/config hashes are compared to pre-evaluation protocols; result inspection never changes registered scientific implementations. Presentation and independent verifier files are intentionally editable and have separate final hashes.'}
    save(ROOT/'data_provenance_verification.json',result)
    print(result)
    return result


if __name__=='__main__':
    run()
