"""Supplementary check of shared-score versus contrast state error."""
import json
from pathlib import Path
import numpy as np
from .relation_error_localization import row_projection
from .longrun_engine import atomic_json
from .two_step_relation_factorial import now
from .permworld_combinations import sha
ROOT=Path('results/readout_null_confirmation')


def run(domain='matrix'):
    root=ROOT/domain;assert (root/'test_opened.json').exists();rows=[];maximum=0.
    for c in ['full_correct','full_wrong','null_correct','null_wrong']:
        for i in range(3):
            file=root/'evaluations'/f'n{i}_{c}.npz';d=dict(np.load(file));h=d['native'].astype(float);w=d['readout_weight'].astype(float)
            e=h[:,0]@d['rho_a']+d['bias_a']-h[:,1]
            p=row_projection(w);pc=row_projection(w-w.mean(0));common=e@(p-pc);contrast=e@pc
            np.testing.assert_allclose(common@(w-w.mean(0)).T,0,atol=1e-9,rtol=0)
            maximum=max(maximum,float(abs(common@(w-w.mean(0)).T).max()))
            np.testing.assert_allclose(e@p,common+contrast,atol=1e-9,rtol=1e-9)
            rows.append({'condition':c,'source':i,'common_fraction_of_row_error':float(np.square(common).sum()/np.square(e@p).sum()),
                         'contrast_fraction_of_row_error':float(np.square(contrast).sum()/np.square(e@p).sum()),
                         'input_sha256':sha(file)})
    atomic_json(root/'common_score_shift_diagnostic.json',{'status':'complete','completed_utc':now(),'code_sha256':sha(__file__),
        'scope':'Posthoc after matrix test outcomes; not a new prospective hypothesis test. Shared-score shifts alone cannot explain predominantly contrast-direction row error.',
        'records':rows,'maximum_common_contrast_logit_change':maximum})

if __name__=='__main__':run()
