"""Stop the later-added scientific diagnostics at the same fixed deadline."""
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import time

from .longrun_engine import atomic_json


if __name__ == '__main__':
    deadline = datetime(2026, 10, 6, 17, tzinfo=timezone.utc).timestamp()
    root = Path('/home/yangx/ICML')
    modules = {'experiments.relation_error_transport_confirmation',
        'experiments.known_route_readout_control', 'experiments.matched_prediction_conditioning',
        'experiments.ordinary_prefix_confirmation', 'experiments.confidence_residual_geometry',
        'experiments.confidence_proxy_geometry'}
    while time.time() < deadline:
        time.sleep(min(45, max(0, deadline - time.time())))
    stopped = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = (path / 'cmdline').read_bytes().split(b'\0')
            if (path / 'cwd').resolve() != root:
                continue
            matching = [module for module in modules if module.encode() in args]
            if matching:
                os.kill(int(path.name), signal.SIGTERM)
                stopped.append({'pid': int(path.name), 'module': matching[0]})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    atomic_json(root / 'results/until_10_followup/deadline_stop_additional.json', {
        'deadline_utc': '2026-10-06T17:00:00+00:00',
        'experiment_analysis_stop_utc': datetime.now(timezone.utc).isoformat(),
        'stopped_analysis_processes': stopped,
        'scope': 'The original guard covers earlier scientific analyses. This guard covers later diagnostics. Source controllers retain their own exact-deadline alarm and partial checkpoints; verification and reporting may finalize saved records.'})
