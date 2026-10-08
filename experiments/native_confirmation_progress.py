"""Read-only live source progress, separate from frozen scientific analysis."""
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import time

import numpy as np


def refresh(root):
    state=json.loads((root/'state.json').read_text())
    source=[]
    for phase in ('multi','single'):
        source.extend({**json.loads(p.read_text()),'phase':phase} for p in (root/phase).glob('*.json'))
    completed=[r for r in source if r['status']=='complete']
    target=[json.loads(p.read_text()) for p in (root/'transfer').glob('*.json')]
    current=json.loads((root/'current_job.json').read_text()) if (root/'current_job.json').exists() else None
    stamp=datetime.now(timezone.utc).isoformat()
    rows=''.join(f"<tr><td>{escape(r['job']['id'])}</td><td>{r['job']['seed']}</td><td>{escape(r['phase'])}</td><td>{100*np.mean([v['accuracy'] for v in r['source_audit']]):.2f}%</td></tr>" for r in sorted(completed,key=lambda r:(r['job']['seed'],r['job']['id'])))
    active=f"当前源模型 {escape(current['job_id'])}：{current['step']}/{current['planned_steps']} 更新，最近源验证均值 {100*current['validation_macro']:.2f}%。" if state['status']=='source_training' and current else ''
    result_link='<p><a href="report.html">最终分析报告</a></p>' if (root/'report.html').exists() else ''
    (root/'progress.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>固定候选复核进度</title><style>body{{max-width:1000px;margin:40px auto;padding:0 20px;font:16px/1.7 system-ui}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}aside{{background:#f2f4f7;padding:16px}}</style><h1>新源种子与完整单任务对照：执行进度</h1><p>更新：{stamp}。阶段 {escape(state['status'])}；完整源模型 {len(completed)}/46，完整目标模型 {sum(r['status']=='complete' for r in target)}/51。</p><p>{active}</p><aside>候选与四个单任务均使用五个新种子；其余七组合使用前三个新种子。迁移测试须等待全部源模型和冻结预测完成。这里列出的源审计成绩不代表迁移收益，完整效应区间将在全部预先固定条件完成后统一计算。</aside><table><tr><th>源任务组</th><th>种子</th><th>类型</th><th>未见输入源审计均值</th></tr>{rows}</table><p><a href="protocol.json">固定方案</a> · <a href="jobs.json">全部源任务队列</a> · <a href="weights.json">冻结预测权重</a> · <a href="dataset/metadata.json">全新输入范围</a> · <a href="state.json">执行状态</a></p>{result_link}</html>''')
    return state['status']


def run():
    root=Path('results/native_confirmation')
    metadata=json.loads((root/'run_process.json').read_text())
    pids=[r['pid'] for r in metadata['owned_run_log_processes']]
    logfile=str((root/'run.log').resolve())
    while True:
        status=refresh(root)
        alive=False
        for pid in pids:
            try:alive|=os.readlink(f'/proc/{pid}/fd/1')==logfile
            except OSError:pass
        if status=='complete' or not alive:break
        time.sleep(45)
    (root/'progress_monitor_state.json').write_text(json.dumps({'status':'stopped','last_worker_state':status,'updated_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')


if __name__=='__main__':run()
