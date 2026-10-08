"""A local, periodically refreshed research-progress artifact."""
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import time

from .inverse_functional_report import table
from .longrun_engine import atomic_json

ROOT=Path('results/overnight_inverse_functional');SUPP=Path('results/overnight_inverse_residual_geometry')


def read(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def update():
    plan=read('configs/overnight_inverse_functional.json');state=read(ROOT/'campaign_state.json');base_state=read(ROOT/'controller_state.json')
    active=SUPP if 'residual'in state.get('stage','')else ROOT;job=read(active/'current_job.json')
    complete=sum(read(p).get('status')=='complete'for p in(ROOT/'training').glob('n*_*.json'))
    stamp=datetime.now(timezone.utc).isoformat();live={'updated_utc':stamp,'observer_pid':os.getpid(),'deadline_utc':plan['deadline_utc'],
        'main_models_complete':complete,'base_state':base_state,'campaign_state':state,'active_job':job}
    atomic_json(ROOT/'live_state.json',live)
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>取逆实验夜间进度</title><style>body{font-family:system-ui;max-width:1100px;margin:30px auto;padding:20px;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}</style><body><h1>取逆实验夜间进度</h1><p>截止：2026年10月7日上午10点（洛杉矶）。更新时间UTC：'+escape(stamp)+'</p>'
    html+='<p>主实验完成：'+str(complete)+'/36；当前阶段：'+escape(state.get('stage',base_state.get('stage','准备')))+'。</p>'
    if job:html+='<p>当前训练：'+escape(job.get('replicate',{}).get('id',''))+' / '+escape(job.get('condition',''))+'；已完成 '+str(job.get('step',0))+' 更新。</p>'
    confirm=read(ROOT/'confirmation/diagnostic/summary.json')
    if confirm:
        rows=[r for r in confirm['means']if r['prior']=='teacher_length_mode']
        html+='<h2>旧模型的新留出集确认</h2>'+table(['方法','整体准确率','非众数准确率'],[[r['condition'],f"{100*r['accuracy']:.2f}%",f"{100*r['nonmodal_accuracy']:.2f}%"]for r in rows])
    html+='<p>新实验的测试结果在相同训练预算的全部条件拟合完毕后打开。这里的训练验证成绩用于监测，不用于测试选择。</p>'
    links=[('confirmation/diagnostic/report.html','旧模型新集确认'),('existing_diagnostic/report.html','既有预测诊断'),('protocol.json','主实验规则'),('../overnight_inverse_residual_geometry/protocol.json','同答案内部对齐规则'),('report.html','主实验结果'),('../overnight_inverse_residual_geometry/report.html','补充实验结果')]
    html+='<p>'+' · '.join('<a href="'+href+'">'+label+'</a>'for href,label in links if(ROOT/href).exists())+'</p></body></html>'
    temporary=ROOT/'progress.html.tmp';temporary.write_text(html);temporary.replace(ROOT/'progress.html')


if __name__=='__main__':
    deadline=datetime.fromisoformat(read('configs/overnight_inverse_functional.json')['deadline_utc']).timestamp()
    while time.time()<deadline:update();time.sleep(min(30,max(.1,deadline-time.time())))
    update()
