"""Source-seed summaries and exportable figures for the follow-up."""
from collections import defaultdict
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .until_10_report import ROOT, source_relation_rows, observed_rows, ordinary_rows, frozen_rows, confidence_rows, prefix_rows, proxy_rows
from .longrun_engine import atomic_json


T975 = {1:12.7062, 2:4.30265, 3:3.18245, 4:2.77645, 5:2.57058, 6:2.44691,
    7:2.36462, 8:2.30600, 9:2.26216, 10:2.22814, 11:2.20099}
COHORTS = ['原三种子', '新增六种子', '三个新世界']
LABELS = ['Original 3 seeds', 'New 6 seeds\nsame data world', 'New 3 worlds\nnew data + initialization']


def estimate(rows, key):
    values = [r[key] for r in rows]
    mean = statistics.mean(values)
    interval = None
    if len(values) > 1 and len(values)-1 in T975:
        half = T975[len(values)-1] * statistics.stdev(values) / math.sqrt(len(values))
        interval = [mean-half, mean+half]
    return {'source_repetitions': len(values), 'mean': mean, 'source_values': values,
        'descriptive_t_interval_95': interval, 'interval_assumption': 'Independent source repetitions, approximately normal source-level effects; exploratory, small n, no multiple-endpoint adjustment.'}


def points(ax, x, values, color, offset=0, marker='o', label=None):
    if not values:
        return
    jitter = np.linspace(-.045, .045, len(values)) if len(values) > 1 else [0]
    ax.scatter(x+offset+np.asarray(jitter), values, color=color, marker=marker, s=30, alpha=.85, label=label)
    ax.plot([x+offset-.07, x+offset+.07], [statistics.mean(values)]*2, color=color, lw=2.5)


def run():
    ROOT.mkdir(exist_ok=True)
    relation = [r for r in source_relation_rows() if r['condition'] == 'correct_relations' and r['method'] == 'native_operators']
    observed = [r for r in observed_rows() if r['condition'] == 'correct_relations' and r['method'] == 'native_operators']
    ordinary = ordinary_rows()
    confidence = confidence_rows()
    prefix = prefix_rows()
    proxy = proxy_rows()
    summary = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'source_native_hidden_words': [], 'observed_known_start': [],
        'ordinary_conditional_geometry': [], 'paired_ordinary_training_improvements': [],
        'paired_ordinary_group_contrasts': [], 'paired_frozen_relation_accuracy_improvements': [],
        'confidence_residual_geometry': [], 'prefix_geometry': [], 'output_proxy_geometry': [], 'fixed_general_scalar_ceiling': .5,
        'scope': 'Source seeds are repetitions; actions/words are correlated endpoints. Words are averaged within source before ordinary geometry summaries. Intervals are descriptive t intervals, not adjusted significance claims. Same-world new seeds and new data+initialization worlds are reported separately.'}
    for cohort in COHORTS:
        for word in ['ci', 'ici']:
            rows = sorted([r for r in relation if r['cohort'] == cohort and r['word'] == word], key=lambda r:r['seed'])
            if rows:
                summary['source_native_hidden_words'].append({'cohort': cohort, 'word': word, 'seeds': [r['seed'] for r in rows],
                    'above_50_count': sum(r['answer_accuracy'] > .5 for r in rows), 'visible_e_C_I': [[r[k] for k in ['e_accuracy','C_accuracy','I_accuracy']] for r in rows],
                    **estimate(rows, 'answer_accuracy')})
        for case in ['observed_C_then_I', 'observed_I_then_CI']:
            rows = sorted([r for r in observed if r['cohort'] == cohort and r['case'] == case], key=lambda r:r['seed'])
            if rows:
                summary['observed_known_start'].append({'cohort': cohort, 'case': case, 'seeds': [r['seed'] for r in rows],
                    'above_50_count': sum(r['answer_accuracy'] > .5 for r in rows), **estimate(rows, 'answer_accuracy')})
    buckets = defaultdict(list)
    for r in ordinary:
        buckets[(r['cohort'],r['group'],r['status'],r['view'],r['kind'])].append(r)
    for (cohort,group,status,view,kind), rows in sorted(buckets.items()):
        rows.sort(key=lambda r:r['seed'])
        summary['ordinary_conditional_geometry'].append({'cohort': cohort,'group': group,'status':status,'view':view,'kind':kind,
            'seeds':[r['seed'] for r in rows], 'pair_target_nmse':estimate(rows,'pair_target_nmse'),
            'pair_action_displacement_nmse':estimate(rows,'pair_action_displacement_nmse')})
    comparison_buckets = defaultdict(dict)
    for row in ordinary:
        comparison_buckets[(row['cohort'], row['group'], row['view'], row['kind'])][(row['seed'], row['status'])] = row
    for (cohort, group, view, kind), entries in sorted(comparison_buckets.items()):
        seeds = sorted({seed for seed, status in entries if (seed, 'trained') in entries and (seed, 'random') in entries})
        if seeds:
            differences = [{'seed': seed, 'improvement': entries[seed, 'random']['pair_target_nmse'] - entries[seed, 'trained']['pair_target_nmse']}
                for seed in seeds]
            summary['paired_ordinary_training_improvements'].append({'cohort': cohort, 'group': group, 'view': view, 'kind': kind,
                'seeds': seeds, 'metric': 'random_minus_trained_pair_target_NMSE_positive_favors_training', **estimate(differences, 'improvement')})
    contrast_buckets = defaultdict(dict)
    for row in ordinary:
        if row['status'] == 'trained':
            contrast_buckets[(row['cohort'], row['view'], row['kind'])][row['seed'], row['group']] = row
    for (cohort, view, kind), entries in sorted(contrast_buckets.items()):
        for control in ['novel_minima_pair', 'novel_peak_mixed']:
            seeds = sorted({seed for seed, group in entries if (seed, 'novel_records') in entries and (seed, control) in entries})
            if seeds:
                differences = [{'seed': seed, 'contrast': entries[seed, control]['pair_target_nmse'] - entries[seed, 'novel_records']['pair_target_nmse']}
                    for seed in seeds]
                summary['paired_ordinary_group_contrasts'].append({'cohort': cohort, 'view': view, 'kind': kind, 'control_group': control,
                    'seeds': seeds, 'metric': 'control_minus_records_pair_target_NMSE_positive_favors_records', **estimate(differences, 'contrast')})
    frozen_buckets = defaultdict(lambda: defaultdict(list))
    for row in frozen_rows():
        pairing = 'correct' if row['pairing'] == 'correct' else 'shuffled'
        frozen_buckets[row['cohort'], row['condition']][row['seed'], pairing].append(row['answer_accuracy'])
    for (cohort, condition), entries in sorted(frozen_buckets.items()):
        seeds = sorted({seed for seed, pairing in entries if (seed, 'correct') in entries and (seed, 'shuffled') in entries})
        if seeds:
            differences = [{'seed': seed, 'improvement': statistics.mean(entries[seed, 'correct']) - statistics.mean(entries[seed, 'shuffled'])}
                for seed in seeds]
            summary['paired_frozen_relation_accuracy_improvements'].append({'cohort': cohort, 'condition': condition, 'seeds': seeds,
                'metric': 'correct_minus_shuffled_accuracy_words_averaged_within_source', **estimate(differences, 'improvement')})
    atomic_json(ROOT/'seed_statistics.json',summary)
    for destination, entries, fields, metrics in [
        ('confidence_residual_geometry', confidence, ['group', 'status', 'kind'], ['pair_target_nmse', 'retained_pair_energy_fraction']),
        ('prefix_geometry', prefix, ['cohort', 'group', 'status', 'kind'], ['pair_target_nmse']),
        ('output_proxy_geometry', proxy, ['group', 'status', 'kind'], ['pair_target_nmse']),
    ]:
        groups_for_stats = defaultdict(list)
        for row in entries:
            groups_for_stats[tuple(row[f] for f in fields)].append(row)
        for values, rows in sorted(groups_for_stats.items()):
            rows.sort(key=lambda r: r['seed'])
            summary[destination].append({**dict(zip(fields, values)), 'seeds': [r['seed'] for r in rows],
                **{metric: estimate(rows, metric) for metric in metrics}})
    atomic_json(ROOT/'seed_statistics.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig, axes = plt.subplots(1,3,figsize=(13.4,4.6),layout='constrained')
    labels = ['Original seeds\nn=' + str(sum(r['cohort']==COHORTS[0] and r['word']=='ci' for r in relation)),
        'New seeds, same world\nn=' + str(sum(r['cohort']==COHORTS[1] and r['word']=='ci' for r in relation)) + '/6',
        'New data + init worlds\nn=' + str(sum(r['cohort']==COHORTS[2] and r['word']=='ci' for r in relation)) + '/3']
    for j,cohort in enumerate(COHORTS):
        for word,offset,color in [('ci',-.13,'#2463ad'),('ici',.13,'#dc7825')]:
            vals=[r['answer_accuracy'] for r in relation if r['cohort']==cohort and r['word']==word]
            points(axes[0],j,vals,color,offset,label=word.upper() if j==0 else None)
        for case,offset,color in [('observed_C_then_I',-.13,'#1b885b'),('observed_I_then_CI',.13,'#a04cad')]:
            vals=[r['answer_accuracy'] for r in observed if r['cohort']==cohort and r['case']==case]
            points(axes[1],j,vals,color,offset,label='True C start, then I' if case=='observed_C_then_I' and j==0 else ('True I start, then CI' if j==0 else None))
        grades=[statistics.mean([r[k] for k in ['e_accuracy','C_accuracy','I_accuracy']]) for r in relation if r['cohort']==cohort and r['word']=='ci']
        points(axes[2],j,grades,'#555e6c')
    for ax in axes:
        ax.set_xticks(range(3),labels,fontsize=9)
        ax.set_ylim(0,1.04)
        ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Accuracy')
    for ax in axes[:2]:
        ax.axhline(.5,color='#777',ls='--',lw=1,label='Scalar-answer ceiling (50%)')
        ax.legend(fontsize=8,loc='lower left')
    axes[0].set_title('Base hidden state: continuous composition')
    axes[1].set_title('Extra true known-state input (different scope)')
    axes[2].set_title('Visible e / C / I task accuracy (mean)')
    fig.suptitle('Explicit relation supervision: completed source models only',fontsize=13)
    fig.savefig(ROOT/'relation_accuracy.png',dpi=180)
    fig.savefig(ROOT/'relation_accuracy.pdf')
    plt.close(fig)
    groups=['novel_records','novel_minima_pair','novel_peak_mixed']
    group_labels=['Four directional record counts','Minima mixed tasks','Peak mixed tasks']
    fig,axes=plt.subplots(2,3,figsize=(13.3,7.4),layout='constrained')
    conditions=[('旧源模型／新联合匹配测试','random','Old init','#777'),('旧源模型／新联合匹配测试','trained','Old trained','#2463ad'),
        ('新增普通源种子','random','New init','#b8a498'),('新增普通源种子','trained','New trained','#1b885b')]
    for i,view in enumerate(['source_query_concat','source_query_numeric_null']):
        for j,group in enumerate(groups):
            ax=axes[i,j]
            tick_labels = []
            for x,(cohort,status,label,color) in enumerate(conditions):
                vals=[r['pair_target_nmse'] for r in ordinary if r['cohort']==cohort and r['status']==status and r['group']==group and r['view']==view and r['kind']=='未拟合复合']
                points(ax,x,vals,color)
                tick_labels.append(label + '\nn=' + str(len(vals)) + '/3')
            ax.set_xticks(range(4),tick_labels,rotation=20,fontsize=9)
            ax.axhline(1,color='#777',ls='--',lw=1)
            ax.set_ylim(.65,1.6)
            ax.grid(axis='y',alpha=.2)
            ax.set_ylabel('Pair-target NMSE (zero prediction = 1)')
            ax.set_title(group_labels[j]+('\nRaw query, frozen 64D maps' if i==0 else '\nNumeric readout null space, frozen 64D maps'))
    fig.suptitle('Ordinary training: answer-orbit-matched test, compositions not fitted',fontsize=13)
    fig.savefig(ROOT/'ordinary_geometry.png',dpi=180)
    fig.savefig(ROOT/'ordinary_geometry.pdf')
    plt.close(fig)
    if confidence:
        fig, axes = plt.subplots(2, 3, figsize=(13.3, 7.5), layout='constrained')
        for j, group in enumerate(groups):
            ax = axes[0, j]
            configurations = [
                ('random', 'Null: init', '#777', False), ('trained', 'Null: trained', '#2463ad', False),
                ('random', 'Residual: init', '#b8a498', True), ('trained', 'Residual: trained', '#c66c35', True),
            ]
            labels_for_axes = []
            for x, (status, label, color, residual) in enumerate(configurations):
                if residual:
                    vals = [r['pair_target_nmse'] for r in confidence if r['group'] == group and r['status'] == status and r['kind'] == '未拟合复合']
                else:
                    vals = [r['pair_target_nmse'] for r in ordinary if r['cohort'] == '新增普通源种子' and r['group'] == group and r['status'] == status
                        and r['view'] == 'source_query_numeric_null' and r['kind'] == '未拟合复合']
                points(ax, x, vals, color)
                labels_for_axes.append(label + '\nn=' + str(len(vals)) + '/3')
            ax.set_xticks(range(4), labels_for_axes, rotation=15, fontsize=8)
            ax.axhline(1, color='#777', ls='--', lw=1)
            ax.set_ylim(.7, 1.4)
            ax.set_ylabel('Pair-target NMSE (zero prediction = 1)')
            ax.set_title(group_labels[j])
            axes[1, j].set_title('Residual pair energy fraction')
            for x, status, color in [(0, 'random', '#777'), (1, 'trained', '#c66c35')]:
                vals = [r['retained_pair_energy_fraction'] for r in confidence if r['group'] == group and r['status'] == status and r['kind'] == '未拟合复合']
                points(axes[1, j], x, vals, color)
            axes[1, j].set_xticks([0, 1], ['Matched init', 'Trained'])
            axes[1, j].set_ylim(0, .45)
            axes[1, j].set_ylabel('Residual / original null-space pair energy')
            for selected_ax in [ax, axes[1, j]]:
                selected_ax.grid(axis='y', alpha=.2)
        fig.suptitle('Exploratory confidence-association control: not a causal intervention', fontsize=13)
        fig.savefig(ROOT/'confidence_geometry.png', dpi=180)
        fig.savefig(ROOT/'confidence_geometry.pdf')
        plt.close(fig)
    if proxy:
        fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.9), layout='constrained')
        configurations = [
            ('random', 'Original: init', '#777', False), ('trained', 'Original: trained', '#2463ad', False),
            ('random', 'Proxy: init', '#b8a498', True), ('trained', 'Proxy: trained', '#1b885b', True),
        ]
        for j, group in enumerate(groups):
            labels_for_axes = []
            for x, (status, label, color, is_proxy) in enumerate(configurations):
                if is_proxy:
                    vals = [r['pair_target_nmse'] for r in proxy if r['group'] == group and r['status'] == status and r['kind'] == '未拟合复合']
                else:
                    vals = [r['pair_target_nmse'] for r in ordinary if r['cohort'] == '新增普通源种子' and r['group'] == group and r['status'] == status
                        and r['view'] == 'source_query_numeric_null' and r['kind'] == '未拟合复合']
                points(axes[j], x, vals, color)
                labels_for_axes.append(label + '\nn=' + str(len(vals)) + '/3')
            axes[j].set_xticks(range(4), labels_for_axes, rotation=15, fontsize=8)
            axes[j].axhline(1, color='#777', ls='--', lw=1)
            axes[j].set_ylim(.7, 1.2)
            axes[j].set_ylabel('Original null-space pair-target NMSE')
            axes[j].set_title(group_labels[j])
            axes[j].grid(axis='y', alpha=.2)
        fig.suptitle('Output-only at prediction: decoder fitted to calibration hidden vectors', fontsize=13)
        fig.supxlabel('Same original target and denominator; proxy adds a learned reconstruction fit', fontsize=10)
        fig.savefig(ROOT/'confidence_proxy_geometry.png', dpi=180)
        fig.savefig(ROOT/'confidence_proxy_geometry.pdf')
        plt.close(fig)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    end=datetime(2026,10,6,17,tzinfo=timezone.utc).timestamp()
    previous=None
    while True:
        folders=[Path('results/relation_seed_extension/evaluations'),Path('results/relation_observed_start/evaluations'),
            Path('results/ordinary_relation_seed_confirmation/evaluations'),Path('results/ordinary_initialization_control/evaluations'),
            Path('results/confidence_residual_geometry/evaluations'),Path('results/ordinary_prefix_confirmation/evaluations'),
            Path('results/confidence_proxy_geometry/evaluations'),
            *Path('results/relation_world_confirmation').glob('world*/evaluations'),
            *Path('results/relation_world_confirmation').glob('world*/observed_start')]
        count=tuple(len(list(p.glob('*.json'))) for p in folders)
        if count!=previous or time.time()>=end:
            run();previous=count
        if not args.watch or time.time()>=end:
            break
        time.sleep(min(45,max(0,end-time.time())))
