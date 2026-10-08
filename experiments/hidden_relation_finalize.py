"""Add explicitly post-evaluation uncertainty notes without changing frozen gates."""
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import statistics


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "results/algebra_hidden_relations"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src") and value and not value.startswith(("https:", "http:", "#")):
                self.links.append(value.split("#")[0])


def run():
    summary = json.loads((ROOT / "summary.json").read_text())
    verification = json.loads((ROOT / "verification.json").read_text())
    assert verification["status"] == "passed"
    seeds = summary["source_seeds"]
    assert len(seeds) == 3
    paired_words = ("ci", "ici")
    rows = summary["correct_condition_per_seed"]
    per_seed = {
        str(seed): statistics.mean(r["answer_accuracy"] for r in rows if r["seed"] == seed)
        for seed in seeds
    }
    mean = statistics.mean(per_seed.values())
    se = statistics.stdev(per_seed.values()) / math.sqrt(3)
    # Exact two-sided 95% Student-t critical value for two degrees of freedom.
    t_critical = math.sqrt(2 * 0.95**2 / (1 - 0.95**2))
    interval = [mean - t_critical * se, mean + t_critical * se]
    interpretation_path = ROOT / "interpretation.json"
    original_report_sha = (
        json.loads(interpretation_path.read_text())["original_frozen_analysis_report_sha256"]
        if interpretation_path.exists() else digest(ROOT / "report.html")
    )
    supplementary = {
        "reported_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-evaluation interpretation; unchanged models, maps, metrics and registered descriptive mean gates. CI and ICI are correlated endpoints, averaged within each source seed before uncertainty calculation.",
        "frozen_summary_sha256": digest(ROOT / "summary.json"),
        "original_frozen_analysis_report_sha256": original_report_sha,
        "code_sha256": digest(__file__),
        "correct_native_per_seed_mean_CI_ICI_accuracy": per_seed,
        "correct_native_mean_CI_ICI_accuracy": mean,
        "exploratory_source_seed_t95_interval": interval,
        "interval_scope": "Classical normal-mean Student-t interval from only three source initializations, conditional on the shared training world and test set; does not include new-world variability. Not a pre-registered significance criterion.",
        "all_three_source_seeds_above_answer_ceiling": all(value > .5 for value in per_seed.values()),
        "interval_lower_bound_above_answer_ceiling": interval[0] > .5,
        "registered_four_mean_gate_values": summary["fixed_predictions"],
        "registered_mean_gates_are_statistical_significance_tests": False,
        "stable_beyond_answer_ceiling_confirmed": False,
        "ordinary_training_spontaneous_rule_discovery_confirmed": False,
        "interpretation": "正确关系监督的组合预测在三个种子中都优于错误配对及普通模型的事后生成元探针，并使用了超出三个已知标量答案的信息；平均准确率仅略高于50%，一个种子低于50%，尚未确认稳定的超上限推断。复合规则由框架提供，不是普通训练自行发现规则的证据。",
    }
    write_json(interpretation_path, supplementary)

    report_path = ROOT / "report.html"
    report = report_path.read_text()
    note = (
        '<p class="note">答案＋任务提示基线能近乎精确重现旧四方向记录组的代数几何。'
        '新实验中，正确关系监督的组合预测明显优于错误配对和普通训练的事后探针，'
        '但稳定超过已知答案的50%上限尚未确认。预设的4/4项均值门槛成立，'
        '这些门槛是描述性比较，不能当作显著性检验或三种子全部超过上限。</p>'
    )
    report = re.sub(r'<p class="note">.*?</p>', note, report, count=1)
    start, end = '<section id="post-evaluation-interpretation">', '</section><!-- interpretation end -->'
    details = (
        start + '<h2>种子差异与解释边界（评估后补充）</h2>'
        '<p>CI与ICI预测同一个缺失的RL-max统计，不能把它们算作两个独立重复。'
        f'先在每个种子内取两者平均，再跨种子平均，准确率为{mean:.2%}。'
        f'三个种子分别为{per_seed[str(seeds[0])]:.2%}、{per_seed[str(seeds[1])]:.2%}、{per_seed[str(seeds[2])]:.2%}；'
        '其中一个低于50%。'
        f'仅基于这三个源初始化的探索性95% t区间为{interval[0]:.2%}–{interval[1]:.2%}，跨越50%。'
        '该区间依赖正态均值假设，并条件于相同训练世界和测试集；不覆盖新世界的不确定性。</p>'
        '<p>三种子中正确关系监督都优于错误配对和普通模型事后探针，且每对同时答对的比例为正，'
        '支持可组合访问超出三个已知标量答案的信息。但平均高于50%约0.7个百分点，'
        '不能据此声称稳定突破答案上限。移除加性线性答案编码也不能排除全部非线性答案信息。</p>'
        '<p>这是显式已知关系监督与框架提供的组合规则下的局部结果。'
        '普通任务训练后事后拟合生成元的隐藏答案准确率约25%，当前没有支持其自行发现隐藏关系。'
        '<a href="interpretation.json">补充统计与解释口径</a>；原始预设门槛、汇总和核验保持原样。</p>' + end
    )
    if start in report:
        report = re.sub(re.escape(start) + r'.*?' + re.escape(end), lambda _: details, report, count=1)
    else:
        report = report.replace('<h2>旧结果的答案基线</h2>', details + '<h2>旧结果的答案基线</h2>', 1)
    report = report.replace(
        '<h2>新训练与关系留出</h2>',
        '<h2>新训练与关系留出</h2><p>LR-max是从左向右扫描出现新最大值的次数；'
        'C为值补操作，将排列元素v替换为n+1−v；I为排列取逆。'
        'CI表示先C后I，ICI表示先I、再C、再I。所有输入与标签留出按完整八状态轨道核验。</p>',
        1,
    ) if 'C为值补操作' not in report else report
    report = report.replace('是预测器的能力上界参考，不能代替组合推断', '用于检查模型直接处理新排列时的任务能力；它不是严格的数值上界，也不能代替组合推断')
    report_path.write_text(report)

    link_count = 0
    for path in [report_path, ROOT / "answer_controls/report.html"]:
        parser = LocalLinks()
        parser.feed(path.read_text())
        for link in parser.links:
            assert (path.parent / link).is_file(), (path, link)
            link_count += 1
    old_root = WORKSPACE / "results/algebra_structure_replication"
    old_completion = json.loads((old_root / "completion.json").read_text())
    for name, expected in old_completion["artifact_sha256"].items():
        assert digest(old_root / name) == expected, name
    for name, expected in old_completion["final_analysis_and_verifier_code_sha256"].items():
        assert digest(WORKSPACE / "experiments" / name) == expected, name
    assert "93 passed" in (ROOT / "tests.log").read_text()
    old_lis = json.loads((WORKSPACE / "results/native_ablation/state.json").read_text())
    assert old_lis["status"] == "paused_for_research_focus_change"
    names = [
        "protocol.json", "claim_gate.json", "pre_evaluation_code_manifest.json",
        "evaluation_protocol.json", "effective_config.json", "data_hashes.json",
        "dataset_audit.json", "data_verification.json", "summary.json",
        "verification.json", "interpretation.json", "report.html", "tests.log",
        "hidden_answers.png", "hidden_answers.pdf", "endpoints.csv", "group_summary.csv",
        "answer_controls/protocol.json", "answer_controls/summary.json", "answer_controls/report.html",
    ]
    write_json(ROOT / "completion.json", {
        "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "new_source_models": 9, "source_seeds": seeds,
        "source_label_exposures_per_model": 1920000,
        "tests_passed": 93, "independent_numeric_verification": "passed",
        "local_report_links_checked": link_count,
        "previous_replication_artifacts_preserved": True,
        "LIS_branch_still_paused": True,
        "registered_mean_gates": summary["fixed_predictions"],
        "stable_beyond_answer_ceiling_confirmed": False,
        "artifact_sha256": {name: digest(ROOT / name) for name in names},
        "finalizer_code_sha256": digest(__file__),
    })
    write_json(ROOT / "state.json", {"status": "complete", "conditions": 9, "verification": "passed", "updated_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "complete", "accuracy": mean, "seed_t95": interval, "links_checked": link_count}))


if __name__ == "__main__":
    run()
