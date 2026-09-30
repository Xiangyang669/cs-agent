"""跑一次完整评测，产出 `ACCEPTANCE.md`。

把入口做成模块而不是一段随手敲的脚本，是因为报告必须**可复现**：
换个人、换台机器、隔一个月，跑同一条命令应当得到同一份报告。

    python -m csagent.evaluation.run_eval

模型响应从 `llm_cache.json` 重放（用 `python -m csagent.evaluation.llm_cache` 重建）。
因此本命令**不发任何外部请求**、不花钱、且结果确定。
"""

import sys
from pathlib import Path

from csagent import config
from csagent.evaluation.cases import build_cases
from csagent.evaluation.fixtures import build_eval_graph_factory
from csagent.evaluation.harness import run_matrix
from csagent.evaluation.llm_cache import CachedLlm
from csagent.evaluation.report import build_report

ACCEPTANCE_PATH = Path(__file__).resolve().parent.parent.parent / "ACCEPTANCE.md"


def collect_warnings(matrix) -> list[str]:
    """区分度自检：某一类上几路差值过小，说明它测不出东西。

    这是 spec §10 判据 2 的落点 —— 但 V1 的矩阵是「配置 × 场景」，
    每一格内部不再分路，所以自检只报「该格样本数是否足够」。
    """
    warnings: list[str] = []
    for (config_name, scenario), count in sorted(matrix.counts.items()):
        if count == 0:
            warnings.append(f"{config_name} × {scenario} 没有样本 —— 该格无法得出结论")
        elif count < 15:
            warnings.append(f"{config_name} × {scenario} 只有 {count} 条，分层统计不稳")
    return warnings


def main() -> int:
    cases = build_cases()
    reviewed = sum(1 for c in cases if c.reviewed)
    print(f"评测集 {len(cases)} 条（其中 {reviewed} 条经人工复核）")

    factory = build_eval_graph_factory(config.DATA_DIR / "business.db", client=CachedLlm())
    matrix = run_matrix(cases, factory)

    report = build_report(matrix, cases, collect_warnings(matrix))
    report = report.replace(
        "# cs-agent-v1 治理报告",
        "# cs-agent-v1 治理报告\n\n"
        f"> 评测集：{len(cases)} 条，其中 **{reviewed} 条经用户逐条人工复核**（2026-09-30）。\n"
        "> 模型响应：由 `llm_cache.json` 重放（真跑一次模型逐条落盘的原始响应）。\n"
        "> 复现：`python -m csagent.evaluation.run_eval`",
        1,
    )

    ACCEPTANCE_PATH.write_text(report, encoding="utf-8")
    print(f"报告写入 {ACCEPTANCE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
