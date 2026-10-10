"""Renders one Markdown file per eval category plus a short combined
index -- 4 genuinely separate reportable outputs, not just separate
code paths sharing one interleaved document.
"""
from __future__ import annotations


def _pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "n/a"


def _score(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


def _metric_table(aggregate: dict) -> list[str]:
    lines = ["| Metric | Cases | Exercised | Mean score | Pass rate |", "|---|---:|---:|---:|---:|"]
    for name, info in aggregate.items():
        lines.append(f"| {name} | {info['cases']} | {info['exercised']} | {_score(info['mean_score'])} | {_pct(info['pass_rate'])} |")
    return lines


def _gates_table(gates: list[dict]) -> list[str]:
    if not gates:
        return ["No quality gates configured for this category."]
    lines = ["| Gate | Threshold | Actual | Result |", "|---|---|---|---|"]
    for g in gates:
        actual = _score(g["actual"]) if isinstance(g["actual"], float) else (g["actual"] if g["actual"] is not None else "n/a")
        lines.append(f"| {g['name']} | {g['comparison']} {g['threshold']} | {actual} | {'PASS' if g['passed'] else 'FAIL'} |")
    return lines


def _not_exercised_list(cases: list[dict]) -> list[str]:
    return [f"- `{c['id']}` ({c['category']}): {c['reason']}" for c in cases if not c["exercised"]]


def render_retriever_report(report: dict) -> str:
    cat = report.get("retriever")
    lines = ["# Retriever alone", "", "Do the retrieved policy chunks fit the question and the golden answer? "
              "Real `RetrievalEngine.retrieve()` output, scored against an independently-built golden answer.", ""]
    if not cat:
        return "\n".join(lines + ["Not run this invocation."])
    lines += [f"Judge model: `{cat['judge_model']}`  ", f"{cat['exercised_cases']}/{cat['total_cases']} cases exercised.", ""]
    lines += ["## DeepEval metrics (Contextual Precision / Recall / Relevancy)", ""] + _metric_table(cat["aggregate"]) + [""]
    lines += ["## Not exercised", ""] + _not_exercised_list(cat["cases"])
    return "\n".join(lines) + "\n"


def render_generator_report(report: dict) -> str:
    cat = report.get("generator")
    lines = ["# Generator alone", "", "Given known-good (golden) context -- retrieval bypassed entirely -- "
              "does the real generation hook produce a faithful, relevant, correctly-cited answer?", ""]
    if not cat:
        return "\n".join(lines + ["Not run this invocation."])
    lines += [f"Judge model: `{cat['judge_model']}`  ", f"Mode: `{cat['mode']}`  ",
              f"{cat['exercised_cases']}/{cat['total_cases']} cases exercised.", ""]
    lines += ["## DeepEval metrics (Faithfulness, Answer Relevancy, Citation Faithfulness)", ""] + _metric_table(cat["aggregate"]) + [""]
    lines += ["## Not exercised", ""] + _not_exercised_list(cat["cases"])
    return "\n".join(lines) + "\n"


def render_pipeline_report(report: dict) -> str:
    cat = report.get("pipeline")
    lines = ["# Full RAG pipeline", "", "The full question flow -- contextualization, retrieval, reranking, "
              "generation -- scored for the RAG triad (Faithfulness, Answer Relevancy, Contextual Relevancy) "
              "plus a GEval Groundedness check against independently-loaded gold context. Must-abstain cases "
              "(out_of_corpus, medical_advice, prompt-injection) are included.", ""]
    if not cat:
        return "\n".join(lines + ["Not run this invocation."])
    lines += [f"Judge model: `{cat['judge_model']}`  ", f"{cat['exercised_cases']}/{cat['total_cases']} cases exercised.", ""]
    lines += ["## DeepEval metrics", ""] + _metric_table(cat["aggregate"]) + [""]
    return "\n".join(lines) + "\n"


def render_application_report(report: dict) -> str:
    cat = report.get("application")
    lines = ["# Application answer quality", "", "Is the final, user-facing answer correct, complete, and safe? "
              "GEval Correctness/Completeness, Citation Faithfulness, comparison-arithmetic ground truth, "
              "a safety bundle (toxicity, PII/data leakage, prompt-injection resistance, role adherence, "
              "non-advice), plus deterministic content checks and readability.", ""]
    if not cat:
        return "\n".join(lines + ["Not run this invocation."])
    lines += [f"Judge model: `{cat['judge_model']}`  ", f"{cat['exercised_cases']}/{cat['total_cases']} cases exercised.", ""]
    lines += ["## Correctness / Completeness / Citation", ""] + _metric_table(cat["aggregate"]) + [""]

    summary = cat["content_checks_summary"]
    lines += ["## Content checks (must_include / must_not_include)", "",
              f"{summary['cases_checked']} case(s) declared phrases; {summary['cases_failed']} failed.", ""]

    aq = cat["answer_quality_aggregate"]
    lines += ["## Readability (answer_quality.py)", "",
              f"Duplicate-source rate: {_pct(aq['duplicate_source_rate'])}; "
              f"redundant-segment rate: {_pct(aq['redundant_segment_rate'])}; "
              f"jargon rate: {_pct(aq['jargon_rate'])} ({aq['cases_with_jargon']}/{aq['cases']} cases). "
              f"Templates with jargon: {len(cat['template_jargon'])}/{cat['template_samples_checked']}.", ""]

    op = cat["operational"]
    lines += ["## Operational (latency / tokens)", "",
              f"{op['cases_with_usage']} case(s) made a real app-generation call. "
              f"Mean latency: {op['mean_latency_seconds']:.2f}s" if op['mean_latency_seconds'] is not None
              else f"{op['cases_with_usage']} case(s) made a real app-generation call (no latency recorded).", ""]
    if op.get("total_tokens"):
        lines.append(f"Total tokens: {op['total_tokens']} (prompt: {op['total_prompt_tokens']}, completion: {op['total_completion_tokens']}). {op['note']}.")
    lines.append("")
    return "\n".join(lines) + "\n"


_RENDERERS = {
    "retriever": render_retriever_report,
    "generator": render_generator_report,
    "pipeline": render_pipeline_report,
    "application": render_application_report,
}


def render_category_report(report: dict, category: str) -> str:
    return _RENDERERS[category](report)


def render_index(report: dict) -> str:
    lines = [
        "# Evaluation report", "",
        f"Generated: {report['timestamp']}  ", f"Mode: **{report['mode']}**  ",
        f"Judge model: `{report['judge_model']}`  ",
        f"Categories run: {', '.join(report['categories_run'])}", "",
        "## Corpus", "",
        f"{report['corpus']['product_count']} products, {report['corpus']['evidence_count']} evidence records.", "",
        "## Categories", "",
    ]
    titles = {
        "retriever": "Retriever alone", "generator": "Generator alone", "pipeline": "Full RAG pipeline",
        "application": "Application answer quality",
    }
    for key in ("retriever", "generator", "pipeline", "application"):
        if key in report:
            lines.append(f"- [{titles[key]}]({key}.md)")
    lines += ["", "## Quality gates", ""]
    lines += _gates_table(report["quality_gates"])
    lines += ["", f"**Overall: {'PASS' if report['quality_gates_all_passed'] else 'FAIL'}**", ""]
    return "\n".join(lines) + "\n"
