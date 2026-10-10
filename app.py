"""Step 3C.2: Streamlit entry point for the Oral Care Competitive
Intelligence Copilot.

    streamlit run app.py

Presentation only. All grounding, citation validation, deterministic
comparison, status-aware phrasing, and safety/scope logic lives in
src/rag/generation.py's answer_query() (reused unchanged) -- this file
never reimplements any of it, and never renders a second citation
format on top of the one answer_query() already produces
(result.answer_text / result.sources_text).
"""
from __future__ import annotations

import re

import streamlit as st

from src.app.engine import get_engine, get_llm_client
from src.rag.generation import answer_query
from src.rag.llm_provider import LLMConfig

_MARKER_RE = re.compile(r"\[S(\d+)\]")


def _linkify_markers(text: str, citation_urls: dict[str, str]) -> str:
    """Turns the [S1]/[S2] markers answer_query() already placed in
    answer_text into markdown hyperlinks, using the marker->URL mapping
    citations.py already computed (AnswerResult.citation_urls) -- pure
    string substitution on already-validated data, not a second
    citation renderer. A marker with no known URL (shouldn't happen,
    but defensively) is left as plain text."""
    def _replace(match: re.Match) -> str:
        marker = f"S{match.group(1)}"
        url = citation_urls.get(marker)
        return f"[{match.group(0)}]({url})" if url else match.group(0)

    return _MARKER_RE.sub(_replace, text)

st.set_page_config(page_title="Oral Care Competitive Intelligence Copilot", page_icon="🪥")

st.title("Oral Care Competitive Intelligence Copilot")
st.caption(
    "Answers are grounded strictly in official Philips US (www.usa.philips.com) and "
    "Oral-B US (oralb.com) public product pages. This assistant never uses third-party "
    "sources and never provides medical or dental advice -- only published product claims."
)

config = LLMConfig.from_env()
if config.available:
    st.caption(f"LLM: {config.model} (configured)")
else:
    st.caption(
        "No LLM configured -- answering with the deterministic grounded fallback "
        "(status templates and the deterministic comparison engine). This is expected, "
        "not an error: the app works fully without an API key."
    )

engine = get_engine()
llm_client = get_llm_client()

query = st.text_input(
    "Ask a question about a Philips Sonicare or Oral-B electric toothbrush:",
    placeholder="e.g. What is the price of the Sonicare Prestige 9900?",
)
ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Thinking..."):
                result = answer_query(query, engine, llm_client=llm_client)
        except Exception as exc:  # noqa: BLE001 - never let an unexpected error crash the page
            st.error(
                "Something went wrong while answering this question. Please try again or "
                "rephrase it."
            )
            st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
        else:
            st.markdown(_linkify_markers(result.answer_text, result.citation_urls))
            if result.sources_text:
                with st.expander("Sources", expanded=False):
                    st.markdown(result.sources_text)
            if result.llm_unavailable_reason:
                st.caption(result.llm_unavailable_reason)

            with st.expander("Debug / Retrieval Details"):
                st.write(f"**used_llm:** {result.used_llm}")
                retrieval_result = result.retrieval_result
                if retrieval_result is not None:
                    st.write(f"**intent:** {retrieval_result.intent}")
                    st.write(f"**product match:** {retrieval_result.product_match.kind}")
                    st.write(f"**attributes detected:** {retrieval_result.attributes}")
                else:
                    st.write("Query was resolved before retrieval ran (a scope/safety refusal "
                             "or a clarification request).")
                valid_count = sum(1 for seg in result.segments if seg.valid)
                st.write(f"**segments:** {len(result.segments)} ({valid_count} valid)")
                for seg in result.segments:
                    status = "valid" if seg.valid else f"rejected: {'; '.join(seg.reasons)}"
                    st.write(f"- [{status}] {seg.text}")
