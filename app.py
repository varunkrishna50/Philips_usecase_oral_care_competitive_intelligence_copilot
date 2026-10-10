"""Step 3C.2: Streamlit entry point for the Oral Care Competitive
Intelligence Copilot.

    streamlit run app.py

Presentation only. All grounding, citation validation, deterministic
comparison, status-aware phrasing, and safety/scope logic lives in
src/rag/generation.py's answer_query() (reused unchanged) -- this file
never reimplements any of it, and never renders a second citation
format on top of the one answer_query() already produces
(result.answer_text / result.sources / result.citation_urls).
"""
from __future__ import annotations

import re

import streamlit as st

from src.app.engine import get_engine, get_llm_client
from src.rag.generation import AnswerResult, answer_query
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


def _render_assistant_message(result: AnswerResult, message_index: int) -> None:
    """Shared by the chat-history replay loop AND the just-answered
    question below it, so citation chips appear immediately on a new
    answer instead of only after the next rerun shows it from
    history -- same rendering either way."""
    st.markdown(_linkify_markers(result.answer_text, result.citation_urls))
    if result.llm_unavailable_reason:
        st.caption(result.llm_unavailable_reason)
    if result.sources:
        with st.container(horizontal=True):
            st.caption("Citations:")
            for entry in result.sources:
                if st.button(entry.marker, key=f"cite-{message_index}-{entry.marker}"):
                    st.session_state.active_message_index = message_index


def _render_sources_panel(result: AnswerResult | None) -> None:
    st.subheader("Sources", icon=":material/link:")
    if result is None:
        st.caption("Ask a question to see its sources here.")
        return
    if not result.sources:
        st.caption("This answer didn't cite any specific page content.")
        return
    for entry in result.sources:
        with st.container(border=True):
            st.markdown(f"**[{entry.marker}]** [{entry.product_name}]({entry.url})")
            if entry.section:
                st.caption(entry.section)
            st.markdown(f":blue-background[{entry.quote}]")


st.set_page_config(
    page_title="Oral Care Competitive Intelligence Copilot",
    page_icon="🪥",
    layout="wide",
)

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

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user", "content": str} | {"role": "assistant", "result": AnswerResult}]
if "active_message_index" not in st.session_state:
    st.session_state.active_message_index = None  # index into messages of the result shown in the Sources panel

chat_col, sources_col = st.columns([3, 2])

with chat_col:
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
            continue

        result: AnswerResult = message["result"]
        with st.chat_message("assistant"):
            _render_assistant_message(result, i)

    prompt = st.chat_input(
        "Ask a question about a Philips Sonicare or Oral-B electric toothbrush",
    )
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    result = answer_query(prompt, engine, llm_client=llm_client)
            except Exception as exc:  # noqa: BLE001 - never let an unexpected error crash the page
                st.error(
                    "Something went wrong while answering this question. Please try again or "
                    "rephrase it."
                )
                st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
            else:
                st.session_state.messages.append({"role": "assistant", "result": result})
                new_index = len(st.session_state.messages) - 1
                st.session_state.active_message_index = new_index
                _render_assistant_message(result, new_index)

with sources_col:
    active_result = None
    if st.session_state.active_message_index is not None:
        active_message = st.session_state.messages[st.session_state.active_message_index]
        if active_message["role"] == "assistant":
            active_result = active_message["result"]
    _render_sources_panel(active_result)

if st.session_state.messages:
    with st.expander("Debug / Retrieval details"):
        last_assistant = next(
            (m["result"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None,
        )
        if last_assistant is None:
            st.caption("No answered question yet.")
        else:
            result = last_assistant
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
