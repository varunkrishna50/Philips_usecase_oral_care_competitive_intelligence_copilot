"""Readable form of an evidence record for display and for the LLM.

Most evidence is a verbatim sentence from the page. Philips specification
rows are the exception: Step 2 stores the verbatim *JSON of the row* as the
quote (``{"csItemRank":"1","csItemName":"Operating time (full to empty)",
"csValue":[{"csValueName":"14 days",...``) because that is the exact string
that exists in the page data. Quoting or verbalizing that blob is useless, so
this module renders it as the line the shopper sees:
``Operating time (full to empty): 14 days``. The raw ``evidence_text`` stays
the source of truth (citations, offsets, validation); this is display only.
"""
from __future__ import annotations

import html
import json
import re

_PAIR = re.compile(
    r'"csItemName":"((?:[^"\\]|\\.)*)"|"csValueName":"((?:[^"\\]|\\.)*)"|"csChapterName":"((?:[^"\\]|\\.)*)"'
)


def _decode(raw: str) -> str:
    try:
        raw = json.loads(f'"{raw}"')
    except ValueError:
        pass
    return " ".join(html.unescape(raw).split())


def is_spec_json(text: str) -> bool:
    return text.lstrip().startswith('{"cs') or '"csItemName"' in text


def readable_evidence(text: str) -> str:
    """``Item: value, value; Item: value`` for a Philips spec-row snippet; any other text is returned unchanged.
    Truncated snippets are fine: only complete name/value strings are used."""
    if not is_spec_json(text):
        return text
    items: list[list] = []
    for m in _PAIR.finditer(text):
        if m.group(1) is not None:
            items.append([_decode(m.group(1)), []])
        elif m.group(2) is not None and items:
            value = _decode(m.group(2))
            if value:
                items[-1][1].append(value)
    parts = [f"{name}: {', '.join(values)}" if values else name for name, values in items]
    return "; ".join(parts) if parts else text
