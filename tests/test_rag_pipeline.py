"""Tests for the 3b CLI (3B.10): `python -m src.rag.pipeline --query ...`."""
from __future__ import annotations

from src.rag import pipeline
from src.rag.retrieve import RetrievalEngine


class TestPipelineCLI:
    def test_no_llm_flag_prints_answer_and_sources(self, monkeypatch, capsys, real_engine: RetrievalEngine):
        monkeypatch.setattr(RetrievalEngine, "load", classmethod(lambda cls, *a, **kw: real_engine))
        product = next(p for p in real_engine.product_corpus if p.attributes.get("price") is not None)
        exit_code = pipeline.main(["--query", f"What is the price of the {product.product_name}?", "--no-llm"])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert product.product_name in captured.out
        assert "Sources" in captured.out
        assert "deterministic" in captured.out.lower()

    def test_debug_flag_prints_llm_config_and_segment_validity(self, monkeypatch, capsys, real_engine: RetrievalEngine):
        monkeypatch.setattr(RetrievalEngine, "load", classmethod(lambda cls, *a, **kw: real_engine))
        product = next(p for p in real_engine.product_corpus if p.attributes.get("price") is not None)
        exit_code = pipeline.main(["--query", f"What is the price of the {product.product_name}?", "--no-llm", "--debug"])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "LLM provider=" in captured.out
        assert "used_llm=False" in captured.out
        assert "segments:" in captured.out

    def test_out_of_scope_query_prints_refusal_without_crashing(self, monkeypatch, capsys, real_engine: RetrievalEngine):
        monkeypatch.setattr(RetrievalEngine, "load", classmethod(lambda cls, *a, **kw: real_engine))
        exit_code = pipeline.main(["--query", "What is the Amazon price of this toothbrush?", "--no-llm"])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "third-party retailers" in captured.out
