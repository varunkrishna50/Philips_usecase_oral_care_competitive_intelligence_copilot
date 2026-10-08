"""Tests for 3C.6 citation metrics aggregation."""
from __future__ import annotations

from src.evaluation.citation_metrics import aggregate_citation_metrics, compute_citation_metrics
from src.rag.citations import ValidatedSegment


def _vs(valid: bool, trusted: bool, citation_ids=None) -> ValidatedSegment:
    return ValidatedSegment(text="x", citation_ids=citation_ids or (["e1"] if valid else []), valid=valid, trusted=trusted)


class TestComputeCitationMetrics:
    def test_empty_input(self):
        m = compute_citation_metrics([])
        assert m.total_segments == 0
        assert m.presence_rate == 0.0

    def test_all_trusted_segments_are_100pct_valid_by_construction(self):
        segs = [_vs(True, trusted=True) for _ in range(5)]
        m = compute_citation_metrics(segs)
        assert m.trusted_segments == 5
        assert m.untrusted_segments == 0
        assert m.validity_rate_all == 1.0
        # No untrusted segments at all -- defined as 1.0 (vacuously true), not 0/0.
        assert m.validity_rate_untrusted_only == 1.0

    def test_mixed_trusted_and_untrusted_reported_separately(self):
        segs = [_vs(True, trusted=True), _vs(True, trusted=True), _vs(False, trusted=False)]
        m = compute_citation_metrics(segs)
        assert m.trusted_segments == 2
        assert m.untrusted_segments == 1
        assert m.validity_rate_all == 2 / 3
        # Only the 1 untrusted segment counts here, and it's invalid.
        assert m.validity_rate_untrusted_only == 0.0

    def test_presence_rate_counts_citation_ids(self):
        segs = [_vs(True, trusted=True, citation_ids=["e1"]), _vs(False, trusted=False, citation_ids=[])]
        m = compute_citation_metrics(segs)
        assert m.presence_rate == 0.5


class TestAggregateCitationMetrics:
    def test_aggregates_across_cases_weighted_by_segment_count(self):
        m1 = compute_citation_metrics([_vs(True, trusted=True), _vs(True, trusted=True)])
        m2 = compute_citation_metrics([_vs(False, trusted=False)])
        agg = aggregate_citation_metrics([m1, m2])
        assert agg.total_segments == 3
        assert agg.trusted_segments == 2
        assert agg.untrusted_segments == 1
        assert agg.validity_rate_untrusted_only == 0.0

    def test_empty_list(self):
        agg = aggregate_citation_metrics([])
        assert agg.total_segments == 0
