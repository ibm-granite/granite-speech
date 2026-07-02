from __future__ import annotations

from granite_speech.reconciliation import (
    find_boundary_agreement,
    normalize_word,
    reconcile_overlapping_chunks,
)


def test_normalize_word_strips_edge_punctuation_and_uses_casefold():
    assert normalize_word('"Cat,"') == "cat"
    assert normalize_word("Straße!") == "strasse"


def test_reconcile_overlapping_chunks_deduplicates_normalized_boundary_agreement():
    assert reconcile_overlapping_chunks(
        ["Hello, shared words!", "shared words next"],
        overlap_fraction=1 / 3,
    ) == ["Hello, shared words!", "next"]


def test_reconcile_overlapping_chunks_drops_unconfirmed_boundary_edges():
    assert reconcile_overlapping_chunks(
        ["alpha noisy", "wrong omega"],
        overlap_fraction=1 / 3,
    ) == ["alpha", "omega"]


def test_reconcile_overlapping_chunks_allows_unstable_tail_before_agreement():
    assert reconcile_overlapping_chunks(
        ["alpha shared words noisy", "shared words omega"],
        overlap_fraction=1 / 3,
    ) == ["alpha shared words", "omega"]


def test_boundary_agreement_reports_skipped_unstable_edges():
    agreement = find_boundary_agreement(
        ["alpha", "shared", "words", "noisy"],
        ["wrong", "shared", "words", "omega"],
        overlap_fraction=1 / 3,
    )

    assert agreement is not None
    assert agreement.length == 2
    assert agreement.previous_drop == 1
    assert agreement.current_drop == 1
