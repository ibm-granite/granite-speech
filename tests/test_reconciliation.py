from __future__ import annotations

from granite_speech.reconciliation import (
    LocalAgreementReconciler,
    find_boundary_agreement,
    normalize_word,
    reconcile_overlapping_chunks,
)


def test_local_agreement_commits_only_consecutive_common_prefix():
    reconciler = LocalAgreementReconciler()

    assert reconciler.update(["The", "cat", "sat"]) == ([], ["The", "cat", "sat"])
    assert reconciler.update(["the", "cat,", "sleeps"]) == (
        ["the", "cat,"],
        ["sleeps"],
    )
    assert reconciler.update(["the", "cat", "sleeps", "now"]) == (
        ["the", "cat,", "sleeps"],
        ["now"],
    )


def test_local_agreement_finalize_and_reset():
    reconciler = LocalAgreementReconciler()

    assert reconciler.finalize(["final", "words"]) == ["final", "words"]
    assert reconciler.confirmed == ["final", "words"]

    reconciler.reset()

    assert reconciler.confirmed == []
    assert reconciler.update(["fresh"]) == ([], ["fresh"])


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
