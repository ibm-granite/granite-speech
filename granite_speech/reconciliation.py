from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


def normalize_word(word: str) -> str:
    start = 0
    end = len(word)
    while start < end and not word[start].isalnum():
        start += 1
    while end > start and not word[end - 1].isalnum():
        end -= 1
    return word[start:end].casefold()


def common_prefix_count(left: Sequence[str], right: Sequence[str]) -> int:
    count = 0
    limit = min(len(left), len(right))
    while count < limit and normalize_word(left[count]) == normalize_word(right[count]):
        count += 1
    return count


@dataclass
class LocalAgreementReconciler:
    confirmed: list[str] = field(default_factory=list)
    _previous_hypothesis: list[str] = field(default_factory=list)

    def update(self, hypothesis: Sequence[str]) -> tuple[list[str], list[str]]:
        words = list(hypothesis)
        agreed_count = common_prefix_count(self._previous_hypothesis, words)
        if agreed_count > len(self.confirmed):
            self.confirmed.extend(words[len(self.confirmed) : agreed_count])
        self._previous_hypothesis = words
        return list(self.confirmed), words[len(self.confirmed) :]

    def finalize(self, words: Sequence[str]) -> list[str]:
        self.confirmed = list(words)
        self._previous_hypothesis = list(words)
        return list(self.confirmed)

    def reset(self) -> None:
        self.confirmed = []
        self._previous_hypothesis = []


@dataclass(frozen=True)
class BoundaryAgreement:
    previous_end: int
    current_end: int
    length: int
    previous_drop: int
    current_drop: int


def reconcile_overlapping_chunks(texts: Sequence[str], *, overlap_fraction: float) -> list[str]:
    tokenized = [text.split() for text in texts]
    if overlap_fraction <= 0 or len(tokenized) < 2:
        return [" ".join(words).strip() for words in tokenized]

    prefix_skip = [0 for _ in tokenized]
    output_end = [len(words) for words in tokenized]

    for index in range(len(tokenized) - 1):
        previous = tokenized[index]
        current = tokenized[index + 1]
        if not previous or not current:
            continue

        agreement = find_boundary_agreement(
            previous,
            current,
            overlap_fraction=overlap_fraction,
        )
        if agreement is not None:
            output_end[index] = min(output_end[index], agreement.previous_end)
            prefix_skip[index + 1] = max(prefix_skip[index + 1], agreement.current_end)
            continue

        previous_drop = edge_word_count(previous, overlap_fraction)
        current_drop = edge_word_count(current, overlap_fraction)
        output_end[index] = min(output_end[index], max(0, len(previous) - previous_drop))
        prefix_skip[index + 1] = max(prefix_skip[index + 1], current_drop)

    reconciled: list[str] = []
    for words, start, end in zip(tokenized, prefix_skip, output_end, strict=True):
        if start >= end:
            reconciled.append("")
        else:
            reconciled.append(" ".join(words[start:end]).strip())
    return reconciled


def edge_word_count(words: Sequence[str], overlap_fraction: float) -> int:
    if not words or overlap_fraction <= 0:
        return 0
    return max(1, min(len(words), math.ceil(len(words) * overlap_fraction)))


def find_boundary_agreement(
    previous: Sequence[str],
    current: Sequence[str],
    *,
    overlap_fraction: float,
) -> BoundaryAgreement | None:
    previous_edge = edge_word_count(previous, overlap_fraction)
    current_edge = edge_word_count(current, overlap_fraction)
    best: BoundaryAgreement | None = None
    best_score: tuple[int, int, int, int] | None = None

    for previous_drop in range(previous_edge + 1):
        previous_end = len(previous) - previous_drop
        if previous_end <= 0:
            continue
        for current_drop in range(current_edge + 1):
            current_start = current_drop
            if current_start >= len(current):
                continue
            max_length = min(previous_end, len(current) - current_start)
            for length in range(max_length, 0, -1):
                previous_start = previous_end - length
                if _word_ranges_agree(
                    previous[previous_start:previous_end],
                    current[current_start : current_start + length],
                ):
                    candidate = BoundaryAgreement(
                        previous_end=previous_end,
                        current_end=current_start + length,
                        length=length,
                        previous_drop=previous_drop,
                        current_drop=current_drop,
                    )
                    score = (
                        length,
                        -(previous_drop + current_drop),
                        -previous_drop,
                        -current_drop,
                    )
                    if best_score is None or score > best_score:
                        best = candidate
                        best_score = score
                    break

    return best


def _word_ranges_agree(left: Sequence[str], right: Sequence[str]) -> bool:
    return len(left) == len(right) and all(
        normalize_word(left_word) == normalize_word(right_word)
        for left_word, right_word in zip(left, right, strict=True)
    )
