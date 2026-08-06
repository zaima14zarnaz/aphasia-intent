"""Parse AphasiaBank CHAT/CLAN ``.cha`` transcripts."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

MEDIA_TIMESTAMP_RE = re.compile(r"\x15(\d+)_(\d+)\x15")
MAIN_TIER_RE = re.compile(r"^\*([A-Za-z0-9]+):\s*(.*)$")
TIER_START_RE = re.compile(r"^[*@%]")

__all__ = ["get_transcript_lines", "parse_cha_transcript"]


@dataclass(frozen=True)
class _ParsedUtterance:
    speaker: str
    text: str
    start_ts: int
    end_ts: int


def get_transcript_lines(
    cha_fname: str | Path,
    start_ts: int,
    end_ts: int,
    context_before_ms: int = 5000,
    context_after_ms: int = 5000,
    min_utterances_before: int = 1,
    min_utterances_after: int = 1,
) -> list[dict]:
    """
    Return timed main-tier utterances from ``cha_fname`` with surrounding context.

    ``start_ts`` and ``end_ts`` define the original video-segment boundaries.
    Transcript selection uses three layers:

    1. Original video interval ``[start_ts, end_ts)``:
       utterances with ``start_ts < end_ts and end_ts > start_ts``.

    2. Expanded transcript interval
       ``[max(0, start_ts - context_before_ms), end_ts + context_after_ms)``:
       all utterances overlapping this wider window are included.

    3. Neighboring-utterance context:
       at least ``min_utterances_before`` utterances immediately preceding, and
       at least ``min_utterances_after`` utterances immediately following, the
       first/last relevant utterance in transcript/file order. When no utterance
       overlaps either interval, neighbors are chosen around the insertion point
       for ``start_ts``.

    All speakers parsed from main tiers (for example ``PAR`` and ``INV``) are
    returned. Each result keeps the utterance's own ``start_ts`` and ``end_ts``.

    Setting all context arguments to zero reproduces strict overlap against only
    ``[start_ts, end_ts)``.
    """
    _validate_query_window(start_ts, end_ts)
    _validate_context_args(
        context_before_ms=context_before_ms,
        context_after_ms=context_after_ms,
        min_utterances_before=min_utterances_before,
        min_utterances_after=min_utterances_after,
    )

    cha_path = _resolve_cha_path(cha_fname)
    utterances = _parse_cha_transcript(cha_path)
    selected_indices = _select_transcript_indices(
        utterances,
        start_ts=start_ts,
        end_ts=end_ts,
        context_before_ms=context_before_ms,
        context_after_ms=context_after_ms,
        min_utterances_before=min_utterances_before,
        min_utterances_after=min_utterances_after,
    )

    return [
        {
            "speaker": utterances[index].speaker,
            "text": utterances[index].text,
            "start_ts": utterances[index].start_ts,
            "end_ts": utterances[index].end_ts,
        }
        for index in selected_indices
    ]


def _utterance_overlaps_window(
    utterance: _ParsedUtterance,
    window_start_ts: int,
    window_end_ts: int,
) -> bool:
    return utterance.start_ts < window_end_ts and utterance.end_ts > window_start_ts


def _validate_context_args(
    *,
    context_before_ms: int,
    context_after_ms: int,
    min_utterances_before: int,
    min_utterances_after: int,
) -> None:
    context_values = {
        "context_before_ms": context_before_ms,
        "context_after_ms": context_after_ms,
        "min_utterances_before": min_utterances_before,
        "min_utterances_after": min_utterances_after,
    }
    for name, value in context_values.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be a nonnegative integer, got {value!r}")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative, got {value}")


def _insertion_index(
    utterances: tuple[_ParsedUtterance, ...],
    start_ts: int,
) -> int:
    """Return the file-order insertion index for ``start_ts`` when there is no overlap."""
    for index, utterance in enumerate(utterances):
        if utterance.start_ts >= start_ts:
            return index
    return len(utterances)


def _select_transcript_indices(
    utterances: tuple[_ParsedUtterance, ...],
    *,
    start_ts: int,
    end_ts: int,
    context_before_ms: int,
    context_after_ms: int,
    min_utterances_before: int,
    min_utterances_after: int,
) -> list[int]:
    if not utterances:
        return []

    expanded_start_ts = max(0, start_ts - context_before_ms)
    expanded_end_ts = end_ts + context_after_ms

    selected_indices: set[int] = set()

    expanded_indices = [
        index
        for index, utterance in enumerate(utterances)
        if _utterance_overlaps_window(utterance, expanded_start_ts, expanded_end_ts)
    ]
    original_indices = [
        index
        for index, utterance in enumerate(utterances)
        if _utterance_overlaps_window(utterance, start_ts, end_ts)
    ]

    selected_indices.update(expanded_indices)

    if original_indices:
        anchor_first = min(original_indices)
        anchor_last = max(original_indices)
    elif expanded_indices:
        anchor_first = min(expanded_indices)
        anchor_last = max(expanded_indices)
    else:
        insertion_index = _insertion_index(utterances, start_ts)
        anchor_first = insertion_index
        anchor_last = insertion_index - 1

    before_added = 0
    index = anchor_first - 1
    while index >= 0 and before_added < min_utterances_before:
        selected_indices.add(index)
        before_added += 1
        index -= 1

    after_added = 0
    index = anchor_last + 1
    while index < len(utterances) and after_added < min_utterances_after:
        selected_indices.add(index)
        after_added += 1
        index += 1

    return sorted(selected_indices)


def parse_cha_transcript(cha_fname: str | Path) -> list[dict]:
    """Parse all timed main-tier utterances from a CHAT ``.cha`` file."""
    cha_path = _resolve_cha_path(cha_fname)
    return [
        {
            "speaker": utterance.speaker,
            "text": utterance.text,
            "start_ts": utterance.start_ts,
            "end_ts": utterance.end_ts,
        }
        for utterance in _parse_cha_transcript(cha_path)
    ]


def _resolve_cha_path(cha_fname: str | Path) -> Path:
    cha_path = Path(cha_fname).expanduser().resolve()
    if not cha_path.is_file():
        raise FileNotFoundError(f"CHA transcript not found: {cha_path}")
    return cha_path


def _validate_query_window(start_ts: int, end_ts: int) -> None:
    if not isinstance(start_ts, int) or isinstance(start_ts, bool):
        raise TypeError(
            f"start_ts must be an integer number of milliseconds, got {start_ts!r}"
        )
    if not isinstance(end_ts, int) or isinstance(end_ts, bool):
        raise TypeError(
            f"end_ts must be an integer number of milliseconds, got {end_ts!r}"
        )
    if start_ts < 0:
        raise ValueError(f"start_ts must be nonnegative, got {start_ts}")
    if end_ts < 0:
        raise ValueError(f"end_ts must be nonnegative, got {end_ts}")
    if start_ts >= end_ts:
        raise ValueError(
            f"start_ts must be strictly less than end_ts, got "
            f"start_ts={start_ts}, end_ts={end_ts}"
        )


@lru_cache(maxsize=32)
def _parse_cha_transcript(cha_path: Path) -> tuple[_ParsedUtterance, ...]:
    """Parse timed main-tier utterances from a CHAT file."""
    text = cha_path.read_text(encoding="utf-8-sig")
    utterances: list[_ParsedUtterance] = []

    current_speaker: str | None = None
    current_parts: list[str] = []

    def flush_current_utterance() -> None:
        nonlocal current_speaker, current_parts
        if current_speaker is None:
            return

        raw_text = " ".join(part for part in current_parts if part)
        parsed = _parse_main_tier_text(current_speaker, raw_text)
        if parsed is not None:
            utterances.append(parsed)

        current_speaker = None
        current_parts = []

    for line in text.splitlines():
        if TIER_START_RE.match(line):
            flush_current_utterance()
            main_tier = MAIN_TIER_RE.match(line)
            if main_tier is None:
                continue

            current_speaker = main_tier.group(1)
            current_parts = [main_tier.group(2).strip()]
            continue

        if current_speaker is not None:
            current_parts.append(line.strip())

    flush_current_utterance()
    return tuple(utterances)


def _parse_main_tier_text(
    speaker: str,
    raw_text: str,
) -> _ParsedUtterance | None:
    """Parse one logical main-tier utterance, or skip it if untimed/invalid."""
    if not raw_text:
        return None

    timestamp_matches = list(MEDIA_TIMESTAMP_RE.finditer(raw_text))
    if not timestamp_matches:
        return None

    final_match = timestamp_matches[-1]
    try:
        start_ts = int(final_match.group(1))
        end_ts = int(final_match.group(2))
    except ValueError:
        return None

    if start_ts < 0 or end_ts < 0 or start_ts >= end_ts:
        return None

    text = MEDIA_TIMESTAMP_RE.sub("", raw_text)
    text = text.strip()
    if not text:
        return None

    return _ParsedUtterance(
        speaker=speaker,
        text=text,
        start_ts=start_ts,
        end_ts=end_ts,
    )


def _run_self_tests() -> None:
    sample = (
        "@Begin\n"
        "*INV: can you tell me about your trip ? \x154075_7935\x15\n"
        "%mor: mod|can pro:per|you v|tell\n"
        "*PAR: &-um I caught the flu from my mom . \x15113123_114473\x15\n"
        "*PAR: and I suppose &uh everything [/] everything better all the\n"
        "\ttime . \x15117175_122645\x15\n"
        "*PAR: no timestamp here .\n"
        "*PAR: bad timestamp \x15100_100\x15\n"
        "@End\n"
    )

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".cha") as handle:
        handle.write(sample)
        handle.flush()
        cha_fname = handle.name

        assert len(parse_cha_transcript(cha_fname)) == 3

        strict_lines = get_transcript_lines(
            cha_fname,
            113000,
            115000,
            context_before_ms=0,
            context_after_ms=0,
            min_utterances_before=0,
            min_utterances_after=0,
        )
        assert len(strict_lines) == 1
        assert strict_lines[0]["speaker"] == "PAR"
        assert strict_lines[0]["start_ts"] == 113123
        assert strict_lines[0]["end_ts"] == 114473
        assert "&-um" in strict_lines[0]["text"]
        assert "\x15" not in strict_lines[0]["text"]

        contextual_lines = get_transcript_lines(
            cha_fname,
            113000,
            115000,
            context_after_ms=0,
            min_utterances_after=0,
        )
        assert len(contextual_lines) == 2
        assert contextual_lines[0]["speaker"] == "INV"
        assert contextual_lines[1]["speaker"] == "PAR"
        assert contextual_lines[1]["start_ts"] == 113123

        generous_lines = get_transcript_lines(cha_fname, 113000, 115000)
        assert len(generous_lines) == 3
        assert generous_lines[-1]["start_ts"] == 117175

        multiline = get_transcript_lines(
            cha_fname,
            117000,
            123000,
            context_before_ms=0,
            context_after_ms=0,
            min_utterances_before=0,
            min_utterances_after=0,
        )
        assert len(multiline) == 1
        assert "[/]" in multiline[0]["text"]
        assert "everything better all the time ." in multiline[0]["text"]

        overlapping = get_transcript_lines(
            cha_fname,
            4000,
            114500,
            context_before_ms=0,
            context_after_ms=0,
            min_utterances_before=0,
            min_utterances_after=0,
        )
        assert len(overlapping) == 2
        assert overlapping[0]["speaker"] == "INV"
        assert overlapping[1]["speaker"] == "PAR"

        gap_lines = get_transcript_lines(
            cha_fname,
            10000,
            11000,
            context_before_ms=0,
            context_after_ms=0,
            min_utterances_before=1,
            min_utterances_after=1,
        )
        assert len(gap_lines) == 2
        assert gap_lines[0]["speaker"] == "INV"
        assert gap_lines[1]["speaker"] == "PAR"

        expanded_only = get_transcript_lines(
            cha_fname,
            10000,
            11000,
            context_before_ms=10000,
            context_after_ms=2000,
            min_utterances_before=0,
            min_utterances_after=0,
        )
        assert len(expanded_only) == 1
        assert expanded_only[0]["speaker"] == "INV"

        try:
            get_transcript_lines(cha_fname, 100, 100)
            raise AssertionError("Expected ValueError for empty query window")
        except ValueError:
            pass

        try:
            get_transcript_lines(cha_fname, True, 200)  # type: ignore[arg-type]
            raise AssertionError("Expected TypeError for bool start_ts")
        except TypeError:
            pass

        try:
            get_transcript_lines(
                cha_fname,
                0,
                1000,
                context_before_ms=-1,
            )
            raise AssertionError("Expected ValueError for negative context")
        except ValueError:
            pass


if __name__ == "__main__":
    _run_self_tests()
    print("read_transcript self-tests passed.")

    example_path = Path(
        "/data/quantization/zaima/aphasiabank/neural2/control/annotations/"
        "aphasia/rc101_day1_AB.elan.cha"
    )
    if example_path.is_file():
        lines = get_transcript_lines(example_path, 113000, 115000)
        print(f"Example query returned {len(lines)} line(s) from {example_path.name}")
        print(lines[:2])
