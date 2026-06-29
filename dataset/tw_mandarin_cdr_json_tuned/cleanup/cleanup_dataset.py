#!/usr/bin/env python3
"""
Strict cleanup pass for the Taiwanese Mandarin CDR JSON dataset.

This script never edits the source files in place. It writes cleaned copies to:
  cleanup/cleaned_dataset/

It also writes before/after audit reports to:
  cleanup/reports/
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUT_ROOT = SCRIPT_DIR / "cleaned_dataset"
REPORT_ROOT = SCRIPT_DIR / "reports"

MARKER_TO_TYPE = {
    "停頓": "pause",
    "長停頓": "long_pause",
    "嘆氣": "sigh",
    "吸氣": "breath",
    "咳嗽": "cough",
    "沉默": "silence",
}
TYPE_TO_MARKER = {v: k for k, v in MARKER_TO_TYPE.items()}
EVENT_TYPES = set(MARKER_TO_TYPE.values())
MARKER_RE = re.compile(r"\[(停頓|長停頓|嘆氣|吸氣|咳嗽|沉默)\]")
LABEL_RE = re.compile(r"(訪談者|訪問者|醫師|護理師|受訪者|患者)[:：]")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")

DEFAULT_DURATIONS_MS = {
    "pause": 650,
    "long_pause": 1500,
    "sigh": 900,
    "breath": 400,
    "cough": 600,
    "silence": 2200,
}

TRADITIONAL_REPLACEMENTS = {
    "变化": "變化",
    "这": "這",
    "说": "說",
    "没": "沒",
    "个": "個",
    "们": "們",
    "为": "為",
    "会": "會",
    "过": "過",
    "还": "還",
    "对": "對",
    "时": "時",
    "医": "醫",
    "药": "藥",
    "岁": "歲",
    "吗": "嗎",
    "点": "點",
    "听": "聽",
    "看见": "看見",
    "买": "買",
    "卖": "賣",
    "鸡": "雞",
    "鱼": "魚",
    "饭": "飯",
    "电视": "電視",
    "没关系": "沒關係",
    "什么": "什麼",
    "妈妈": "媽媽",
    "爷爷": "爺爺",
    "奶奶": "奶奶",
}

WORD_FINDING_TERMS = (
    "想不起來",
    "忘記",
    "叫什麼",
    "那個",
    "怎麼說",
    "記不得",
    "記不清",
)


@dataclass
class Audit:
    files_checked: int = 0
    invalid_json: list[dict[str, Any]] = field(default_factory=list)
    issue_counts: Counter = field(default_factory=Counter)
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    manual_review: list[dict[str, Any]] = field(default_factory=list)

    def add_issue(self, issue: str, rel_path: str, severity: str, detail: str = "") -> None:
        self.issue_counts[issue] += 1
        if len(self.examples[issue]) < 25:
            self.examples[issue].append(
                {"file": rel_path, "severity": severity, "detail": detail}
            )


def rel(path: Path, root: Path = ROOT) -> str:
    return path.relative_to(root).as_posix()


def sample_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.json"):
        if path.name == "manifest.json":
            continue
        if root == ROOT and SCRIPT_DIR in path.parents:
            continue
        files.append(path)
    return sorted(files)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def apply_traditional_replacements(text: str) -> tuple[str, list[str]]:
    hits: list[str] = []
    for src, dst in TRADITIONAL_REPLACEMENTS.items():
        if src in text:
            hits.append(f"{src}->{dst}")
            text = text.replace(src, dst)
    return text, hits


def normalize_labels(text: str) -> tuple[str, bool]:
    return re.subn(r"患者[:：]", "受訪者：", text)[0:2]


def normalize_marker_spacing(text: str) -> str:
    text = re.sub(r"\s*(\[(?:停頓|長停頓|嘆氣|吸氣|咳嗽|沉默)\])\s*", r" \1 ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_spoken_from_marked(marked: str) -> str:
    text = MARKER_RE.sub("", marked)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def marker_sequence(text: str) -> list[str]:
    return [MARKER_TO_TYPE[m.group(1)] for m in MARKER_RE.finditer(text)]


def old_non_speech_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    events = data.get("event_script")
    if not isinstance(events, list):
        return []
    return [e for e in events if isinstance(e, dict) and e.get("type") in EVENT_TYPES]


def duration_for(event_type: str, old_events: list[dict[str, Any]], marker_index: int) -> int:
    if marker_index < len(old_events):
        duration = old_events[marker_index].get("duration_ms")
        if isinstance(duration, (int, float)) and duration > 0:
            return int(duration)
    return DEFAULT_DURATIONS_MS[event_type]


def speaker_for_chunk(chunk: str, previous: str, interaction_type: str) -> str:
    labels = list(LABEL_RE.finditer(chunk))
    if labels:
        label = labels[-1].group(1)
        if label in {"訪談者", "訪問者", "醫師", "護理師"}:
            return "interviewer"
        return "participant"
    if interaction_type == "two_person_conversation":
        return previous
    return "participant"


def strip_event_label(chunk: str) -> str:
    return LABEL_RE.sub(lambda m: "受訪者：" if m.group(1) in {"患者", "受訪者"} else m.group(0), chunk).strip()


def build_event_script(marked: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    old_events = old_non_speech_events(data)
    marker_index = 0
    previous_speaker = "participant"
    interaction_type = data.get("interaction_type", "")
    pos = 0

    for match in MARKER_RE.finditer(marked):
        chunk = marked[pos : match.start()]
        chunk_text = strip_event_label(chunk)
        if chunk_text:
            previous_speaker = speaker_for_chunk(chunk, previous_speaker, interaction_type)
            events.append(
                {
                    "type": "speech",
                    "speaker": previous_speaker,
                    "text": chunk_text,
                }
            )

        event_type = MARKER_TO_TYPE[match.group(1)]
        events.append(
            {
                "type": event_type,
                "duration_ms": duration_for(event_type, old_events, marker_index),
            }
        )
        marker_index += 1
        pos = match.end()

    tail = marked[pos:]
    tail_text = strip_event_label(tail)
    if tail_text:
        previous_speaker = speaker_for_chunk(tail, previous_speaker, interaction_type)
        events.append(
            {
                "type": "speech",
                "speaker": previous_speaker,
                "text": tail_text,
            }
        )

    return events


def chinese_char_count(text: str) -> int:
    return len(CHINESE_RE.findall(text or ""))


def recalc_event_stats(data: dict[str, Any]) -> dict[str, int]:
    events = data.get("event_script") if isinstance(data.get("event_script"), list) else []
    counter = Counter(e.get("type") for e in events if isinstance(e, dict))
    pause_events = [
        e
        for e in events
        if isinstance(e, dict) and e.get("type") in {"pause", "long_pause", "silence"}
    ]
    return {
        "speech_chunk_count": counter["speech"],
        "pause_event_count": len(pause_events),
        "sigh_event_count": counter["sigh"],
        "cough_event_count": counter["cough"],
        "breath_event_count": counter["breath"],
        "total_pause_ms": sum(int(e.get("duration_ms") or 0) for e in pause_events),
        "chinese_char_count": chinese_char_count(data.get("spoken_transcript", "")),
    }


def visible_hesitation_count(text: str) -> int:
    filler_count = len(re.findall(r"嗯|欸|呃|那個", text))
    pause_count = len(re.findall(r"\[(?:停頓|長停頓|沉默)\]", text))
    return filler_count + pause_count


def visible_word_finding_count(text: str) -> int:
    return sum(1 for term in WORD_FINDING_TERMS if term in text)


def clinically_suspicious(data: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    cdr = data.get("cdr_level")
    stats = data.get("event_stats", {})
    features = data.get("impairment_features", {})
    pauses = stats.get("pause_event_count", 0)
    chars = stats.get("chinese_char_count", 0)
    coherence = features.get("coherence_score")
    hesitation = features.get("hesitation_count", 0)
    word_finding = features.get("word_finding_count", 0)
    marked = data.get("marked_transcript", "")

    if cdr == 0 and pauses >= 5:
        issues.append("CDR 0 has too many pause events for normal speech")
    if cdr == 0.5 and (pauses >= 6 or hesitation >= 5 or word_finding >= 1):
        issues.append("CDR 0.5 may be too impaired")
    if cdr == 1 and pauses <= 2 and hesitation <= 3 and word_finding <= 1:
        issues.append("CDR 1 may be too normal")
    if cdr == 2 and data.get("scenario") == "picture_description" and chars > 85:
        issues.append("CDR 2 picture description may be too complete")
    if cdr == 3 and chars > 135:
        issues.append("CDR 3 sample may be too long/fluent")
    if isinstance(coherence, (int, float)) and cdr in {2, 3} and coherence > 0.85:
        issues.append("coherence_score is high for assigned CDR level")
    if features.get("word_finding_count", 0) > 0 and visible_word_finding_count(marked) == 0:
        issues.append("word_finding_count is not visibly supported")

    return issues


def audit_file(data: dict[str, Any], rel_path: str, audit: Audit) -> None:
    audit.files_checked += 1
    required = {
        "sample_id",
        "cdr_level",
        "cdr_label",
        "scenario",
        "task_type",
        "interaction_type",
        "picture_reference",
        "speaker",
        "acoustic_condition",
        "spoken_transcript",
        "marked_transcript",
        "event_script",
        "event_stats",
        "impairment_features",
    }
    missing = sorted(required - set(data))
    if missing:
        audit.add_issue("missing_required_fields", rel_path, "severe", ", ".join(missing))

    if "speech_rate_target" not in data:
        audit.add_issue("missing_top_level_speech_rate_target", rel_path, "medium")

    features = data.get("impairment_features", {})
    if isinstance(features, dict) and "pause_event_count" in features:
        audit.add_issue("duplicated_pause_event_count", rel_path, "medium")

    spoken = data.get("spoken_transcript", "")
    marked = data.get("marked_transcript", "")
    if MARKER_RE.search(spoken):
        audit.add_issue("spoken_transcript_contains_markers", rel_path, "severe")

    if "患者：" in spoken + marked or "患者:" in spoken + marked:
        audit.add_issue("contains_patient_label", rel_path, "medium")

    marked_seq = marker_sequence(marked)
    event_seq = [
        e.get("type")
        for e in data.get("event_script", [])
        if isinstance(e, dict) and e.get("type") in EVENT_TYPES
    ]
    if marked_seq != event_seq:
        audit.add_issue(
            "marker_event_sequence_mismatch",
            rel_path,
            "severe",
            f"marked={len(marked_seq)} event={len(event_seq)}",
        )

    recalculated = recalc_event_stats(data)
    if data.get("event_stats") != recalculated:
        audit.add_issue("event_stats_mismatch", rel_path, "severe")

    for src in TRADITIONAL_REPLACEMENTS:
        if src in spoken + marked:
            audit.add_issue("simplified_chinese_detected", rel_path, "medium", src)
            break

    for issue in clinically_suspicious(data):
        audit.manual_review.append(
            {
                "file": rel_path,
                "sample_id": data.get("sample_id"),
                "cdr_level": data.get("cdr_level"),
                "issue": issue,
            }
        )


def clean_file(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    cleaned = json.loads(json.dumps(data, ensure_ascii=False))
    changes: list[str] = []

    for field_name in ("spoken_transcript", "marked_transcript"):
        value = cleaned.get(field_name)
        if isinstance(value, str):
            normalized, label_count = normalize_labels(value)
            normalized, replacements = apply_traditional_replacements(normalized)
            if field_name == "marked_transcript":
                normalized = normalize_marker_spacing(normalized)
            if normalized != value:
                cleaned[field_name] = normalized
                changes.append(f"{field_name}:normalized")
            if label_count:
                changes.append(f"{field_name}:patient_label_to_respondent")
            if replacements:
                changes.append(f"{field_name}:traditional:{'|'.join(replacements)}")

    cleaned["spoken_transcript"] = clean_spoken_from_marked(cleaned.get("marked_transcript", ""))
    changes.append("spoken_transcript:rebuilt_from_marked_without_markers")

    cleaned["event_script"] = build_event_script(cleaned.get("marked_transcript", ""), cleaned)
    changes.append("event_script:rebuilt_from_marked_transcript")

    cleaned["event_stats"] = recalc_event_stats(cleaned)
    changes.append("event_stats:recalculated")

    features = cleaned.get("impairment_features")
    if isinstance(features, dict):
        if "pause_event_count" in features:
            features.pop("pause_event_count", None)
            changes.append("impairment_features:removed_pause_event_count")
        if "speech_rate_target" in features:
            cleaned["speech_rate_target"] = features["speech_rate_target"]
            changes.append("speech_rate_target:promoted_to_top_level")

    for event in cleaned.get("event_script", []):
        if isinstance(event, dict) and event.get("speaker") == "patient":
            event["speaker"] = "participant"
            changes.append("event_script:patient_speaker_to_participant")

    return cleaned, sorted(set(changes))


def build_manifest(files: list[Path], out_root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for path in sorted(files):
        data = read_json(path)
        speaker = data.get("speaker", {}) if isinstance(data.get("speaker"), dict) else {}
        cleaned_rel = path.relative_to(out_root).as_posix()
        manifest.append(
            {
                "sample_id": data.get("sample_id"),
                "path": f"cleanup/cleaned_dataset/{cleaned_rel}",
                "cdr_level": data.get("cdr_level"),
                "cdr_label": data.get("cdr_label"),
                "scenario": data.get("scenario"),
                "task_type": data.get("task_type"),
                "interaction_type": data.get("interaction_type"),
                "voice_id": speaker.get("voice_id"),
                "speaker_group": speaker.get("speaker_group"),
                "split": data.get("split"),
                "acoustic_condition": data.get("acoustic_condition"),
            }
        )
    return manifest


def write_audit(prefix: str, audit: Audit) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "files_checked": audit.files_checked,
        "invalid_json_count": len(audit.invalid_json),
        "issue_counts": dict(sorted(audit.issue_counts.items())),
        "examples": audit.examples,
        "manual_review_count": len(audit.manual_review),
    }
    write_json(REPORT_ROOT / f"{prefix}_audit_summary.json", summary)
    write_json(REPORT_ROOT / f"{prefix}_manual_review.json", audit.manual_review)

    with (REPORT_ROOT / f"{prefix}_manual_review.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "sample_id", "cdr_level", "issue"])
        writer.writeheader()
        writer.writerows(audit.manual_review)


def audit_dataset(root: Path) -> Audit:
    audit = Audit()
    for path in sample_files(root):
        try:
            data = read_json(path)
        except Exception as exc:
            audit.invalid_json.append({"file": rel(path, root), "error": str(exc)})
            continue
        audit_file(data, rel(path, root), audit)
    return audit


def main() -> int:
    source_files = sample_files(ROOT)
    if not source_files:
        raise SystemExit("No source JSON files found.")

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    before = audit_dataset(ROOT)
    write_audit("before", before)

    change_rows: list[dict[str, Any]] = []
    cleaned_files: list[Path] = []
    for source_path in source_files:
        data = read_json(source_path)
        cleaned, changes = clean_file(data)
        out_path = OUT_ROOT / source_path.relative_to(ROOT)
        write_json(out_path, cleaned)
        cleaned_files.append(out_path)
        change_rows.append(
            {
                "file": source_path.relative_to(ROOT).as_posix(),
                "sample_id": cleaned.get("sample_id"),
                "change_count": len(changes),
                "changes": "; ".join(changes),
            }
        )

    write_json(OUT_ROOT / "manifest.json", build_manifest(cleaned_files, OUT_ROOT))

    with (REPORT_ROOT / "changes.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "sample_id", "change_count", "changes"])
        writer.writeheader()
        writer.writerows(change_rows)

    after = audit_dataset(OUT_ROOT)
    write_audit("after", after)

    strict_blockers = {
        "missing_required_fields",
        "duplicated_pause_event_count",
        "spoken_transcript_contains_markers",
        "contains_patient_label",
        "marker_event_sequence_mismatch",
        "event_stats_mismatch",
        "simplified_chinese_detected",
    }
    remaining_blockers = {
        key: count for key, count in after.issue_counts.items() if key in strict_blockers and count
    }

    markdown = [
        "# Cleanup Run Report",
        "",
        f"- Source sample files: {len(source_files)}",
        f"- Cleaned sample files: {len(cleaned_files)}",
        f"- Before blocker counts: {dict(sorted(before.issue_counts.items()))}",
        f"- After blocker counts: {dict(sorted(after.issue_counts.items()))}",
        f"- Manual clinical review items after cleanup: {len(after.manual_review)}",
        "",
        "## Strict Result",
        "",
    ]
    if remaining_blockers:
        markdown.append("FAIL: mechanical blockers remain and require script/code review.")
        markdown.append("")
        markdown.append(json.dumps(remaining_blockers, ensure_ascii=False, indent=2))
    else:
        markdown.append("PASS: strict mechanical cleanup checks passed.")
    markdown.extend(
        [
            "",
            "## Outputs",
            "",
            "- `cleanup/cleaned_dataset/`",
            "- `cleanup/cleaned_dataset/manifest.json`",
            "- `cleanup/reports/before_audit_summary.json`",
            "- `cleanup/reports/after_audit_summary.json`",
            "- `cleanup/reports/changes.csv`",
            "- `cleanup/reports/after_manual_review.csv`",
        ]
    )
    (REPORT_ROOT / "cleanup_run_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"source_files={len(source_files)}")
    print(f"cleaned_files={len(cleaned_files)}")
    print(f"before_issues={dict(sorted(before.issue_counts.items()))}")
    print(f"after_issues={dict(sorted(after.issue_counts.items()))}")
    print(f"after_manual_review={len(after.manual_review)}")
    print(f"strict_blockers={remaining_blockers}")
    return 1 if remaining_blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
