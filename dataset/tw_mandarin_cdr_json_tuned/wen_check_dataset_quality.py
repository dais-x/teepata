#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use Qwen via Ollama to check Taiwanese Mandarin dataset quality.

Checks:
1. Taiwanese Mandarin naturalness
2. Simplified Chinese / Mainland-style vocabulary
3. marked_transcript and event_script consistency
4. Event marker count consistency
5. Whether event_script speech text contains unwanted markers/biomarkers

This script DOES NOT modify JSON files.
It only creates a TXT report for manual review.

Example:
python wen_check_dataset_quality.py --input-dir cdr_0 --model qwen35b-q4 --output-file qwen_dataset_quality_report.txt
"""

import argparse
import json
import time
import re
import urllib.request
import urllib.error
from pathlib import Path


MARKER_TO_EVENT_TYPE = {
    "[停頓]": "pause",
    "[長停頓]": "long_pause",
    "[沉默]": "silence",
    "[嘆氣]": "sigh",
    "[咳嗽]": "cough",
}

EVENT_MARKERS = list(MARKER_TO_EVENT_TYPE.keys())


def extract_relevant_text(data):
    """
    Extract only fields needed for Qwen review.
    """
    extracted = {}

    if not isinstance(data, dict):
        return extracted

    for key in [
        "sample_id",
        "cdr_level",
        "cdr_label",
        "scenario",
        "task_type",
        "interaction_type",
        "spoken_transcript",
        "marked_transcript",
    ]:
        if key in data:
            extracted[key] = data.get(key)

    event_texts = []
    event_script = data.get("event_script")

    if isinstance(event_script, list):
        for i, event in enumerate(event_script):
            if isinstance(event, dict):
                event_texts.append({
                    "index": i,
                    "type": event.get("type", ""),
                    "speaker": event.get("speaker", ""),
                    "text": event.get("text", ""),
                    "duration_ms": event.get("duration_ms", None),
                })

    extracted["event_script"] = event_texts

    return extracted


def count_markers(marked_transcript):
    """
    Count pause/sigh/cough/silence markers inside marked_transcript.
    """
    counts = {}

    if not isinstance(marked_transcript, str):
        marked_transcript = ""

    for marker in EVENT_MARKERS:
        counts[marker] = marked_transcript.count(marker)

    return counts


def count_event_types(event_script):
    """
    Count pause/long_pause/silence/sigh/cough events inside event_script.
    """
    counts = {
        "pause": 0,
        "long_pause": 0,
        "silence": 0,
        "sigh": 0,
        "cough": 0,
    }

    if not isinstance(event_script, list):
        return counts

    for event in event_script:
        if isinstance(event, dict):
            event_type = event.get("type")
            if event_type in counts:
                counts[event_type] += 1

    return counts


def remove_markers(text):
    """
    Remove event markers from marked_transcript.
    """
    if not isinstance(text, str):
        return ""

    for marker in EVENT_MARKERS:
        text = text.replace(marker, "")

    return normalize_text(text)


def normalize_text(text):
    """
    Normalize text for rough comparison.
    """
    if not isinstance(text, str):
        return ""

    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    text = text.replace("，", "")
    text = text.replace("。", "")
    text = text.replace("、", "")
    text = text.replace("？", "")
    text = text.replace("！", "")
    text = text.replace(",", "")
    text = text.replace(".", "")
    text = text.replace("?", "")
    text = text.replace("!", "")
    return text.strip()


def reconstruct_event_speech(event_script):
    """
    Join speech text from event_script.
    """
    if not isinstance(event_script, list):
        return ""

    parts = []

    for event in event_script:
        if not isinstance(event, dict):
            continue

        if event.get("type") == "speech":
            text = event.get("text", "")
            if isinstance(text, str):
                parts.append(text)

    return normalize_text("".join(parts))


def detect_markers_inside_speech(event_script):
    """
    Speech events should not contain [停頓], [長停頓], [嘆氣], etc.
    Those should be separate event types.
    """
    findings = []

    if not isinstance(event_script, list):
        return findings

    for i, event in enumerate(event_script):
        if not isinstance(event, dict):
            continue

        if event.get("type") != "speech":
            continue

        text = event.get("text", "")
        if not isinstance(text, str):
            continue

        found = [marker for marker in EVENT_MARKERS if marker in text]

        if found:
            findings.append({
                "index": i,
                "text": text,
                "markers_found": found,
            })

    return findings


def rule_based_event_check(data):
    """
    Fast deterministic checks before Qwen.
    """
    findings = []

    marked_transcript = data.get("marked_transcript", "")
    event_script = data.get("event_script", [])

    marker_counts = count_markers(marked_transcript)
    event_counts = count_event_types(event_script)

    for marker, event_type in MARKER_TO_EVENT_TYPE.items():
        marker_count = marker_counts.get(marker, 0)
        event_count = event_counts.get(event_type, 0)

        if marker_count != event_count:
            findings.append({
                "type": "marker_event_count_mismatch",
                "marker": marker,
                "event_type": event_type,
                "marked_transcript_count": marker_count,
                "event_script_count": event_count,
                "message": f"{marker} count in marked_transcript is {marker_count}, but {event_type} count in event_script is {event_count}."
            })

    markers_inside_speech = detect_markers_inside_speech(event_script)

    for item in markers_inside_speech:
        findings.append({
            "type": "marker_inside_speech_event",
            "event_index": item["index"],
            "markers_found": item["markers_found"],
            "text": item["text"],
            "message": "Speech event contains event markers. These should usually be separate event_script events."
        })

    marked_without_markers = remove_markers(marked_transcript)
    event_speech_joined = reconstruct_event_speech(event_script)

    if marked_without_markers and event_speech_joined:
        if marked_without_markers != event_speech_joined:
            findings.append({
                "type": "speech_text_mismatch",
                "message": "marked_transcript text without markers does not exactly match joined event_script speech text.",
                "marked_without_markers_preview": marked_without_markers[:120],
                "event_speech_joined_preview": event_speech_joined[:120],
            })

    return findings


def build_prompt(filename, extracted, rule_findings):
    return f"""
You are checking a Taiwanese Mandarin dementia speech dataset.

Your task has TWO parts:

PART A: Taiwanese Mandarin language quality
Check whether the transcript sounds like natural Taiwanese Mandarin used in Taiwan.

Flag:
1. Simplified Chinese characters
2. Mainland China vocabulary
3. Mainland-style phrasing
4. Wording unnatural in Taiwan
5. Words that should be replaced with common Taiwan Mandarin

Examples:
- 公交車 → 公車
- 出租車 → 計程車
- 地鐵 → 捷運
- 視頻 → 影片 / 視訊
- 短信 → 簡訊
- 外賣 → 外送
- 小區 → 社區
- 老年人 → 長輩 / 老人家
- 康復 → 復健
- 普通話 → 國語 / 中文
- 信息 → 資訊 / 訊息
- 質量 → 品質
- 打印 → 列印
- 屏幕 → 螢幕
- 軟件 → 軟體
- 硬件 → 硬體
- 文件夾 → 資料夾

PART B: marked_transcript and event_script consistency
Check whether:
1. marked_transcript and event_script describe the same speech.
2. Speech order in event_script matches marked_transcript.
3. [停頓], [長停頓], [沉默], [嘆氣], [咳嗽] markers are aligned with corresponding event_script events.
4. event_script speech events do NOT contain markers like [停頓] inside text.
5. Pauses/sighs/coughs are represented as separate events, not hidden inside speech text.
6. The transcript should still keep dementia-like hesitation, repetition, incomplete speech, and word-finding behavior. Do NOT treat dementia symptoms as errors.

Important:
- Do NOT rewrite the full transcript.
- Do NOT judge whether the CDR level is clinically correct.
- Do NOT complain about pauses or incomplete speech unless the event_script alignment is wrong.
- Be strict but fair.

Return ONLY valid JSON in this exact format:

{{
  "needs_manual_review": true,
  "overall_rating": 0,
  "language_rating": 0,
  "event_alignment_rating": 0,
  "summary": "short explanation",
  "language_issues": [
    {{
      "field": "spoken_transcript",
      "problem_phrase": "公交車",
      "problem_type": "Mainland vocabulary",
      "suggested_taiwan_usage": "公車",
      "reason": "In Taiwan, 公車 is the normal term."
    }}
  ],
  "event_alignment_issues": [
    {{
      "problem_type": "marker_event_mismatch",
      "field": "marked_transcript/event_script",
      "problem": "[停頓] appears in marked_transcript but corresponding pause event is missing or misplaced.",
      "suggested_fix": "Add a pause event at the matching position in event_script."
    }}
  ]
}}

Rules:
- overall_rating, language_rating, and event_alignment_rating must be 1 to 10.
- 10 = very good.
- 7 to 8 = mostly okay, minor issues.
- 5 to 6 = several issues needing review.
- below 5 = unreliable.
- If there are no issues:
  - needs_manual_review: false
  - language_issues: []
  - event_alignment_issues: []
- If rule_based_findings already show problems, include them in event_alignment_issues.

Filename:
{filename}

Rule-based findings:
{json.dumps(rule_findings, ensure_ascii=False, indent=2)}

JSON content to check:
{json.dumps(extracted, ensure_ascii=False, indent=2)}
""".strip()


def call_ollama(prompt, model, host, temperature=0.0):
    url = f"{host.rstrip('/')}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
        }
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama connection failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")


def extract_json_from_response(response):
    response = response.strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Could not parse JSON from model response.")


def format_report_entry(json_path, result, rule_findings):
    lines = []
    lines.append("=" * 100)
    lines.append(f"FILE: {json_path}")
    lines.append(f"Needs manual review: {result.get('needs_manual_review')}")
    lines.append(f"Overall rating: {result.get('overall_rating')}/10")
    lines.append(f"Language rating: {result.get('language_rating')}/10")
    lines.append(f"Event alignment rating: {result.get('event_alignment_rating')}/10")
    lines.append(f"Summary: {result.get('summary', '')}")
    lines.append("")

    if rule_findings:
        lines.append("Rule-based findings:")
        for i, finding in enumerate(rule_findings, start=1):
            lines.append(f"  {i}. {finding.get('message', finding)}")
        lines.append("")
    else:
        lines.append("Rule-based findings: None")
        lines.append("")

    language_issues = result.get("language_issues", [])
    event_issues = result.get("event_alignment_issues", [])

    if language_issues:
        lines.append(f"Language issues found: {len(language_issues)}")
        for idx, issue in enumerate(language_issues, start=1):
            lines.append(f"  Issue {idx}:")
            lines.append(f"    Field: {issue.get('field', '')}")
            lines.append(f"    Problem phrase: {issue.get('problem_phrase', '')}")
            lines.append(f"    Problem type: {issue.get('problem_type', '')}")
            lines.append(f"    Suggested Taiwan usage: {issue.get('suggested_taiwan_usage', '')}")
            lines.append(f"    Reason: {issue.get('reason', '')}")
        lines.append("")
    else:
        lines.append("Language issues: None")
        lines.append("")

    if event_issues:
        lines.append(f"Event alignment issues found: {len(event_issues)}")
        for idx, issue in enumerate(event_issues, start=1):
            lines.append(f"  Issue {idx}:")
            lines.append(f"    Problem type: {issue.get('problem_type', '')}")
            lines.append(f"    Field: {issue.get('field', '')}")
            lines.append(f"    Problem: {issue.get('problem', '')}")
            lines.append(f"    Suggested fix: {issue.get('suggested_fix', '')}")
        lines.append("")
    else:
        lines.append("Event alignment issues: None")
        lines.append("")

    return "\n".join(lines)


def scan_dataset(input_dir, output_file, model, host, limit=None, sleep_seconds=0.2):
    input_dir = Path(input_dir)
    json_files = sorted(input_dir.rglob("*.json"))

    if limit is not None:
        json_files = json_files[:limit]

    total_files = len(json_files)
    checked_files = 0
    flagged_files = 0
    error_files = 0

    report_lines = []
    report_lines.append("Qwen Taiwanese Mandarin + Event Script Dataset Quality Report")
    report_lines.append("=" * 100)
    report_lines.append(f"Input folder: {input_dir}")
    report_lines.append(f"Recursive search: YES")
    report_lines.append(f"Model: {model}")
    report_lines.append(f"Ollama host: {host}")
    report_lines.append(f"Total JSON files selected: {total_files}")
    report_lines.append("")

    for index, json_path in enumerate(json_files, start=1):
        print(f"[{index}/{total_files}] Checking: {json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            rule_findings = rule_based_event_check(data)
            extracted = extract_relevant_text(data)
            prompt = build_prompt(str(json_path), extracted, rule_findings)

            response = call_ollama(
                prompt=prompt,
                model=model,
                host=host,
                temperature=0.0,
            )

            result = extract_json_from_response(response)
            checked_files += 1

            needs_review = bool(result.get("needs_manual_review", False))
            language_issues = result.get("language_issues", [])
            event_issues = result.get("event_alignment_issues", [])

            if needs_review or language_issues or event_issues or rule_findings:
                flagged_files += 1
                report_lines.append(format_report_entry(json_path, result, rule_findings))

            time.sleep(sleep_seconds)

        except Exception as e:
            error_files += 1
            report_lines.append("=" * 100)
            report_lines.append(f"FILE: {json_path}")
            report_lines.append(f"ERROR: {e}")
            report_lines.append("")

    summary = []
    summary.append("SUMMARY")
    summary.append("=" * 100)
    summary.append(f"Total files selected: {total_files}")
    summary.append(f"Successfully checked: {checked_files}")
    summary.append(f"Flagged for manual review: {flagged_files}")
    summary.append(f"Errors: {error_files}")
    summary.append("")

    final_report = "\n".join(summary + report_lines)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_report)

    print("")
    print("Done.")
    print(f"Total files selected: {total_files}")
    print(f"Successfully checked: {checked_files}")
    print(f"Flagged for manual review: {flagged_files}")
    print(f"Errors: {error_files}")
    print(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Use Qwen via Ollama to check Taiwanese Mandarin quality and event_script alignment."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing JSON files. Subfolders are scanned automatically."
    )

    parser.add_argument(
        "--output-file",
        default="qwen_dataset_quality_report.txt",
        help="Output TXT report file."
    )

    parser.add_argument(
        "--model",
        default="qwen2.5:14b",
        help="Ollama model name, for example qwen2.5:14b or your Qwen3.5 model name."
    )

    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama host URL."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only check first N files. Useful for testing."
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between files."
    )

    args = parser.parse_args()

    scan_dataset(
        input_dir=args.input_dir,
        output_file=args.output_file,
        model=args.model,
        host=args.host,
        limit=args.limit,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()