#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python-controlled Taiwanese Mandarin CDR JSON generator.

Strategy:
- Python controls final JSON structure, metadata, speakers, splits, folders, event_script, event_stats.
- Qwen only generates transcript content with simple event markers.
- This reduces malformed JSON and makes audio generation cleaner.

Run example:
python -u generate_tw_mandarin_cdr_json_python_controlled.py \
  --model qwen3.5:35b \
  --output-dir dataset/tw_mandarin_cdr_json_v3 \
  --cdr-counts "0:300,0.5:300,1:200,2:120,3:80" \
  --temperature 0.5 \
  --top-p 0.85
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests


# ============================================================
# Configuration
# ============================================================

CDR_LABELS = {
    "0": "normal",
    "0.5": "very_mild",
    "1": "mild",
    "2": "moderate",
    "3": "severe",
}

VOICE_POOL = [
    {"voice_id": "voice1_male", "speaker_group": "male", "gender": "male", "description": "男性長輩，台灣華語，自然清楚。"},
    {"voice_id": "voice2_female", "speaker_group": "female", "gender": "female", "description": "女性長輩，台灣華語，自然清楚。"},
    {"voice_id": "voice3_male", "speaker_group": "male", "gender": "male", "description": "男性長輩，台灣華語，語速稍慢。"},
    {"voice_id": "voice4_female", "speaker_group": "female", "gender": "female", "description": "女性長輩，台灣華語，語氣柔和。"},
]

INTERVIEWER_VOICE = {
    "voice_id": "interviewer_neutral",
    "speaker_group": "interviewer",
    "gender": "neutral",
    "description": "訪談者，語氣清楚自然。",
}

SCENARIOS = [
    "clinic",
    "family",
    "home",
    "market",
    "medicine_routine",
    "daily_life",
]

TASK_TYPES = [
    {"task_type": "picture_description", "interaction_type": "one_person_description", "weight": 30},
    {"task_type": "daily_life_description", "interaction_type": "one_person_description", "weight": 25},
    {"task_type": "orientation_conversation", "interaction_type": "two_person_conversation", "weight": 15},
    {"task_type": "memory_recall_conversation", "interaction_type": "two_person_conversation", "weight": 20},
    {"task_type": "structured_cognitive_interview", "interaction_type": "two_person_conversation", "weight": 10},
]

PICTURE_REFERENCE = {
    "picture_id": "living_room_family_scene_001",
    "name": "living_room_family_scene",
    "description": (
        "一個客廳裡的家庭場景。阿公坐在沙發上看相簿，旁邊有一副老花眼鏡和一杯茶。"
        "一個小孩坐在地上玩積木，積木散在地毯上。"
        "媽媽站在窗邊接電話，一隻貓跳到茶几旁邊，好像快要碰倒茶杯。"
        "牆上有時鐘和家庭照片，窗外看起來像是下午。"
        "畫面裡有幾個人和物品，每個人都在做不同的事情。"
    ),
}

ACOUSTIC_CONDITIONS = [
    {"name": "clean", "weight": 45, "description": "乾淨近距離錄音，低背景噪音。"},
    {"name": "clinic_room_noise", "weight": 25, "description": "診間環境，輕微背景聲與空調聲。"},
    {"name": "phone_mic_degraded", "weight": 20, "description": "手機麥克風錄音，音質稍微壓縮。"},
    {"name": "home_background_noise", "weight": 10, "description": "家中背景聲，例如電風扇、遠處人聲或碗盤聲。"},
]

# Python-enforced feature ranges. These help keep CDR levels separated.
CDR_FEATURE_RANGES = {
    "0": {
        "hesitation_count": (0, 2),
        "repetition_count": (0, 1),
        "word_finding_count": (0, 0),
        "orientation_error_count": (0, 0),
        "memory_error_count": (0, 0),
        "topic_drift_score": (0.0, 0.08),
        "coherence_score": (0.92, 1.0),
        "speech_rate_target": "normal",
        "pause_events": (0, 2),
        "long_pause_events": (0, 0),
        "sigh_events": (0, 0),
        "cough_events": (0, 0),
        "breath_events": (0, 1),
    },
    "0.5": {
        "hesitation_count": (2, 5),
        "repetition_count": (0, 1),
        "word_finding_count": (0, 1),
        "orientation_error_count": (0, 1),
        "memory_error_count": (0, 1),
        "topic_drift_score": (0.08, 0.20),
        "coherence_score": (0.78, 0.92),
        "speech_rate_target": "slightly_slow",
        "pause_events": (1, 3),
        "long_pause_events": (0, 1),
        "sigh_events": (0, 1),
        "cough_events": (0, 0),
        "breath_events": (0, 1),
    },
    "1": {
        "hesitation_count": (4, 8),
        "repetition_count": (1, 3),
        "word_finding_count": (1, 3),
        "orientation_error_count": (0, 1),
        "memory_error_count": (1, 3),
        "topic_drift_score": (0.20, 0.38),
        "coherence_score": (0.62, 0.82),
        "speech_rate_target": "slow",
        "pause_events": (2, 5),
        "long_pause_events": (0, 2),
        "sigh_events": (0, 1),
        "cough_events": (0, 1),
        "breath_events": (0, 2),
    },
    "2": {
        "hesitation_count": (7, 13),
        "repetition_count": (3, 6),
        "word_finding_count": (3, 6),
        "orientation_error_count": (1, 3),
        "memory_error_count": (3, 6),
        "topic_drift_score": (0.40, 0.65),
        "coherence_score": (0.35, 0.62),
        "speech_rate_target": "very_slow",
        "pause_events": (4, 8),
        "long_pause_events": (1, 3),
        "sigh_events": (0, 2),
        "cough_events": (0, 1),
        "breath_events": (1, 3),
    },
    "3": {
        "hesitation_count": (10, 18),
        "repetition_count": (5, 10),
        "word_finding_count": (5, 10),
        "orientation_error_count": (2, 5),
        "memory_error_count": (5, 10),
        "topic_drift_score": (0.62, 0.88),
        "coherence_score": (0.15, 0.38),
        "speech_rate_target": "very_slow",
        "pause_events": (6, 12),
        "long_pause_events": (2, 5),
        "sigh_events": (1, 3),
        "cough_events": (0, 1),
        "breath_events": (1, 4),
    },
}

PICTURE_DETAIL_KEYWORDS = [
    "阿公", "沙發", "相簿", "老花眼鏡", "眼鏡", "茶", "小孩", "囝仔", "積木",
    "媽媽", "阿母", "窗", "電話", "貓", "茶几", "茶杯", "時鐘", "照片", "下午"
]


# ============================================================
# Utility functions
# ============================================================

def weighted_choice(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return random.choices(items, weights=[x.get("weight", 1) for x in items], k=1)[0]


def parse_cdr_counts(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for part in text.split(","):
        key, value = part.split(":", 1)
        key = key.strip()
        if key not in CDR_LABELS:
            raise ValueError(f"Invalid CDR level: {key}")
        out[key] = int(value.strip())
    return out


def make_split_list(total: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> List[str]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    train_n = int(total * train_ratio)
    val_n = int(total * val_ratio)
    test_n = total - train_n - val_n
    splits = ["train"] * train_n + ["val"] * val_n + ["test"] * test_n
    rng = random.Random(seed)
    rng.shuffle(splits)
    return splits


def cdr_to_filename_part(cdr: str) -> str:
    return cdr.replace(".", "_")


def cdr_to_json_value(cdr: str):
    return 0.5 if cdr == "0.5" else int(cdr)


def chinese_char_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Common Simplified -> Traditional cleanup. Not a full converter, but catches frequent leaks.
    replacements = {
        "这": "這", "个": "個", "们": "們", "说": "說", "来": "來", "为": "為",
        "会": "會", "过": "過", "后": "後", "东": "東", "车": "車", "门": "門",
        "买": "買", "卖": "賣", "饭": "飯", "医": "醫", "药": "藥", "头": "頭",
        "发": "發", "没": "沒", "听": "聽", "话": "話", "点": "點", "边": "邊",
        "里": "裡", "价": "價", "还": "還", "记": "記", "对": "對", "应": "應",
        "该": "該", "时": "時", "间": "間", "现": "現", "处": "處", "给": "給",
        "觉": "覺", "蓝": "藍", "绿": "綠", "红": "紅", "黄": "黃", "儿": "兒",
        "妈": "媽", "爷": "爺", "岁": "歲", "声": "聲", "气": "氣", "请": "請",
        "问": "問", "吗": "嗎", "号": "號", "钟": "鐘", "刚": "剛", "写": "寫",
        "层": "層", "样": "樣", "种": "種", "数": "數", "词": "詞", "实": "實",
        "脑": "腦", "糊涂": "糊塗", "样子": "樣子",
        "（停頓）": "[停頓]", "(停頓)": "[停頓]",
        "（長停頓）": "[長停頓]", "(長停頓)": "[長停頓]",
        "（嘆氣）": "[嘆氣]", "(嘆氣)": "[嘆氣]",
        "（咳嗽）": "[咳嗽]", "(咳嗽)": "[咳嗽]",
        "（吸氣）": "[吸氣]", "(吸氣)": "[吸氣]",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = text.replace("...", "……")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_event_text_for_tts(text: str) -> str:
    text = clean_text(text)
    # Remove labels and markers so BreezyVoice does not speak them.
    for item in ["訪談者：", "患者：", "訪談者:", "患者:", "[停頓]", "[長停頓]", "[嘆氣]", "[咳嗽]", "[吸氣]"]:
        text = text.replace(item, "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ，。\n\t")
    return text


def remove_event_markers_for_spoken(text: str) -> str:
    text = clean_text(text)
    for marker in ["[停頓]", "[長停頓]", "[嘆氣]", "[咳嗽]", "[吸氣]"]:
        text = text.replace(marker, "")
    return clean_text(text)


def count_marker(text: str, marker: str) -> int:
    return text.count(marker)


def picture_detail_count(text: str) -> int:
    return sum(1 for kw in PICTURE_DETAIL_KEYWORDS if kw in text)


# ============================================================
# Prompt building: Qwen only generates transcript content
# ============================================================

def get_cdr_rules(cdr: str) -> str:
    if cdr == "0":
        return (
            "CDR 0 normal: coherent, accurate, no dementia-like forgetting. "
            "No orientation error. Do not use 記不得/忘記 unless truly normal and rare. "
            "Use 0-2 [停頓], no [長停頓], no [嘆氣]/[咳嗽], at most one [吸氣]."
        )
    if cdr == "0.5":
        return (
            "CDR 0.5 very mild: mostly coherent. Add only subtle hesitation and one small uncertainty. "
            "Use 那個/我想一下/好像 lightly. Avoid multiple serious forgetting events. "
            "Use 1-3 [停頓], at most one [長停頓]."
        )
    if cdr == "1":
        return (
            "CDR 1 mild: clear word-finding difficulty, mild repetition, mild memory uncertainty, "
            "but still understandable and mostly on topic. Can have one orientation or memory error. "
            "Use 2-5 pauses total, not too fragmented."
        )
    if cdr == "2":
        return (
            "CDR 2 moderate: noticeable memory problems, more repetition, fragmented recall, needs support, "
            "can confuse time/place/events. Still keep enough meaning to understand the answer. "
            "Use 4-8 pauses and 1-3 long pauses."
        )
    if cdr == "3":
        return (
            "CDR 3 severe: short, fragmented, incomplete, poor recall, frequent word-finding failure, "
            "orientation confusion, repeated simple phrases. Do not make a complete organized description. "
            "Use 6-12 pauses, long pauses, and broken phrases."
        )
    raise ValueError(f"Unknown CDR: {cdr}")


def get_picture_severity_rule(cdr: str) -> str:
    if cdr == "0":
        return "Picture rule: describe 8-12 important details accurately and coherently."
    if cdr == "0.5":
        return "Picture rule: describe 7-10 details; one mild hesitation or uncertainty is okay."
    if cdr == "1":
        return "Picture rule: describe 5-8 details; miss small details; use some hesitation and word-finding."
    if cdr == "2":
        return "Picture rule: describe only 3-6 details; confuse or miss one action; sentences partly fragmented."
    if cdr == "3":
        return "Picture rule: describe only 1-4 details; very fragmented; do NOT list the full scene."
    return ""


def build_content_prompt(
    cdr: str,
    cdr_label: str,
    scenario: str,
    task_type: str,
    interaction_type: str,
) -> str:
    cdr_rules = get_cdr_rules(cdr)

    if task_type == "picture_description":
        task_instruction = f"""
Task: one-person picture description.
Use this fixed picture reference:
{PICTURE_REFERENCE['description']}
{get_picture_severity_rule(cdr)}
"""
    elif task_type == "daily_life_description":
        task_instruction = f"""
Task: one-person daily-life description.
Scenario: {scenario}.
The patient describes a daily-life situation such as morning routine, market, medicine, family, home, or clinic.
"""
    elif task_type == "orientation_conversation":
        task_instruction = f"""
Task: two-person orientation conversation.
Scenario: {scenario}.
The interviewer asks short questions about date/day, place, reason for visit, or recent event.
"""
    elif task_type == "memory_recall_conversation":
        task_instruction = f"""
Task: two-person memory recall conversation.
Scenario: {scenario}.
The interviewer asks about breakfast, recent family event, medicine, or something to remember.
"""
    elif task_type == "structured_cognitive_interview":
        task_instruction = f"""
Task: two-person structured cognitive interview.
Scenario: {scenario}.
Use a short cognitive-style interview with orientation, memory recall, and daily-life questions. Do not copy real MMSE text.
"""
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    if interaction_type == "one_person_description":
        format_rule = """
Return ONLY this minimal JSON object:
{
  "marked_transcript": "patient speech only with markers",
  "feature_notes": "brief note in Chinese or English"
}
Do not include speaker labels in marked_transcript.
"""
    else:
        format_rule = """
Return ONLY this minimal JSON object:
{
  "marked_transcript": "訪談者：short question\n患者：patient answer with markers\n訪談者：short question\n患者：patient answer with markers",
  "feature_notes": "brief note in Chinese or English"
}
Use speaker labels only in marked_transcript lines: 訪談者： and 患者：.
Do not put labels inside the patient's sentence after the line starts.
Keep interviewer turns short and normal.
Patient speech should contain the dementia markers.
"""

    prompt = f"""
You generate Taiwanese Mandarin transcript content for a dementia speech simulation dataset.

Important language rules:
- Use Traditional Chinese only.
- Use Taiwanese Mandarin style, not Mainland Mandarin.
- Avoid full Taigi; small Taiwan-style words are okay: 囝仔, 阿母, 菜市場, 診所, 拿藥.
- Natural oral phrases are okay: 嗯, 欸, 啊, 那個, 我想一下, 好像, 應該是, 記不得.
- Do not overuse symptoms. Match the assigned CDR level.
- Do not mention AI, dataset, Qwen, synthetic, or prompt.
- Output only valid JSON. No markdown.

Assigned severity:
- CDR level: {cdr}
- CDR label: {cdr_label}

CDR rules:
{cdr_rules}

{task_instruction}

Marker rules:
- Use [停頓], [長停頓], [嘆氣], [咳嗽], [吸氣] only when appropriate.
- Do not insert too many markers for low CDR.
- Markers must appear only inside marked_transcript.

{format_rule}
"""
    return prompt.strip()


# ============================================================
# Ollama/Qwen call and parsing
# ============================================================

def call_ollama(prompt: str, model: str, ollama_host: str, temperature: float, top_p: float, timeout_sec: int) -> str:
    url = f"{ollama_host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "top_p": top_p},
    }
    response = requests.post(url, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    return response.json().get("response", "")


def extract_json_from_model_output(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Qwen response")
    return json.loads(match.group(0))


# ============================================================
# Python converts marked_transcript -> event_script
# ============================================================

MARKER_TO_EVENT = {
    "[停頓]": ("pause", 600, 900),
    "[長停頓]": ("pause", 1100, 1700),
    "[嘆氣]": ("sigh", 800, 1400),
    "[咳嗽]": ("cough", 350, 800),
    "[吸氣]": ("breath", 350, 800),
}
MARKER_PATTERN = re.compile(r"(\[停頓\]|\[長停頓\]|\[嘆氣\]|\[咳嗽\]|\[吸氣\])")


def clamp_marker_counts(marked: str, cdr: str) -> str:
    """Reduce excessive non-speech markers according to CDR ranges."""
    ranges = CDR_FEATURE_RANGES[cdr]
    limits = {
        "[停頓]": ranges["pause_events"][1],
        "[長停頓]": ranges["long_pause_events"][1],
        "[嘆氣]": ranges["sigh_events"][1],
        "[咳嗽]": ranges["cough_events"][1],
        "[吸氣]": ranges["breath_events"][1],
    }
    for marker, limit in limits.items():
        seen = 0
        parts = marked.split(marker)
        if len(parts) == 1:
            continue
        rebuilt = [parts[0]]
        for part in parts[1:]:
            seen += 1
            if seen <= limit:
                rebuilt.append(marker)
            else:
                # For excessive markers, keep a short punctuation-like hesitation instead of event.
                rebuilt.append("……")
            rebuilt.append(part)
        marked = "".join(rebuilt)
    return marked


def parse_marked_transcript_to_events(marked: str, interaction_type: str, rng: random.Random) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    current_speaker = "patient"

    if interaction_type == "two_person_conversation":
        lines = [ln.strip() for ln in marked.splitlines() if ln.strip()]
    else:
        lines = [marked.strip()]

    for line in lines:
        if not line:
            continue

        if line.startswith("訪談者："):
            current_speaker = "interviewer"
            line = line.replace("訪談者：", "", 1).strip()
        elif line.startswith("患者："):
            current_speaker = "patient"
            line = line.replace("患者：", "", 1).strip()
        elif interaction_type == "one_person_description":
            current_speaker = "patient"

        parts = MARKER_PATTERN.split(line)
        buffer = ""
        for part in parts:
            if not part:
                continue
            if part in MARKER_TO_EVENT:
                # Flush previous speech before non-speech event.
                clean_buf = clean_event_text_for_tts(buffer)
                if clean_buf:
                    events.append({"type": "speech", "speaker": current_speaker, "text": clean_buf})
                buffer = ""

                event_type, lo, hi = MARKER_TO_EVENT[part]
                events.append({"type": event_type, "duration_ms": rng.randint(lo, hi)})
            else:
                buffer += part

        clean_buf = clean_event_text_for_tts(buffer)
        if clean_buf:
            events.append({"type": "speech", "speaker": current_speaker, "text": clean_buf})

    # Remove empty speech and merge consecutive speech by same speaker.
    cleaned: List[Dict[str, Any]] = []
    for ev in events:
        if ev.get("type") == "speech":
            text = clean_event_text_for_tts(ev.get("text", ""))
            if not text:
                continue
            ev = {"type": "speech", "speaker": ev.get("speaker", "patient"), "text": text}
        cleaned.append(ev)

    merged: List[Dict[str, Any]] = []
    for ev in cleaned:
        if (
            ev.get("type") == "speech"
            and merged
            and merged[-1].get("type") == "speech"
            and merged[-1].get("speaker") == ev.get("speaker")
        ):
            merged[-1]["text"] = clean_event_text_for_tts(merged[-1]["text"] + "。" + ev["text"])
        else:
            merged.append(ev)

    return merged


def build_spoken_transcript_from_marked(marked: str) -> str:
    spoken = remove_event_markers_for_spoken(marked)
    # Keep speaker labels in spoken_transcript for two-person readability, but no markers.
    return spoken


def repair_event_stats(sample: Dict[str, Any]) -> None:
    stats = {
        "speech_chunk_count": 0,
        "pause_event_count": 0,
        "sigh_event_count": 0,
        "cough_event_count": 0,
        "breath_event_count": 0,
        "total_pause_ms": 0,
        "chinese_char_count": chinese_char_count(sample.get("spoken_transcript", "")),
    }
    for ev in sample.get("event_script", []):
        t = ev.get("type")
        if t == "speech":
            stats["speech_chunk_count"] += 1
        elif t == "pause":
            stats["pause_event_count"] += 1
            stats["total_pause_ms"] += int(ev.get("duration_ms", 0) or 0)
        elif t == "sigh":
            stats["sigh_event_count"] += 1
            stats["total_pause_ms"] += int(ev.get("duration_ms", 0) or 0)
        elif t == "cough":
            stats["cough_event_count"] += 1
            stats["total_pause_ms"] += int(ev.get("duration_ms", 0) or 0)
        elif t == "breath":
            stats["breath_event_count"] += 1
            stats["total_pause_ms"] += int(ev.get("duration_ms", 0) or 0)
    sample["event_stats"] = stats


# ============================================================
# Feature extraction / validation
# ============================================================

def clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def clamp_float(value: float, lo: float, hi: float) -> float:
    return round(max(lo, min(hi, float(value))), 2)


def estimate_features(marked: str, event_stats: Dict[str, Any], cdr: str) -> Dict[str, Any]:
    text = clean_text(marked)
    ranges = CDR_FEATURE_RANGES[cdr]

    raw_hes = len(re.findall(r"嗯|呃|欸|那個|我想一下|想一下|記不得|忘記|好像", text))
    raw_rep = len(re.findall(r"(\w{1,3})[、，……\s]*\1", text))
    raw_wf = len(re.findall(r"那個|叫什麼|不知道|想不起來|忘記怎麼|記不得", text))
    raw_ori = len(re.findall(r"幾號|星期|在哪裡|早上還是晚上|禮拜|時間", text)) if cdr in ["2", "3"] else len(re.findall(r"不確定|忘記今天|幾號", text))
    raw_mem = len(re.findall(r"忘記|記不得|不記得|想不起來|記不太清楚", text))

    features = {
        "hesitation_count": clamp_int(raw_hes, *ranges["hesitation_count"]),
        "repetition_count": clamp_int(raw_rep, *ranges["repetition_count"]),
        "word_finding_count": clamp_int(raw_wf, *ranges["word_finding_count"]),
        "orientation_error_count": clamp_int(raw_ori, *ranges["orientation_error_count"]),
        "memory_error_count": clamp_int(raw_mem, *ranges["memory_error_count"]),
        "topic_drift_score": clamp_float((ranges["topic_drift_score"][0] + ranges["topic_drift_score"][1]) / 2, *ranges["topic_drift_score"]),
        "coherence_score": clamp_float((ranges["coherence_score"][0] + ranges["coherence_score"][1]) / 2, *ranges["coherence_score"]),
        "speech_rate_target": ranges["speech_rate_target"],
        "pause_event_count": event_stats.get("pause_event_count", 0),
    }
    return features


def validate_sample_quality(sample: Dict[str, Any], cdr: str, task_type: str, interaction_type: str) -> Tuple[bool, str]:
    spoken = sample.get("spoken_transcript", "")
    marked = sample.get("marked_transcript", "")
    events = sample.get("event_script", [])

    if len(spoken.strip()) < 20:
        return False, "spoken_transcript too short"

    # No markers/labels inside event_script speech text.
    for ev in events:
        if ev.get("type") == "speech":
            t = ev.get("text", "")
            if any(x in t for x in ["[停頓]", "[長停頓]", "[嘆氣]", "[咳嗽]", "[吸氣]", "訪談者：", "患者："]):
                return False, "dirty speech event text"

    if interaction_type == "two_person_conversation":
        has_i = any(ev.get("type") == "speech" and ev.get("speaker") == "interviewer" for ev in events)
        has_p = any(ev.get("type") == "speech" and ev.get("speaker") == "patient" for ev in events)
        if not (has_i and has_p):
            return False, "two-person conversation missing interviewer or patient"

    # Low CDR must not contain too many strong impairment words.
    if cdr == "0":
        if any(x in spoken for x in ["記不得", "忘記", "不記得", "想不起來"]):
            return False, "CDR 0 contains strong forgetting"
        if sample["event_stats"]["pause_event_count"] > 2:
            return False, "CDR 0 has too many pauses"

    if cdr == "0.5":
        if len(re.findall(r"記不得|忘記|不記得|想不起來", spoken)) > 2:
            return False, "CDR 0.5 too impaired"

    # Picture task detail control.
    if task_type == "picture_description":
        details = picture_detail_count(spoken)
        if cdr == "0" and details < 7:
            return False, f"CDR 0 picture too few details ({details})"
        if cdr == "0.5" and not (5 <= details <= 11):
            return False, f"CDR 0.5 picture bad detail count ({details})"
        if cdr == "1" and not (4 <= details <= 9):
            return False, f"CDR 1 picture bad detail count ({details})"
        if cdr == "2" and not (2 <= details <= 7):
            return False, f"CDR 2 picture bad detail count ({details})"
        if cdr == "3" and details > 5:
            return False, f"CDR 3 picture too complete ({details})"

    return True, "ok"


# ============================================================
# Full sample creation
# ============================================================

def make_final_sample(
    qwen_minimal: Dict[str, Any],
    sample_id: str,
    cdr: str,
    cdr_label: str,
    scenario: str,
    task_type: str,
    interaction_type: str,
    patient_voice: Dict[str, str],
    split: str,
    acoustic_condition: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    marked = clean_text(qwen_minimal.get("marked_transcript", ""))
    marked = clamp_marker_counts(marked, cdr)
    spoken = build_spoken_transcript_from_marked(marked)
    events = parse_marked_transcript_to_events(marked, interaction_type, rng)

    sample = {
        "sample_id": sample_id,
        "cdr_level": cdr_to_json_value(cdr),
        "cdr_label": cdr_label,
        "scenario": scenario,
        "task_type": task_type,
        "interaction_type": interaction_type,
        "picture_reference": PICTURE_REFERENCE if task_type == "picture_description" else None,
        "speaker": {
            "speaker_id": patient_voice["voice_id"],
            "voice_id": patient_voice["voice_id"],
            "speaker_group": patient_voice["speaker_group"],
            "gender": patient_voice["gender"],
            "split": split,
        },
        "interviewer": {
            "speaker_id": INTERVIEWER_VOICE["voice_id"],
            "voice_id": INTERVIEWER_VOICE["voice_id"],
            "speaker_group": INTERVIEWER_VOICE["speaker_group"],
        },
        "split": split,
        "spoken_transcript": spoken,
        "marked_transcript": marked,
        "event_script": events,
        "event_stats": {},
        "impairment_features": {},
        "acoustic_condition": acoustic_condition["name"],
        "acoustic_condition_info": {
            "weight": acoustic_condition["weight"],
            "description": acoustic_condition["description"],
        },
        "generation_notes": clean_text(qwen_minimal.get("feature_notes", "")),
    }
    repair_event_stats(sample)
    sample["impairment_features"] = estimate_features(marked, sample["event_stats"], cdr)
    return sample


def generate_one_sample(
    sample_id: str,
    cdr: str,
    cdr_label: str,
    scenario: str,
    task_type: str,
    interaction_type: str,
    patient_voice: Dict[str, str],
    split: str,
    acoustic_condition: Dict[str, Any],
    args: argparse.Namespace,
    rng: random.Random,
) -> Dict[str, Any]:
    prompt = build_content_prompt(cdr, cdr_label, scenario, task_type, interaction_type)
    last_error: Optional[Exception] = None

    for attempt in range(1, args.max_retries + 1):
        try:
            raw = call_ollama(prompt, args.model, args.ollama_host, args.temperature, args.top_p, args.timeout_sec)
            minimal = extract_json_from_model_output(raw)
            sample = make_final_sample(
                minimal, sample_id, cdr, cdr_label, scenario, task_type, interaction_type,
                patient_voice, split, acoustic_condition, rng
            )
            ok, message = validate_sample_quality(sample, cdr, task_type, interaction_type)
            if not ok:
                raise ValueError(message)
            return sample
        except Exception as exc:
            last_error = exc
            print(f"[WARN] {sample_id} attempt {attempt}/{args.max_retries} failed: {exc}", flush=True)
            time.sleep(args.retry_sleep_sec)

    raise RuntimeError(f"Failed to generate {sample_id}: {last_error}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3.5:35b")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434")
    parser.add_argument("--output-dir", type=str, default="dataset/tw_mandarin_cdr_json_v3")
    parser.add_argument("--cdr-counts", type=str, default="0:350,0.5:300,1:200,2:100,3:50")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-sec", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cdr_counts = parse_cdr_counts(args.cdr_counts)
    total = sum(cdr_counts.values())
    splits = make_split_list(total, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    print("=" * 80, flush=True)
    print("Python-controlled Taiwanese Mandarin CDR JSON Generator", flush=True)
    print("Qwen generates only marked_transcript; Python builds final JSON.", flush=True)
    print("=" * 80, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"CDR counts: {cdr_counts}", flush=True)
    print(f"Total samples: {total}", flush=True)
    print(f"Voices: {[v['voice_id'] for v in VOICE_POOL]}", flush=True)
    print("=" * 80, flush=True)

    if args.dry_run:
        print("Dry run only. No files generated.", flush=True)
        return

    manifest = []
    global_index = 0

    for cdr, count in cdr_counts.items():
        cdr_label = CDR_LABELS[cdr]
        for _ in range(count):
            global_index += 1
            split = splits[global_index - 1]
            patient_voice = rng.choice(VOICE_POOL)
            task_info = weighted_choice(TASK_TYPES)
            task_type = task_info["task_type"]
            interaction_type = task_info["interaction_type"]
            acoustic_condition = weighted_choice(ACOUSTIC_CONDITIONS)
            scenario = "picture_description" if task_type == "picture_description" else rng.choice(SCENARIOS)

            safe_cdr = cdr_to_filename_part(cdr)
            sample_id = f"cdr_{safe_cdr}_{patient_voice['voice_id']}_{scenario}_{global_index:04d}"

            out_dir = output_dir / f"cdr_{safe_cdr}" / interaction_type
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"{sample_id}.json"

            if output_path.exists():
                print(f"[SKIP] {global_index}/{total} {sample_id}", flush=True)
                continue

            start = time.time()
            try:
                sample = generate_one_sample(
                    sample_id, cdr, cdr_label, scenario, task_type, interaction_type,
                    patient_voice, split, acoustic_condition, args, rng
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(sample, f, ensure_ascii=False, indent=2)

                elapsed = time.time() - start
                stats = sample["event_stats"]
                print(
                    f"[OK] {global_index}/{total} | {sample_id} | cdr={cdr} | split={split} | "
                    f"voice={patient_voice['voice_id']} | task={task_type} | "
                    f"{elapsed:.2f}s | chars={stats['chinese_char_count']} | pauses={stats['pause_event_count']}",
                    flush=True,
                )
                manifest.append({
                    "sample_id": sample_id,
                    "path": str(output_path),
                    "cdr_level": cdr_to_json_value(cdr),
                    "cdr_label": cdr_label,
                    "scenario": scenario,
                    "task_type": task_type,
                    "interaction_type": interaction_type,
                    "voice_id": patient_voice["voice_id"],
                    "speaker_group": patient_voice["speaker_group"],
                    "split": split,
                    "acoustic_condition": acoustic_condition["name"],
                })
            except Exception as exc:
                print(f"[ERROR] {global_index}/{total} | {sample_id} | {exc}", flush=True)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 80, flush=True)
    print("Generation finished.", flush=True)
    print(f"Manifest saved to: {manifest_path}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
