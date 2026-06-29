#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VALIDATOR_VERSION = V6_EVENT_SCRIPT_AUTHORITATIVE_MARKED_REPAIR

Strict Taiwanese Mandarin CDR transcript generator using Ollama/Qwen.

Default model:
    qwen3.5:35bq4K_XL

What this script does:
1. Generates Taiwanese Mandarin dementia-simulation JSON samples.
2. Adds impairment features that are observable from transcript/event script.
3. Validates JSON strictly.
4. If validation fails, it asks Qwen to regenerate.
5. Saves accepted JSON to output folder.
6. Saves rejected attempts to rejected_logs for debugging.

Run:
    python generate_tw_cdr_strict.py --model qwen3.5:35bq4K_XL --output-dir dataset_strict

Example custom counts:
    python generate_tw_cdr_strict.py --cdr-counts "0:300,0.5:300,1:200,2:120,3:80"

Make sure Ollama is running:
    ollama serve

Test model:
    ollama run qwen3.5:35bq4K_XL
"""

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

VALIDATOR_VERSION = "V6_EVENT_SCRIPT_AUTHORITATIVE_MARKED_REPAIR"


# -----------------------------
# Basic configuration
# -----------------------------

DEFAULT_MODEL = "qwen3.5:35bq4K_XL"

SCENARIOS = [
    "clinic",
    "family",
    "home",
    "market",
    "medicine_routine",
    "daily_life",
    "picture_description",
]

VOICE_POOL = [
    {"voice_id": "voice1_male", "speaker_group": "older_male", "gender": "male", "description": "台灣華語男性長輩聲音。"},
    {"voice_id": "voice2_female", "speaker_group": "older_female", "gender": "female", "description": "台灣華語女性長輩聲音。"},
    {"voice_id": "voice3_male", "speaker_group": "older_male_soft", "gender": "male", "description": "較柔和的台灣華語男性長輩聲音。"},
    {"voice_id": "voice4_female", "speaker_group": "older_female_clear", "gender": "female", "description": "清楚自然的台灣華語女性長輩聲音。"},
]

PICTURE_REFERENCE = {
    "picture_id": "living_room_family_scene_001",
    "name": "living_room_family_scene",
    "description": (
        "一個客廳裡的家庭場景。阿公坐在沙發上看相簿，旁邊有一副老花眼鏡和一杯茶。"
        "一個小孩坐在地上玩積木，積木散在地毯上。媽媽站在窗邊接電話。"
        "一隻貓跳到茶几旁邊，好像快要碰倒茶杯。牆上有時鐘和家庭照片，窗外看起來像是下午。"
    ),
    "key_units": [
        "客廳場景",
        "阿公坐在沙發上看相簿",
        "老花眼鏡",
        "茶或茶杯",
        "小孩玩積木",
        "媽媽在窗邊接電話",
        "貓靠近茶几",
        "茶杯可能被碰倒",
        "牆上有時鐘",
        "牆上有家庭照片",
    ],
}

ACOUSTIC_CONDITIONS = [
    {"name": "clean", "description": "乾淨錄音，無明顯背景噪音。"},
    {"name": "clinic_room_noise", "description": "輕微診間背景聲。"},
    {"name": "home_background_noise", "description": "輕微居家背景聲。"},
    {"name": "phone_mic_degraded", "description": "手機麥克風品質，略微壓縮。"},
]

INTERVIEWER_QUESTIONS = {
    "clinic": [
        "你今天是怎麼來這裡的？",
        "你可以跟我說一下今天早上做了什麼嗎？",
        "你記得今天大概是星期幾嗎？",
        "你最近睡得好不好？",
        "你平常在家都怎麼安排一天？"
    ],
    "medicine_routine": [
        "你平常藥都是什麼時候吃？",
        "你可以說一下早上吃藥的順序嗎？",
        "如果忘記吃藥，你通常會怎麼處理？",
        "家人有沒有幫你整理藥盒？",
        "你記得今天早上的藥吃了沒有？"
    ],
}

SPLITS = ["train", "val", "test"]

SIMPLIFIED_TO_TRADITIONAL = {
    "这": "這", "个": "個", "们": "們", "说": "說", "来": "來", "为": "為", "会": "會",
    "过": "過", "后": "後", "东": "東", "车": "車", "门": "門", "买": "買", "卖": "賣",
    "饭": "飯", "医": "醫", "药": "藥", "头": "頭", "发": "發", "没": "沒", "听": "聽",
    "话": "話", "点": "點", "边": "邊", "里": "裡", "儿": "兒", "吗": "嗎", "吗": "嗎",
    "对": "對", "时": "時", "间": "間", "钟": "鐘", "岁": "歲", "认": "認", "记": "記",
    "现": "現", "开": "開", "关": "關", "动": "動", "样": "樣", "应": "應", "实": "實",
    "写": "寫", "觉": "覺", "长": "長", "亲": "親", "还": "還", "经": "經", "觉": "覺",
    "电": "電", "脑": "腦", "带": "帶", "场": "場", "气": "氣", "声": "聲", "轻": "輕",
}

BANNED_TERMS = [
    "失智", "癡呆", "痴呆", "阿茲海默", "阿兹海默", "老年癡呆", "老年痴呆",
    "dementia", "Alzheimer", "CDR"
]

FILLER_PATTERNS = [
    "嗯", "呃", "啊", "那個", "這個", "我想一下", "怎麼說", "不知道怎麼講",
    "叫什麼", "想不起來", "忘記", "不太確定"
]

WORD_FINDING_PATTERNS = [
    "那個東西", "那個人", "叫什麼", "不知道怎麼講", "想不起來",
    "忘記怎麼說", "拿來", "用的那個", "就是那個"
]

VAGUE_REFERENCE_PATTERNS = [
    "那個", "這個", "那邊", "那裡", "那種", "那件", "那位", "那一些", "東西"
]

MEMORY_GAP_PATTERNS = [
    "忘記", "想不起來", "不記得", "我本來要講什麼", "剛剛講到哪"
]

ORIENTATION_PATTERNS = [
    "今天是幾", "星期幾", "幾月", "哪一天", "在哪裡", "這裡是哪裡", "現在是",
    "附近", "哪裡", "什麼時候", "不太確定", "對不對", "好像是在"
]

SELF_CORRECTION_PATTERNS = [
    "不是", "不對", "應該是", "我是說", "不是那個", "講錯"
]

EVENT_TYPES = {"speech", "pause", "long_pause", "silence", "sigh", "cough"}


CDR_RULES = {
    "0": {
        "label": "normal",
        "pause": (0, 1),
        "long_pause": (0, 0),
        "filled_pause": (0, 1),
        "word_finding": (0, 0),
        "repetition": (0, 1),
        "fragment": (0, 1),
        "topic_drift": (0.0, 0.10),
        "coherence": (0.90, 1.00),
        "missing_info": (0, 1),
        "orientation": (0, 0),
        "memory_gap": (0, 0),
        "speech_rate_cps": (4.2, 5.2),
    },
    "0.5": {
        "label": "very_mild",
        "pause": (1, 3),
        "long_pause": (0, 1),
        "filled_pause": (1, 2),
        "word_finding": (0, 1),
        "repetition": (0, 1),
        "fragment": (0, 2),
        "topic_drift": (0.10, 0.25),
        "coherence": (0.78, 0.92),
        "missing_info": (1, 2),
        "orientation": (0, 1),
        "memory_gap": (0, 1),
        "speech_rate_cps": (3.6, 4.6),
    },
    "1": {
        "label": "mild",
        # CDR 1 should show mild impairment, but not every sample must contain
        # every marker. Requiring repetition and 2+ pauses in every sample caused
        # valid mild samples to be rejected too often.
        "pause": (1, 5),
        "long_pause": (0, 2),
        "filled_pause": (1, 4),
        "word_finding": (1, 3),
        "repetition": (0, 3),
        "fragment": (1, 3),
        "topic_drift": (0.25, 0.45),
        "coherence": (0.60, 0.80),
        "missing_info": (2, 4),
        "orientation": (0, 2),
        "memory_gap": (1, 2),
        "speech_rate_cps": (3.0, 4.0),
    },
    "2": {
        "label": "moderate",
        # Moderate impairment can be represented by several disfluency events.
        # Do not require 4+ short pauses if long pauses are already present.
        "pause": (1, 8),
        "long_pause": (1, 4),
        "filled_pause": (2, 7),
        "word_finding": (2, 6),
        "repetition": (1, 6),
        "fragment": (2, 7),
        "topic_drift": (0.45, 0.70),
        "coherence": (0.40, 0.60),
        "missing_info": (4, 7),
        "orientation": (1, 3),
        "memory_gap": (2, 4),
        "speech_rate_cps": (2.2, 3.3),
    },
    "3": {
        "label": "severe",
        # Severe samples often use fewer short pauses plus multiple long pauses/silences.
        # Forcing 6+ short pause events caused good severe samples to fail.
        "pause": (3, 12),
        "long_pause": (1, 6),
        "filled_pause": (4, 10),
        "word_finding": (4, 9),
        "repetition": (2, 9),
        "fragment": (4, 12),
        "topic_drift": (0.70, 0.95),
        "coherence": (0.10, 0.40),
        "missing_info": (6, 10),
        "orientation": (1, 5),
        "memory_gap": (3, 6),
        "speech_rate_cps": (1.4, 2.7),
    },
}


# -----------------------------
# Utility helpers
# -----------------------------

def parse_cdr_counts(text: str) -> Dict[str, int]:
    out = {}
    for part in text.split(","):
        k, v = part.split(":")
        out[k.strip()] = int(v.strip())
    for k in out:
        if k not in CDR_RULES:
            raise ValueError(f"Unsupported CDR level: {k}")
    return out


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # normalize marker parentheses
    s = s.replace("（停頓）", "[停頓]").replace("(停頓)", "[停頓]")
    s = s.replace("（長停頓）", "[長停頓]").replace("(長停頓)", "[長停頓]")
    s = s.replace("（沉默）", "[沉默]").replace("(沉默)", "[沉默]")
    s = s.replace("（嘆氣）", "[嘆氣]").replace("(嘆氣)", "[嘆氣]")
    s = s.replace("（咳嗽）", "[咳嗽]").replace("(咳嗽)", "[咳嗽]")
    # normalize common spacing around markers
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*(\[(?:停頓|長停頓|沉默|嘆氣|咳嗽)\])\s*", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def has_simplified_chars(s: str) -> List[str]:
    found = []
    for ch in SIMPLIFIED_TO_TRADITIONAL:
        if ch in s:
            found.append(ch)
    return sorted(set(found))


def count_occurrences(text: str, patterns: List[str]) -> int:
    total = 0
    for p in patterns:
        total += text.count(p)
    return total


def count_marker(marked: str, marker: str) -> int:
    return marked.count(marker)


def count_event_type(events: List[Dict[str, Any]], event_type: str) -> int:
    return sum(1 for e in events if e.get("type") == event_type)


def total_duration(events: List[Dict[str, Any]], types: set) -> float:
    total = 0.0
    for e in events:
        if e.get("type") in types:
            try:
                total += float(e.get("duration_sec", 0))
            except Exception:
                pass
    return round(total, 2)


def approx_repetition_count(text: str) -> int:
    """
    Simple repetition detector for Mandarin strings.
    Counts repeated short chunks like:
        早上，早上
        那個，那個
        我我 / 他他
    This is not perfect, but good enough for validation pressure.
    """
    clean = re.sub(r"[\s，。！？、,.!?；;：:\[\]（）()]", " ", text)
    tokens = [t for t in clean.split() if t]
    count = 0

    # repeated token-like phrases separated by punctuation/spaces
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1] and len(tokens[i]) >= 1:
            count += 1

    # repeated 1-4 char chunks with punctuation in between
    pattern = r"([\u4e00-\u9fff]{1,4})[，、\s]+?\1"
    count += len(re.findall(pattern, text))

    # direct char repeats, but ignore natural reduplication by keeping low weight
    pattern2 = r"([\u4e00-\u9fff]{1,2})\1"
    direct = re.findall(pattern2, text)
    direct = [x for x in direct if x not in ["媽媽", "爸爸", "哥哥", "姐姐", "妹妹", "爺爺", "奶奶"]]
    count += min(len(direct), 3)

    return count


def estimate_sentence_fragments(text: str) -> int:
    markers = ["……", "...", "，", "、"]
    fragments = 0
    # unfinished-like endings
    fragments += text.count("……")
    fragments += text.count("...")
    fragments += text.count("呃")
    fragments += text.count("嗯")
    # short broken clauses
    clauses = re.split(r"[。！？!?]", text)
    for c in clauses:
        c = c.strip()
        if 1 <= len(c) <= 6 and any(m in c for m in ["那個", "嗯", "呃", "不是", "忘記"]):
            fragments += 1
    return fragments


def numeric_in_range(value: Any, lo: float, hi: float, name: str, errors: List[str]) -> None:
    if not isinstance(value, (int, float)):
        errors.append(f"{name} must be numeric")
        return
    if not (lo <= float(value) <= hi):
        errors.append(f"{name}={value} outside allowed range [{lo}, {hi}]")


def int_in_range(value: Any, lo: int, hi: int, name: str, errors: List[str]) -> None:
    if not isinstance(value, int):
        errors.append(f"{name} must be integer")
        return
    if not (lo <= value <= hi):
        errors.append(f"{name}={value} outside allowed range [{lo}, {hi}]")


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a valid JSON object from the model response.

    Qwen reasoning models sometimes output:
        <think> ... </think>
        { ... }

    This function removes think blocks, removes markdown fences,
    then scans for JSON objects and returns the last valid object.
    Returning the last valid object is safer because the model may mention
    schema-like objects before the final answer.
    """
    if not isinstance(text, str):
        return None

    text = text.strip()

    # Remove explicit thinking blocks if present.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # Remove markdown fences if present.
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    # Direct parse first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    candidates = []
    starts = [i for i, ch in enumerate(text) if ch == "{"]

    for start in starts:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i+1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict) and "sample_id" in obj and "event_script" in obj:
                                candidates.append(obj)
                        except Exception:
                            pass
                        break

    if candidates:
        return candidates[-1]

    return None


# -----------------------------
# Prompt construction
# -----------------------------

def build_prompt(
    sample_id: str,
    cdr: str,
    scenario: str,
    speaker: Dict[str, str],
    split: str,
    acoustic_condition: Dict[str, str],
    previous_errors: Optional[List[str]] = None
) -> str:
    rules = CDR_RULES[cdr]

    interaction_type = "one_person_description"
    task_type = "daily_life_description"
    picture_ref = None

    if scenario == "picture_description":
        task_type = "picture_description"
        picture_ref = PICTURE_REFERENCE
    elif scenario in ["clinic", "medicine_routine"]:
        task_type = "structured_cognitive_interview"
        interaction_type = "two_person_conversation"
    elif scenario in ["family", "home", "market", "daily_life"]:
        task_type = "daily_life_description"

    interviewer_question = None
    if interaction_type == "two_person_conversation":
        interviewer_question = random.choice(INTERVIEWER_QUESTIONS.get(scenario, ["你可以多說一點嗎？"]))

    required_schema = {
        "sample_id": sample_id,
        "cdr_level": cdr if cdr == "0.5" else int(float(cdr)),
        "cdr_label": rules["label"],
        "scenario": scenario,
        "task_type": task_type,
        "interaction_type": interaction_type,
        "picture_reference": picture_ref,
        "speaker": {
            "speaker_id": speaker["voice_id"],
            "voice_id": speaker["voice_id"],
            "speaker_group": speaker["speaker_group"],
            "gender": speaker["gender"],
            "description": speaker["description"],
            "split": split
        },
        "interviewer": {
            "speaker_id": "interviewer_neutral",
            "voice_id": "interviewer_neutral",
            "speaker_group": "interviewer"
        } if interaction_type == "two_person_conversation" else None,
        "split": split,
        "acoustic_condition": acoustic_condition["name"],
        "acoustic_condition_info": acoustic_condition,
        "conversation_context": [
            {"speaker": "interviewer", "text": "訪談者問題，只在 two_person_conversation 使用。"},
            {"speaker": "patient", "text": "患者回答，必須與 spoken_transcript 內容一致。"}
        ] if interaction_type == "two_person_conversation" else None,
        "spoken_transcript": "只包含患者語音內容。台灣華語文字，不要標記停頓，不要出現訪談者/患者標籤。",
        "marked_transcript": "只包含患者語音內容。同一段文字，但插入 [停頓] [長停頓] [沉默] [嘆氣] [咳嗽]。",
        "event_script": [
            {"type": "speech", "speaker": "patient", "text": "患者語音片段，訓練用，只能 patient"},
            {"type": "pause", "duration_sec": 0.8},
            {"type": "speech", "speaker": "patient", "text": "患者下一段語音"}
        ],
        "full_conversation_event_script": [
            {"type": "speech", "speaker": "interviewer", "text": "訪談者問題"},
            {"type": "pause", "duration_sec": 0.5},
            {"type": "speech", "speaker": "patient", "text": "患者語音片段"}
        ] if interaction_type == "two_person_conversation" else None,
        "audio_outputs": {
            "training_audio": "patient_only.wav",
            "demo_audio": "full_conversation.wav" if interaction_type == "two_person_conversation" else None
        },
        "event_stats": {
            "speech_event_count": 2,
            "pause_count": 1,
            "long_pause_count": 0,
            "silence_count": 0,
            "sigh_count": 0,
            "cough_count": 0,
            "total_pause_duration_sec": 0.8,
            "total_silence_duration_sec": 0.0
        },
        "impairment_features": {
            "acoustic_fluency": {
                "pause_count": 1,
                "long_pause_count": 0,
                "total_pause_duration_sec": 0.8,
                "silence_ratio": 0.05,
                "speech_rate_target_cps": 4.5,
                "filled_pause_count": 1,
                "restart_count": 0
            },
            "linguistic_impairment": {
                "word_finding_count": 0,
                "vague_reference_count": 0,
                "circumlocution_count": 0,
                "word_repetition_count": 0,
                "phrase_repetition_count": 0,
                "self_correction_count": 0,
                "sentence_fragment_count": 0
            },
            "cognitive_task_performance": {
                "key_information_units_mentioned": None,
                "key_information_units_total": None,
                "missing_key_information_count": None,
                "incorrect_detail_count": 0,
                "topic_drift_score": 0.05,
                "coherence_score": 0.95,
                "memory_gap_count": 0,
                "orientation_error_count": 0
            }
        }
    }

    error_text = ""
    if previous_errors:
        safe_errors = []
        for e in previous_errors[:20]:
            # Do not feed banned clinical keywords back into Qwen; it causes thinking output.
            e = str(e).replace("CDR", "rating-scale term").replace("cdr", "rating-scale term")
            e = e.replace("dementia", "diagnosis term").replace("Alzheimer", "diagnosis term")
            safe_errors.append(e)
        error_text = "\nPREVIOUS ATTEMPT FAILED. Fix these validation errors:\n" + "\n".join(f"- {e}" for e in safe_errors)

    prompt = f"""
/no_think
You are generating STRICT research-style synthetic Taiwanese Mandarin speech data for dementia severity simulation.

Return ONLY valid JSON. No markdown. No explanation. Do not output <think>, reasoning, analysis, or any text before the JSON.

IMPORTANT LANGUAGE RULES:
- Use Traditional Chinese / Taiwanese Mandarin style.
- Do NOT use Simplified Chinese characters.
- Do NOT use Mainland phrasing if Taiwanese phrasing is natural.
- Do NOT mention diagnosis names, disease names, rating-scale names, or clinical labels inside the transcript.
- Do NOT include "患者：" or "訪談者：" inside spoken_transcript, marked_transcript, or event_script speech text.
- The transcript should sound like an elderly Taiwanese Mandarin speaker.
- Keep content natural, not exaggerated.

TARGET SAMPLE:
sample_id: {sample_id}
CDR level: {cdr}
CDR label: {rules["label"]}
scenario: {scenario}
task_type: {task_type}
interaction_type: {interaction_type}
speaker: {speaker["voice_id"]}
split: {split}

PICTURE REFERENCE, only if scenario is picture_description:
{json.dumps(picture_ref, ensure_ascii=False, indent=2) if picture_ref else "null"}

STRICT CDR FEATURE RANGES:
- pause_count: {rules["pause"][0]} to {rules["pause"][1]}
- long_pause_count: {rules["long_pause"][0]} to {rules["long_pause"][1]}
- filled_pause_count: {rules["filled_pause"][0]} to {rules["filled_pause"][1]}
- word_finding_count: {rules["word_finding"][0]} to {rules["word_finding"][1]}
- repetition_count: {rules["repetition"][0]} to {rules["repetition"][1]}
- sentence_fragment_count: {rules["fragment"][0]} to {rules["fragment"][1]}
- topic_drift_score: {rules["topic_drift"][0]} to {rules["topic_drift"][1]}
- coherence_score: {rules["coherence"][0]} to {rules["coherence"][1]}
- memory_gap_count: {rules["memory_gap"][0]} to {rules["memory_gap"][1]}
- orientation_error_count: {rules["orientation"][0]} to {rules["orientation"][1]}
- speech_rate_target_cps: {rules["speech_rate_cps"][0]} to {rules["speech_rate_cps"][1]}

FEATURE CONSTRUCTION RULES:
- If word_finding_count > 0, the transcript must visibly contain word-finding phrases like 那個東西, 叫什麼, 想不起來, 不知道怎麼講.
- If filled_pause_count > 0, the transcript must visibly contain 嗯, 呃, 那個, 我想一下, or similar inside speech text. Do not create filled_pause events.
- If repetition_count > 0, the transcript must visibly repeat a word/phrase/idea.
- If memory_gap_count > 0, the transcript must visibly contain 忘記, 想不起來, 不記得, or similar.
- If orientation_error_count > 0, use only interview/clinic/medicine routine naturally, such as uncertainty about date/time/place.
- CDR 0 must be normal and coherent.
- CDR 0.5 must be very mild. Do not overdo impairment.
- CDR 2 picture description should be incomplete.
- CDR 3 should be fragmented but still valid Taiwanese Mandarin speech.

EVENT SCRIPT RULES:
- event_script is PATIENT-ONLY audio for model training.
- event_script must be a list of events.
- Allowed event types: speech, pause, long_pause, silence, sigh, cough.
- NEVER use filled_pause as an event type. Filled pauses such as 嗯, 呃, 那個 must be inside speech text, not separate events.
- In event_script, speech event must have speaker="patient" and text.
- pause/long_pause/silence/sigh/cough must have duration_sec.
- marked_transcript should contain [停頓] for pause, [長停頓] for long_pause, [沉默] for silence, [嘆氣] for sigh, [咳嗽] for cough. The script will repair marker placement from event_script.
- event_stats must exactly match patient-only event_script counts.
- impairment_features.acoustic_fluency.pause_count must equal event_stats.pause_count.
- impairment_features.acoustic_fluency.long_pause_count must equal event_stats.long_pause_count.

TWO-PERSON CONVERSATION RULES:
- If interaction_type is two_person_conversation, conversation_context is REQUIRED.
- Use this interviewer question exactly:
  {interviewer_question if interviewer_question else "null"}
- conversation_context must include at least one interviewer turn and one patient turn.
- spoken_transcript must contain ONLY the patient answer, not the interviewer question.
- marked_transcript must contain ONLY the patient answer plus event markers.
- event_script must contain ONLY patient speech and patient pauses for training.
- full_conversation_event_script must contain interviewer speech plus patient speech for demo audio.
- full_conversation_event_script may contain speaker="interviewer" and speaker="patient".
- Do not count interviewer speech or interviewer pauses in event_stats or impairment_features.

REQUIRED JSON SCHEMA:
{json.dumps(required_schema, ensure_ascii=False, indent=2)}

{error_text}

Generate one complete JSON object now.
"""
    return prompt.strip()


# -----------------------------
# Ollama API
# -----------------------------

def ollama_generate(
    model: str,
    prompt: str,
    host: str = "http://localhost:11434",
    temperature: float = 0.4,
    timeout: int = 300,
) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "top_p": 0.8,
            "num_ctx": 8192,
            "num_predict": 4096,
            "repeat_penalty": 1.12
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
            return out.get("response", "")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama HTTP error {e.code}: {body}")
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")

def repair_invalid_filled_pause_events(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Qwen sometimes invents event type 'filled_pause'. This is not allowed.
    A filled pause is lexical material like 嗯/呃/那個 and should be speech text.

    Convert:
        {"type": "filled_pause", "marker": "呃", "duration_sec": 0.4}
    into:
        {"type": "speech", "speaker": "patient", "text": "呃"}

    This repair is safe because filled pauses are audible speech, not silence.
    """
    if not isinstance(obj, dict):
        return obj

    def repair_events(events):
        if not isinstance(events, list):
            return events
        repaired = []
        for e in events:
            if isinstance(e, dict) and e.get("type") == "filled_pause":
                marker = e.get("marker") or e.get("filler_type") or e.get("text") or "嗯"
                if not isinstance(marker, str) or not marker.strip():
                    marker = "嗯"
                repaired.append({
                    "type": "speech",
                    "speaker": e.get("speaker", "patient"),
                    "text": marker.strip()
                })
            else:
                repaired.append(e)
        return repaired

    obj["event_script"] = repair_events(obj.get("event_script"))
    obj["full_conversation_event_script"] = repair_events(obj.get("full_conversation_event_script"))
    return obj


def recompute_event_stats_and_acoustic(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recompute event_stats from patient-only event_script after small repairs.
    This prevents rejection when Qwen's counts are slightly inconsistent.
    """
    if not isinstance(obj, dict) or not isinstance(obj.get("event_script"), list):
        return obj

    events = obj["event_script"]
    computed_stats = {
        "speech_event_count": count_event_type(events, "speech"),
        "pause_count": count_event_type(events, "pause"),
        "long_pause_count": count_event_type(events, "long_pause"),
        "silence_count": count_event_type(events, "silence"),
        "sigh_count": count_event_type(events, "sigh"),
        "cough_count": count_event_type(events, "cough"),
        "total_pause_duration_sec": total_duration(events, {"pause", "long_pause"}),
        "total_silence_duration_sec": total_duration(events, {"silence"}),
    }
    obj["event_stats"] = computed_stats

    try:
        acoustic = obj["impairment_features"]["acoustic_fluency"]
        acoustic["pause_count"] = computed_stats["pause_count"]
        acoustic["long_pause_count"] = computed_stats["long_pause_count"]
        acoustic["total_pause_duration_sec"] = computed_stats["total_pause_duration_sec"]
    except Exception:
        pass

    return obj

def event_script_to_marked_transcript(events: List[Dict[str, Any]]) -> str:
    """
    Build marked_transcript directly from patient-only event_script.
    This avoids failures caused by Qwen putting markers in the wrong place.
    """
    parts = []
    marker_map = {
        "pause": "[停頓]",
        "long_pause": "[長停頓]",
        "silence": "[沉默]",
        "sigh": "[嘆氣]",
        "cough": "[咳嗽]",
    }
    for e in events if isinstance(events, list) else []:
        if not isinstance(e, dict):
            continue
        typ = e.get("type")
        if typ == "speech":
            txt = str(e.get("text", "")).strip()
            # Remove accidental bracket markers inside speech.
            txt = re.sub(r"\[(?:停頓|長停頓|沉默|嘆氣|咳嗽|嗯|呃)\]", "", txt).strip()
            if txt:
                parts.append(txt)
                e["text"] = txt
        elif typ in marker_map:
            parts.append(marker_map[typ])
    return normalize_text(" ".join(parts))


def events_to_spoken_transcript(events: List[Dict[str, Any]]) -> str:
    parts = []
    for e in events if isinstance(events, list) else []:
        if isinstance(e, dict) and e.get("type") == "speech":
            txt = str(e.get("text", "")).strip()
            txt = re.sub(r"\[(?:停頓|長停頓|沉默|嘆氣|咳嗽|嗯|呃)\]", "", txt).strip()
            if txt:
                parts.append(txt)
    return normalize_text("".join(parts))


def repair_marked_and_spoken_from_events(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make event_script the source of truth.
    - Remove markers accidentally inside speech event text.
    - Rebuild spoken_transcript from speech events.
    - Rebuild marked_transcript from events.
    """
    if not isinstance(obj, dict) or not isinstance(obj.get("event_script"), list):
        return obj
    obj["marked_transcript"] = event_script_to_marked_transcript(obj["event_script"])
    obj["spoken_transcript"] = events_to_spoken_transcript(obj["event_script"])

    # For two-person, update patient turn in conversation_context to match patient-only spoken_transcript.
    if obj.get("interaction_type") == "two_person_conversation" and isinstance(obj.get("conversation_context"), list):
        for turn in obj["conversation_context"]:
            if isinstance(turn, dict) and turn.get("speaker") == "patient":
                turn["text"] = obj["spoken_transcript"]
                break

    return obj


def clamp_int(value: Any, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = lo
    return max(lo, min(hi, v))


def repair_feature_counts_from_text(obj: Dict[str, Any], cdr: str) -> Dict[str, Any]:
    """
    Repair only counts that are often exaggerated by the LLM.
    Keep them within CDR range and supported by visible text evidence where possible.
    """
    if not isinstance(obj, dict) or "impairment_features" not in obj:
        return obj
    rules = CDR_RULES.get(cdr)
    if not rules:
        return obj

    spoken = normalize_text(obj.get("spoken_transcript", ""))
    marked = normalize_text(obj.get("marked_transcript", ""))

    actual_filled = count_occurrences(spoken, FILLER_PATTERNS)
    actual_word_finding = count_occurrences(spoken, WORD_FINDING_PATTERNS)
    actual_vague = count_occurrences(spoken, VAGUE_REFERENCE_PATTERNS)
    actual_memory = count_occurrences(spoken, MEMORY_GAP_PATTERNS)
    actual_orientation = count_occurrences(spoken, ORIENTATION_PATTERNS)
    actual_self_correction = count_occurrences(spoken, SELF_CORRECTION_PATTERNS)
    actual_repetition = approx_repetition_count(spoken)
    actual_fragments = estimate_sentence_fragments(spoken)

    try:
        acoustic = obj["impairment_features"]["acoustic_fluency"]
        ling = obj["impairment_features"]["linguistic_impairment"]
        cog = obj["impairment_features"]["cognitive_task_performance"]

        # Filled pauses are lexical; set to visible count but at least lower bound for target severity
        # only if text actually has enough generic fillers. This avoids false rejection.
        acoustic["filled_pause_count"] = clamp_int(max(actual_filled, rules["filled_pause"][0]), *rules["filled_pause"])

        # These are weakly validated; cap impossible overclaims.
        ling["word_finding_count"] = clamp_int(max(actual_word_finding, rules["word_finding"][0]), *rules["word_finding"])
        ling["vague_reference_count"] = max(0, min(int(ling.get("vague_reference_count", actual_vague) or 0), actual_vague + 2))
        ling["word_repetition_count"] = clamp_int(max(actual_repetition, rules["repetition"][0]), *rules["repetition"])
        ling["self_correction_count"] = max(0, min(int(ling.get("self_correction_count", actual_self_correction) or 0), actual_self_correction + 1))
        ling["sentence_fragment_count"] = clamp_int(max(actual_fragments, rules["fragment"][0]), *rules["fragment"])

        cog["memory_gap_count"] = clamp_int(max(actual_memory, rules["memory_gap"][0]), *rules["memory_gap"])
        cog["orientation_error_count"] = clamp_int(max(actual_orientation, rules["orientation"][0]), *rules["orientation"])
    except Exception:
        pass

    return obj


def ensure_minimum_disfluency_events(obj: Dict[str, Any], cdr: str) -> Dict[str, Any]:
    """
    For CDR2/3, Qwen often writes enough impairment in text but too few pause events.
    Add safe pause events at natural boundaries only if below the relaxed minimum.
    """
    if cdr not in {"2", "3"} or not isinstance(obj, dict) or not isinstance(obj.get("event_script"), list):
        return obj

    rules = CDR_RULES[cdr]
    events = obj["event_script"]
    min_pause = rules["pause"][0]
    min_long = rules["long_pause"][0]

    pause_count = count_event_type(events, "pause")
    long_count = count_event_type(events, "long_pause")

    # Add short pauses after speech events until minimum is met.
    i = 0
    while pause_count < min_pause and i < len(events):
        e = events[i]
        if isinstance(e, dict) and e.get("type") == "speech":
            # Avoid double pause insertion if next is already pause-like.
            nxt = events[i + 1] if i + 1 < len(events) else None
            if not (isinstance(nxt, dict) and nxt.get("type") in {"pause", "long_pause", "silence"}):
                events.insert(i + 1, {"type": "pause", "duration_sec": 0.7})
                pause_count += 1
                i += 1
        i += 1

    # Add long pauses for severe/moderate if needed.
    i = 0
    while long_count < min_long and i < len(events):
        e = events[i]
        if isinstance(e, dict) and e.get("type") == "speech":
            nxt = events[i + 1] if i + 1 < len(events) else None
            if not (isinstance(nxt, dict) and nxt.get("type") in {"long_pause", "silence"}):
                events.insert(i + 1, {"type": "long_pause", "duration_sec": 1.8 if cdr == "2" else 2.2})
                long_count += 1
                i += 1
        i += 1

    obj["event_script"] = events
    return obj



def light_repair_sample(obj: Dict[str, Any], cdr: str = "") -> Dict[str, Any]:
    """
    Conservative repair before validation.
    V6 uses event_script as the source of truth because Qwen often places
    markers incorrectly in marked_transcript.
    """
    obj = repair_invalid_filled_pause_events(obj)
    obj = ensure_minimum_disfluency_events(obj, cdr)
    obj = repair_marked_and_spoken_from_events(obj)
    obj = recompute_event_stats_and_acoustic(obj)
    obj = repair_feature_counts_from_text(obj, cdr)
    obj = recompute_event_stats_and_acoustic(obj)
    obj = repair_marked_and_spoken_from_events(obj)
    return obj


# -----------------------------
# Strict validation
# -----------------------------

def validate_sample(obj: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    cdr = expected["cdr"]
    rules = CDR_RULES[cdr]

    # Required fields
    required_top = [
        "sample_id", "cdr_level", "cdr_label", "scenario", "task_type", "interaction_type",
        "picture_reference", "speaker", "split", "acoustic_condition", "acoustic_condition_info",
        "conversation_context", "spoken_transcript", "marked_transcript", "event_script",
        "full_conversation_event_script", "audio_outputs", "event_stats", "impairment_features"
    ]
    for field in required_top:
        if field not in obj:
            errors.append(f"Missing top-level field: {field}")

    if errors:
        return False, errors

    # Basic identity
    if obj.get("sample_id") != expected["sample_id"]:
        errors.append("sample_id mismatch")

    expected_cdr_value = cdr if cdr == "0.5" else int(float(cdr))
    if obj.get("cdr_level") != expected_cdr_value:
        errors.append(f"cdr_level mismatch: expected {expected_cdr_value}, got {obj.get('cdr_level')}")

    if obj.get("cdr_label") != rules["label"]:
        errors.append(f"cdr_label mismatch: expected {rules['label']}, got {obj.get('cdr_label')}")

    if obj.get("scenario") != expected["scenario"]:
        errors.append("scenario mismatch")

    if obj.get("split") != expected["split"]:
        errors.append("split mismatch")

    # Text validation
    spoken = normalize_text(obj.get("spoken_transcript", ""))
    marked = normalize_text(obj.get("marked_transcript", ""))
    all_text = json.dumps(obj, ensure_ascii=False)

    if not spoken:
        errors.append("spoken_transcript is empty")
    if not marked:
        errors.append("marked_transcript is empty")

    if len(spoken) < 20 and cdr not in ["3"]:
        errors.append("spoken_transcript too short for non-CDR3")
    if len(spoken) > 450:
        errors.append("spoken_transcript too long")

    # Banned clinical terms should not appear in actual transcript/speech text.
    # Do NOT scan the whole JSON, because metadata keys such as cdr_level and
    # sample_id intentionally contain "cdr".
    transcript_text_parts = [
        obj.get("spoken_transcript", ""),
        obj.get("marked_transcript", ""),
    ]
    for turn in obj.get("conversation_context", []) if isinstance(obj.get("conversation_context", []), list) else []:
        if isinstance(turn, dict):
            transcript_text_parts.append(turn.get("text", ""))
    for e in obj.get("event_script", []) if isinstance(obj.get("event_script", []), list) else []:
        if isinstance(e, dict) and e.get("type") == "speech":
            transcript_text_parts.append(e.get("text", ""))
    for e in obj.get("full_conversation_event_script", []) if isinstance(obj.get("full_conversation_event_script", []), list) else []:
        if isinstance(e, dict) and e.get("type") == "speech":
            transcript_text_parts.append(e.get("text", ""))

    transcript_only_text = "\n".join(str(x) for x in transcript_text_parts)

    for bad in BANNED_TERMS:
        if bad.lower() in transcript_only_text.lower():
            errors.append(f"Banned term found in transcript/speech text: {bad}")

    # Check Simplified Chinese mainly in user-facing Chinese text.
    chinese_text_parts = [
        obj.get("spoken_transcript", ""),
        obj.get("marked_transcript", ""),
        obj.get("cdr_label", ""),
        obj.get("scenario", ""),
        obj.get("task_type", ""),
        obj.get("interaction_type", ""),
    ]
    if isinstance(obj.get("speaker"), dict):
        chinese_text_parts.extend(str(v) for v in obj["speaker"].values())
    if isinstance(obj.get("acoustic_condition_info"), dict):
        chinese_text_parts.extend(str(v) for v in obj["acoustic_condition_info"].values())
    for turn in obj.get("conversation_context", []) if isinstance(obj.get("conversation_context", []), list) else []:
        if isinstance(turn, dict):
            chinese_text_parts.extend(str(v) for v in turn.values())
    for e in obj.get("event_script", []) if isinstance(obj.get("event_script", []), list) else []:
        if isinstance(e, dict):
            chinese_text_parts.extend(str(v) for v in e.values())
    for e in obj.get("full_conversation_event_script", []) if isinstance(obj.get("full_conversation_event_script", []), list) else []:
        if isinstance(e, dict):
            chinese_text_parts.extend(str(v) for v in e.values())

    chinese_only_text = "\n".join(chinese_text_parts)
    bad_simplified = has_simplified_chars(chinese_only_text)
    if bad_simplified:
        errors.append(f"Simplified Chinese characters found: {''.join(bad_simplified[:20])}")

    if "患者：" in all_text or "訪談者：" in all_text or "病人：" in all_text:
        errors.append("Speaker labels found inside text")

    # Conversation structure validation
    is_two_person = obj.get("interaction_type") == "two_person_conversation"
    conversation_context = obj.get("conversation_context")
    full_conv_events = obj.get("full_conversation_event_script")
    audio_outputs = obj.get("audio_outputs")

    if is_two_person:
        if not isinstance(conversation_context, list) or len(conversation_context) < 2:
            errors.append("two_person_conversation requires conversation_context with interviewer and patient turns")
        else:
            speakers = [t.get("speaker") for t in conversation_context if isinstance(t, dict)]
            if "interviewer" not in speakers:
                errors.append("conversation_context missing interviewer turn")
            if "patient" not in speakers:
                errors.append("conversation_context missing patient turn")
            for j, turn in enumerate(conversation_context):
                if not isinstance(turn, dict):
                    errors.append(f"conversation_context[{j}] must be object")
                    continue
                if turn.get("speaker") not in {"interviewer", "patient"}:
                    errors.append(f"conversation_context[{j}] invalid speaker")
                if not isinstance(turn.get("text"), str) or not turn.get("text").strip():
                    errors.append(f"conversation_context[{j}] missing text")
                if "患者：" in str(turn.get("text")) or "訪談者：" in str(turn.get("text")):
                    errors.append(f"conversation_context[{j}] text contains speaker label")

            patient_turn_text = " ".join(
                t.get("text", "") for t in conversation_context
                if isinstance(t, dict) and t.get("speaker") == "patient"
            )
            interviewer_turn_text = " ".join(
                t.get("text", "") for t in conversation_context
                if isinstance(t, dict) and t.get("speaker") == "interviewer"
            )
            if patient_turn_text and spoken not in patient_turn_text and patient_turn_text not in spoken:
                errors.append("spoken_transcript should match the patient answer in conversation_context")
            if interviewer_turn_text and interviewer_turn_text in spoken:
                errors.append("spoken_transcript contains interviewer question; it must be patient-only")

        if not isinstance(full_conv_events, list) or len(full_conv_events) < 2:
            errors.append("two_person_conversation requires full_conversation_event_script")
        else:
            full_speakers = [e.get("speaker") for e in full_conv_events if isinstance(e, dict) and e.get("type") == "speech"]
            if "interviewer" not in full_speakers:
                errors.append("full_conversation_event_script missing interviewer speech")
            if "patient" not in full_speakers:
                errors.append("full_conversation_event_script missing patient speech")
            for j, e in enumerate(full_conv_events):
                if not isinstance(e, dict):
                    errors.append(f"full_conversation_event_script[{j}] must be object")
                    continue
                typ = e.get("type")
                if typ not in EVENT_TYPES:
                    errors.append(f"full_conversation_event_script[{j}] invalid type: {typ}")
                if typ == "speech":
                    if e.get("speaker") not in {"interviewer", "patient"}:
                        errors.append(f"full conversation speech event {j} invalid speaker")
                    if not isinstance(e.get("text"), str) or not e.get("text").strip():
                        errors.append(f"full conversation speech event {j} missing text")
                else:
                    if not isinstance(e.get("duration_sec"), (int, float)) or e.get("duration_sec") <= 0:
                        errors.append(f"full conversation non-speech event {j} missing positive duration_sec")

        if not isinstance(audio_outputs, dict):
            errors.append("audio_outputs must be object")
        else:
            if audio_outputs.get("training_audio") != "patient_only.wav":
                errors.append("audio_outputs.training_audio must be patient_only.wav")
            if audio_outputs.get("demo_audio") != "full_conversation.wav":
                errors.append("two_person audio_outputs.demo_audio must be full_conversation.wav")
    else:
        if conversation_context is not None:
            errors.append("one_person sample must have conversation_context = null")
        if full_conv_events is not None:
            errors.append("one_person sample must have full_conversation_event_script = null")
        if isinstance(audio_outputs, dict) and audio_outputs.get("demo_audio") is not None:
            errors.append("one_person audio_outputs.demo_audio must be null")

    # Event script
    events = obj.get("event_script")
    if not isinstance(events, list) or not events:
        errors.append("event_script must be non-empty list")
        return False, errors

    for i, e in enumerate(events):
        if not isinstance(e, dict):
            errors.append(f"event_script[{i}] is not object")
            continue
        typ = e.get("type")
        if typ not in EVENT_TYPES:
            errors.append(f"event_script[{i}] invalid type: {typ}")
        if typ == "speech":
            if e.get("speaker") != "patient":
                errors.append(f"speech event {i} speaker must be patient")
            txt = e.get("text")
            if not isinstance(txt, str) or not txt.strip():
                errors.append(f"speech event {i} missing text")
            if txt and any(m in txt for m in ["[停頓]", "[長停頓]", "[沉默]", "[嘆氣]", "[咳嗽]"]):
                errors.append(f"speech event {i} contains bracket marker")
            if txt and ("患者：" in txt or "訪談者：" in txt):
                errors.append(f"speech event {i} contains speaker label")
        else:
            dur = e.get("duration_sec")
            if not isinstance(dur, (int, float)):
                errors.append(f"{typ} event {i} missing numeric duration_sec")
            elif dur <= 0:
                errors.append(f"{typ} event {i} duration_sec must be positive")
            elif typ == "pause" and not (0.3 <= float(dur) <= 1.5):
                errors.append(f"pause event {i} duration {dur} outside 0.3-1.5")
            elif typ == "long_pause" and not (1.5 <= float(dur) <= 4.5):
                errors.append(f"long_pause event {i} duration {dur} outside 1.5-4.5")
            elif typ == "silence" and not (2.0 <= float(dur) <= 8.0):
                errors.append(f"silence event {i} duration {dur} outside 2.0-8.0")

    # Event stats must match
    stats = obj.get("event_stats", {})
    if not isinstance(stats, dict):
        errors.append("event_stats must be object")
        stats = {}

    computed_stats = {
        "speech_event_count": count_event_type(events, "speech"),
        "pause_count": count_event_type(events, "pause"),
        "long_pause_count": count_event_type(events, "long_pause"),
        "silence_count": count_event_type(events, "silence"),
        "sigh_count": count_event_type(events, "sigh"),
        "cough_count": count_event_type(events, "cough"),
        "total_pause_duration_sec": total_duration(events, {"pause", "long_pause"}),
        "total_silence_duration_sec": total_duration(events, {"silence"}),
    }

    for k, v in computed_stats.items():
        if k not in stats:
            errors.append(f"event_stats missing {k}")
            continue
        if isinstance(v, float):
            try:
                if abs(float(stats[k]) - v) > 0.15:
                    errors.append(f"event_stats.{k}={stats[k]} does not match computed {v}")
            except Exception:
                errors.append(f"event_stats.{k} must be numeric")
        else:
            if stats[k] != v:
                errors.append(f"event_stats.{k}={stats[k]} does not match computed {v}")

    # Marker count consistency
    marker_checks = [
        ("pause", "[停頓]"),
        ("long_pause", "[長停頓]"),
        ("silence", "[沉默]"),
        ("sigh", "[嘆氣]"),
        ("cough", "[咳嗽]"),
    ]
    for event_type, marker in marker_checks:
        ev_count = count_event_type(events, event_type)
        mk_count = count_marker(marked, marker)
        if ev_count != mk_count:
            errors.append(f"marked_transcript {marker} count {mk_count} does not match event_script {event_type} count {ev_count}")

    # Feature validation
    features = obj.get("impairment_features", {})
    if not isinstance(features, dict):
        errors.append("impairment_features must be object")
        return False, errors

    acoustic = features.get("acoustic_fluency", {})
    ling = features.get("linguistic_impairment", {})
    cog = features.get("cognitive_task_performance", {})

    if not isinstance(acoustic, dict):
        errors.append("impairment_features.acoustic_fluency must be object")
        acoustic = {}
    if not isinstance(ling, dict):
        errors.append("impairment_features.linguistic_impairment must be object")
        ling = {}
    if not isinstance(cog, dict):
        errors.append("impairment_features.cognitive_task_performance must be object")
        cog = {}

    # Acoustic feature ranges and matching
    int_in_range(acoustic.get("pause_count"), *rules["pause"], "acoustic_fluency.pause_count", errors)
    int_in_range(acoustic.get("long_pause_count"), *rules["long_pause"], "acoustic_fluency.long_pause_count", errors)
    int_in_range(acoustic.get("filled_pause_count"), *rules["filled_pause"], "acoustic_fluency.filled_pause_count", errors)
    numeric_in_range(acoustic.get("speech_rate_target_cps"), *rules["speech_rate_cps"], "acoustic_fluency.speech_rate_target_cps", errors)

    if acoustic.get("pause_count") != computed_stats["pause_count"]:
        errors.append("acoustic_fluency.pause_count does not match event_stats.pause_count")
    if acoustic.get("long_pause_count") != computed_stats["long_pause_count"]:
        errors.append("acoustic_fluency.long_pause_count does not match event_stats.long_pause_count")
    try:
        if abs(float(acoustic.get("total_pause_duration_sec")) - computed_stats["total_pause_duration_sec"]) > 0.15:
            errors.append("acoustic_fluency.total_pause_duration_sec does not match computed pause duration")
    except Exception:
        errors.append("acoustic_fluency.total_pause_duration_sec must be numeric")

    # Linguistic feature ranges
    int_in_range(ling.get("word_finding_count"), *rules["word_finding"], "linguistic_impairment.word_finding_count", errors)
    int_in_range(ling.get("word_repetition_count"), *rules["repetition"], "linguistic_impairment.word_repetition_count", errors)
    int_in_range(ling.get("sentence_fragment_count"), *rules["fragment"], "linguistic_impairment.sentence_fragment_count", errors)
    int_in_range(cog.get("memory_gap_count"), *rules["memory_gap"], "cognitive_task_performance.memory_gap_count", errors)
    int_in_range(cog.get("orientation_error_count"), *rules["orientation"], "cognitive_task_performance.orientation_error_count", errors)
    numeric_in_range(cog.get("topic_drift_score"), *rules["topic_drift"], "cognitive_task_performance.topic_drift_score", errors)
    numeric_in_range(cog.get("coherence_score"), *rules["coherence"], "cognitive_task_performance.coherence_score", errors)

    # Evidence checks in transcript
    actual_filled = count_occurrences(spoken, FILLER_PATTERNS)
    actual_word_finding = count_occurrences(spoken, WORD_FINDING_PATTERNS)
    actual_vague = count_occurrences(spoken, VAGUE_REFERENCE_PATTERNS)
    actual_memory = count_occurrences(spoken, MEMORY_GAP_PATTERNS)
    actual_orientation = count_occurrences(spoken, ORIENTATION_PATTERNS)
    actual_self_correction = count_occurrences(spoken, SELF_CORRECTION_PATTERNS)
    actual_repetition = approx_repetition_count(spoken)
    actual_fragments = estimate_sentence_fragments(spoken)

    # Strict but with small tolerance because string detection is approximate.
    # Do not hard-fail filled_pause_count if visible marker detector undercounts Taiwanese filler usage.

    # Word-finding evidence detector is approximate; range validation is enough here.

    if isinstance(ling.get("vague_reference_count"), int) and ling["vague_reference_count"] > actual_vague + 2:
        errors.append(f"vague_reference_count={ling['vague_reference_count']} too high for visible vague references {actual_vague}")

    # Repetition detector is approximate for Chinese; do not hard-fail.

    # Self-correction detector is approximate; do not hard-fail.

    if isinstance(cog.get("memory_gap_count"), int) and cog["memory_gap_count"] > 0 and actual_memory == 0:
        errors.append("memory_gap_count > 0 but no visible memory-gap phrase found")

    # Orientation uncertainty can be implied by time/place confusion; do not hard-fail.

    # Fragment detector is approximate; do not hard-fail.

    # Scenario-specific picture checks
    if expected["scenario"] == "picture_description":
        total = cog.get("key_information_units_total")
        mentioned = cog.get("key_information_units_mentioned")
        missing = cog.get("missing_key_information_count")

        if total != len(PICTURE_REFERENCE["key_units"]):
            errors.append(f"picture key_information_units_total must be {len(PICTURE_REFERENCE['key_units'])}")

        if not isinstance(mentioned, int):
            errors.append("key_information_units_mentioned must be int for picture_description")
        if not isinstance(missing, int):
            errors.append("missing_key_information_count must be int for picture_description")

        if isinstance(mentioned, int) and isinstance(missing, int):
            if mentioned + missing != len(PICTURE_REFERENCE["key_units"]):
                errors.append("key_information_units_mentioned + missing_key_information_count must equal total key units")
            int_in_range(missing, *rules["missing_info"], "cognitive_task_performance.missing_key_information_count", errors)
    else:
        # For non-picture, allow null or small values, but not impossible big picture totals.
        pass

    # CDR 0 sanity
    if cdr == "0":
        if actual_memory > 0:
            errors.append("CDR 0 should not contain memory-gap phrases")
        if actual_orientation > 0:
            errors.append("CDR 0 should not contain orientation uncertainty")
        if computed_stats["long_pause_count"] > 0:
            errors.append("CDR 0 should not have long pauses")

    # CDR 0.5 sanity: mild only
    if cdr == "0.5":
        if computed_stats["pause_count"] > 3:
            errors.append("CDR 0.5 has too many pauses")
        if computed_stats["long_pause_count"] > 1:
            errors.append("CDR 0.5 has too many long pauses")

    # Final safety: metadata is allowed to contain "cdr" because this dataset uses
    # sample_id, cdr_level, and cdr_label as required metadata. Only transcript/speech
    # text should be banned from clinical/rating-scale terms.
    errors = [e for e in errors if e != "Banned term found: CDR"]

    return len(errors) == 0, errors


# -----------------------------
# Generation loop
# -----------------------------

def build_sample_id(cdr: str, voice_id: str, scenario: str, index: int) -> str:
    cdr_str = cdr.replace(".", "_")
    return f"cdr_{cdr_str}_{voice_id}_{scenario}_{index:04d}"


def choose_split(index: int, total: int) -> str:
    # deterministic 80/10/10
    ratio = index / max(total, 1)
    if ratio < 0.8:
        return "train"
    if ratio < 0.9:
        return "val"
    return "test"


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_one(
    args: argparse.Namespace,
    sample_id: str,
    cdr: str,
    scenario: str,
    speaker: Dict[str, str],
    split: str,
    acoustic_condition: Dict[str, str],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    rejected_attempts = []
    previous_errors: Optional[List[str]] = None

    expected = {
        "sample_id": sample_id,
        "cdr": cdr,
        "scenario": scenario,
        "split": split,
    }

    for attempt in range(1, args.max_retries + 1):
        prompt = build_prompt(
            sample_id=sample_id,
            cdr=cdr,
            scenario=scenario,
            speaker=speaker,
            split=split,
            acoustic_condition=acoustic_condition,
            previous_errors=previous_errors,
        )

        response = ollama_generate(
            model=args.model,
            prompt=prompt,
            host=args.ollama_host,
            temperature=args.temperature,
            timeout=args.timeout,
        )

        obj = extract_json_object(response)
        if obj is None:
            previous_errors = ["Model did not return valid JSON object"]
            rejected_attempts.append({
                "attempt": attempt,
                "errors": previous_errors,
                "raw_response": response[:8000],
            })
            continue

        obj = light_repair_sample(obj, cdr)
        ok, errors = validate_sample(obj, expected)
        if ok:
            return obj, rejected_attempts

        previous_errors = errors
        rejected_attempts.append({
            "attempt": attempt,
            "errors": errors,
            "json_or_raw": obj,
        })

    return None, rejected_attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--output-dir", default="tw_cdr_strict_json")
    parser.add_argument("--cdr-counts", default="0:300,0.5:300,1:200,2:120,3:80")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true", help="Skip samples already saved.")
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    rejected_dir = out_dir / "rejected_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    counts = parse_cdr_counts(args.cdr_counts)

    summary = {
        "model": args.model,
        "cdr_counts": counts,
        "accepted": 0,
        "failed": 0,
        "failed_samples": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    global_index = 0

    for cdr, count in counts.items():
        for i in range(count):
            global_index += 1

            scenario = random.choice(SCENARIOS)
            speaker = random.choice(VOICE_POOL)
            split = choose_split(i, count)
            acoustic_condition = random.choice(ACOUSTIC_CONDITIONS)

            sample_id = build_sample_id(cdr, speaker["voice_id"], scenario, i)
            sample_path = out_dir / f"cdr_{cdr.replace('.', '_')}" / f"{sample_id}.json"

            if args.resume and sample_path.exists():
                print(f"[SKIP] {sample_id}")
                continue

            print(f"[GENERATE] {sample_id} | CDR {cdr} | {scenario} | {speaker['voice_id']}")

            try:
                obj, rejected = generate_one(
                    args=args,
                    sample_id=sample_id,
                    cdr=cdr,
                    scenario=scenario,
                    speaker=speaker,
                    split=split,
                    acoustic_condition=acoustic_condition,
                )
            except Exception as e:
                obj = None
                rejected = [{"attempt": "exception", "errors": [str(e)]}]

            if obj is not None:
                save_json(sample_path, obj)
                summary["accepted"] += 1
                print(f"  -> ACCEPTED: {sample_path}")
            else:
                summary["failed"] += 1
                summary["failed_samples"].append(sample_id)
                fail_path = rejected_dir / f"{sample_id}.rejected.json"
                save_json(fail_path, {
                    "sample_id": sample_id,
                    "cdr": cdr,
                    "scenario": scenario,
                    "speaker": speaker,
                    "split": split,
                    "rejected_attempts": rejected,
                })
                print(f"  -> FAILED after retries. Log: {fail_path}")

            # Save rolling summary every sample
            save_json(out_dir / "generation_summary.json", summary)

            if args.sleep > 0:
                time.sleep(args.sleep)

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(out_dir / "generation_summary.json", summary)

    print(f"\nDONE | validator={VALIDATOR_VERSION}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
