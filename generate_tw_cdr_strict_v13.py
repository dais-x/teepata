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
import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

VALIDATOR_VERSION = "V12_CDR2_CDR3_NONCOGNITIVE_RELIABLE_SCORING"


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

# Task-group organization used for saving folders and dataset balance.
# Final folder path: output_dir / split / cdr_x / task_group / sample.json
TASK_GROUPS = ["picture_description", "cognitive_interview", "real_life_scenarios"]
TASK_GROUP_WEIGHTS = {
    "picture_description": 0.35,
    "cognitive_interview": 0.35,
    "real_life_scenarios": 0.30,
}

REAL_LIFE_SCENARIOS = ["medicine_routine", "daily_life", "family", "home", "market"]

COGNITIVE_INTERVIEW_TASK = {
    "task_type": "cognitive_interview",
    "prompt_sequence": [
        "我等一下會請你記三個東西：鑰匙、蘋果、火車。你先跟著說一次。",
        "你知道今天大概是星期幾嗎？現在大概是上午還是下午？我們現在是在什麼地方？",
        "剛剛那三個東西，你還記得是哪三個嗎？",
    ],
    "expected_items": ["鑰匙", "蘋果", "火車"],
    "orientation_questions": ["星期幾", "上午或下午", "所在地方"],
    "max_score": 9,
}

REAL_LIFE_TASK_BANK = {
    "medicine_routine": {
        "task_type": "medicine_routine_recall",
        "prompt": "你可以說一下今天早上吃藥的順序嗎？",
        "expected_steps": ["起床", "吃早餐", "看藥袋或藥盒", "吃藥"],
        "max_score": 4,
    },
    "daily_life": {
        "task_type": "daily_life_narrative",
        "prompt": "你可以跟我說一下今天早上起來之後做了哪些事情嗎？",
        "expected_steps": ["起床", "盥洗", "吃早餐", "安排活動"],
        "max_score": 4,
    },
    "family": {
        "task_type": "family_event_recall",
        "prompt": "你可以說一下最近一次跟家人吃飯或聊天的情形嗎？",
        "expected_steps": ["人物", "地點", "活動", "時間或順序"],
        "max_score": 4,
    },
    "home": {
        "task_type": "home_activity_recall",
        "prompt": "你平常在家裡一天大概會做哪些事情？",
        "expected_steps": ["家務或休息", "用餐", "看電視或散步", "時間順序"],
        "max_score": 4,
    },
    "market": {
        "task_type": "market_activity_recall",
        "prompt": "你最近去市場或買東西時，通常會怎麼安排？",
        "expected_steps": ["出門", "買東西", "付錢", "回家"],
        "max_score": 4,
    },
}

MAINLAND_PHRASES = [
    "視頻", "公交", "出租車", "地鐵", "普通話", "小區", "早上好", "老伴兒", "哪兒", "大白天", "具體", "这", "说", "吗"
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
    "脸": "臉", "刚": "剛", "着": "著", "干": "乾", "别": "別", "种": "種",
    "离": "離", "给": "給", "让": "讓", "从": "從", "发": "發", "图": "圖",
    "体": "體", "处": "處", "备": "備", "复": "複", "为": "為", "还": "還",
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
    "忘記怎麼說", "拿來", "用的那個", "就是那個", "什麼來著", "叫什麼來著", "不知道叫什麼"
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
    Conservative abnormal repetition detector.
    Do NOT count normal repeated content words such as 積木 appearing twice in a picture description.
    Count only obvious disfluent repetitions like 我我, 是是, 那個、那個, or repeated short phrases.
    """
    if not isinstance(text, str):
        return 0
    t = normalize_text(text)
    count = 0

    # Direct disfluent character/word repetitions. Keep a whitelist so normal kinship words are ignored.
    direct = re.findall(r"([我你他她是有在要那這嗯呃啊])\1+", t)
    count += len(direct)

    # Repeated fillers or short function chunks separated by punctuation/space.
    count += len(re.findall(r"(那個|這個|嗯|呃|啊|我想一下)[，、\s.。…]+\1", t))

    # Same 2-4 character phrase repeated with punctuation, but ignore common normal content words.
    repeated_phrases = re.findall(r"([\u4e00-\u9fff]{2,4})[，、\s.。…]+\1", t)
    ignore = {
        "媽媽", "爸爸", "哥哥", "姐姐", "妹妹", "爺爺", "奶奶", "阿公", "阿嬤",
        "積木", "茶杯", "照片", "相簿", "鑰匙", "蘋果", "火車", "醫院", "診間"
    }
    count += sum(1 for p in repeated_phrases if p not in ignore)

    return max(0, min(count, 9))


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




def cdr_folder_name(cdr: str) -> str:
    return f"cdr_{cdr.replace('.', '_')}"


def choose_task_group(index: int, total: int) -> str:
    """Deterministic 35/35/30 allocation inside each CDR level."""
    if total <= 0:
        return "picture_description"
    pos = index / total
    if pos < TASK_GROUP_WEIGHTS["picture_description"]:
        return "picture_description"
    if pos < TASK_GROUP_WEIGHTS["picture_description"] + TASK_GROUP_WEIGHTS["cognitive_interview"]:
        return "cognitive_interview"
    return "real_life_scenarios"


def build_task_plan(task_group: str, scenario: str) -> Dict[str, Any]:
    if task_group == "picture_description":
        return {
            "task_group": "picture_description",
            "task_type": "picture_description",
            "prompt": "請你看這張圖，告訴我你看到什麼。可以從人物、物品、動作和可能發生的事情開始講。",
            "expected_response_type": "open_narrative",
            "scoring_method": "key_information_units",
            "key_units_total": len(PICTURE_REFERENCE["key_units"]),
            "expected_key_units": PICTURE_REFERENCE["key_units"],
            "max_score": len(PICTURE_REFERENCE["key_units"]),
        }
    if task_group == "cognitive_interview":
        return {
            "task_group": "cognitive_interview",
            **COGNITIVE_INTERVIEW_TASK,
            "scoring_method": "orientation_plus_delayed_recall_speech_score",
        }
    task = REAL_LIFE_TASK_BANK.get(scenario, REAL_LIFE_TASK_BANK["daily_life"])
    return {
        "task_group": "real_life_scenarios",
        **task,
        "scoring_method": "expected_steps_and_sequence",
    }


def choose_scenario_for_task_group(task_group: str) -> str:
    if task_group == "picture_description":
        return "picture_description"
    if task_group == "cognitive_interview":
        return "clinic"
    return random.choice(REAL_LIFE_SCENARIOS)


def expected_score_range_for_cdr(cdr: str, max_score: int) -> Tuple[int, int]:
    """Loose but useful speech-cognitive score envelope by severity."""
    if cdr == "0":
        return max(0, int(round(max_score * 0.85))), max_score
    if cdr == "0.5":
        return max(0, int(round(max_score * 0.65))), max_score
    if cdr == "1":
        return max(0, int(round(max_score * 0.40))), max(1, int(round(max_score * 0.85)))
    if cdr == "2":
        return max(0, int(round(max_score * 0.15))), max(1, int(round(max_score * 0.60)))
    return 0, max(1, int(round(max_score * 0.35)))


def build_speech_cognitive_schema(task_plan: Dict[str, Any], cdr: str) -> Dict[str, Any]:
    max_score = int(task_plan.get("max_score", 10))
    lo, hi = expected_score_range_for_cdr(cdr, max_score)
    base = {
        "task_group": task_plan["task_group"],
        "task_type": task_plan["task_type"],
        "prompt": task_plan.get("prompt") or " / ".join(task_plan.get("prompt_sequence", [])),
        "scoring_method": task_plan.get("scoring_method"),
        "score": lo,
        "max_score": max_score,
        "score_interpretation": "speech-based cognitive task score, not clinical MMSE score"
    }
    if task_plan["task_group"] == "picture_description":
        base.update({
            "expected_key_units": task_plan["expected_key_units"],
            "key_units_total": max_score,
            "key_units_mentioned": lo,
            "missing_key_information_count": max_score - lo,
            "incorrect_detail_count": 0 if cdr in {"0", "0.5"} else 1,
        })
    elif task_plan["task_group"] == "cognitive_interview":
        base.update({
            "expected_items": task_plan["expected_items"],
            "orientation_questions": task_plan["orientation_questions"],
            "subtasks": {
                "immediate_repetition": {"score": 3 if cdr in {"0", "0.5"} else 2, "max_score": 3},
                "orientation": {"score": 3 if cdr in {"0", "0.5"} else 2, "max_score": 3},
                "delayed_recall": {"score": max(0, min(3, lo - 4)), "max_score": 3},
            },
            "patient_recalled_items": [],
            "orientation_correct_count": 3 if cdr in {"0", "0.5"} else 2,
            "orientation_error_count": 0 if cdr in {"0", "0.5"} else 1,
        })
    else:
        base.update({
            "expected_steps": task_plan.get("expected_steps", []),
            "steps_mentioned": lo,
            "sequence_error_count": 0 if cdr in {"0", "0.5"} else 1,
            "memory_gap_count": 0 if cdr in {"0", "0.5"} else 1,
        })
    return base


def contains_mainland_phrasing(text: str) -> List[str]:
    return [p for p in MAINLAND_PHRASES if p in text]


def write_manifest_row(manifest_path: Path, obj: Dict[str, Any], json_path: Path, out_dir: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    task = obj.get("speech_cognitive_task", {}) if isinstance(obj.get("speech_cognitive_task"), dict) else {}
    acoustic = obj.get("impairment_features", {}).get("acoustic_fluency", {}) if isinstance(obj.get("impairment_features"), dict) else {}
    ling = obj.get("impairment_features", {}).get("linguistic_impairment", {}) if isinstance(obj.get("impairment_features"), dict) else {}
    cog = obj.get("impairment_features", {}).get("cognitive_task_performance", {}) if isinstance(obj.get("impairment_features"), dict) else {}
    row = {
        "sample_id": obj.get("sample_id", ""),
        "split": obj.get("split", ""),
        "cdr_level": obj.get("cdr_level", ""),
        "cdr_label": obj.get("cdr_label", ""),
        "task_group": obj.get("task_group", ""),
        "task_type": obj.get("task_type", ""),
        "scenario": obj.get("scenario", ""),
        "voice_id": obj.get("speaker", {}).get("voice_id", "") if isinstance(obj.get("speaker"), dict) else "",
        "gender": obj.get("speaker", {}).get("gender", "") if isinstance(obj.get("speaker"), dict) else "",
        "score": task.get("score", ""),
        "max_score": task.get("max_score", ""),
        "pause_count": acoustic.get("pause_count", ""),
        "long_pause_count": acoustic.get("long_pause_count", ""),
        "word_finding_count": ling.get("word_finding_count", ""),
        "memory_gap_count": cog.get("memory_gap_count", ""),
        "coherence_score": cog.get("coherence_score", ""),
        "json_path": str(json_path.relative_to(out_dir)),
    }
    fieldnames = list(row.keys())
    exists = manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

# -----------------------------
# Prompt construction
# -----------------------------

def build_prompt(
    sample_id: str,
    cdr: str,
    task_group: str,
    scenario: str,
    task_plan: Dict[str, Any],
    speaker: Dict[str, str],
    split: str,
    previous_errors: Optional[List[str]] = None
) -> str:
    rules = CDR_RULES[cdr]

    if task_group == "picture_description":
        interaction_type = "one_person_description"
        task_type = "picture_description"
        picture_ref = PICTURE_REFERENCE
        interviewer_question = None
    elif task_group == "cognitive_interview":
        interaction_type = "two_person_conversation"
        task_type = "cognitive_interview"
        picture_ref = None
        interviewer_question = " ".join(task_plan.get("prompt_sequence", []))
    else:
        interaction_type = "two_person_conversation" if scenario in {"medicine_routine", "clinic"} else "one_person_description"
        task_type = task_plan.get("task_type", "daily_life_narrative")
        picture_ref = None
        interviewer_question = task_plan.get("prompt")

    speech_cognitive_schema = build_speech_cognitive_schema(task_plan, cdr)

    if interaction_type == "two_person_conversation" and task_group == "cognitive_interview":
        conversation_context_schema = []
        full_conversation_schema = []
        for idx, q in enumerate(task_plan.get("prompt_sequence", []), start=1):
            conversation_context_schema.append({"speaker": "interviewer", "text": q})
            conversation_context_schema.append({"speaker": "patient", "text": f"患者第{idx}段回答，必須只包含患者語音。"})
            full_conversation_schema.append({"type": "speech", "speaker": "interviewer", "text": q})
            full_conversation_schema.append({"type": "pause", "duration_sec": 0.5})
            full_conversation_schema.append({"type": "speech", "speaker": "patient", "text": f"患者第{idx}段語音片段"})
            if idx < len(task_plan.get("prompt_sequence", [])):
                full_conversation_schema.append({"type": "pause", "duration_sec": 0.4})
    elif interaction_type == "two_person_conversation":
        conversation_context_schema = [
            {"speaker": "interviewer", "text": interviewer_question or speech_cognitive_schema["prompt"]},
            {"speaker": "patient", "text": "患者回答，必須與 spoken_transcript 內容一致。"}
        ]
        full_conversation_schema = [
            {"type": "speech", "speaker": "interviewer", "text": interviewer_question or speech_cognitive_schema["prompt"]},
            {"type": "pause", "duration_sec": 0.5},
            {"type": "speech", "speaker": "patient", "text": "患者語音片段"}
        ]
    else:
        conversation_context_schema = None
        full_conversation_schema = None

    required_schema = {
        "sample_id": sample_id,
        "cdr_level": cdr if cdr == "0.5" else int(float(cdr)),
        "cdr_label": rules["label"],
        "task_group": task_group,
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
        "speech_cognitive_task": speech_cognitive_schema,
        "conversation_context": conversation_context_schema,
        "spoken_transcript": "只包含患者語音內容。台灣華語文字，不要標記停頓，不要出現訪談者/患者標籤。",
        "marked_transcript": "只包含患者語音內容。同一段文字，但插入 [停頓] [長停頓] [沉默] [嘆氣] [咳嗽]。",
        "event_script": [
            {"type": "speech", "speaker": "patient", "text": "患者語音片段，訓練用，只能 patient"},
            {"type": "pause", "duration_sec": 0.8},
            {"type": "speech", "speaker": "patient", "text": "患者下一段語音"}
        ],
        "full_conversation_event_script": full_conversation_schema,
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
        },
        "repair_log": {
            "spoken_transcript_rebuilt_from_event_script": False,
            "marked_transcript_rebuilt_from_event_script": False,
            "feature_counts_repaired": False
        }
    }

    score_lo, score_hi = expected_score_range_for_cdr(cdr, int(speech_cognitive_schema["max_score"]))

    error_text = ""
    if previous_errors:
        safe_errors = []
        for e in previous_errors[:25]:
            e = str(e).replace("CDR", "rating-scale term").replace("cdr", "rating-scale term")
            e = e.replace("dementia", "diagnosis term").replace("Alzheimer", "diagnosis term")
            safe_errors.append(e)
        error_text = "\nPREVIOUS ATTEMPT FAILED. Fix these validation errors:\n" + "\n".join(f"- {e}" for e in safe_errors)

    prompt = f"""
/no_think
You are generating STRICT research-style synthetic Taiwanese Mandarin speech data for cognitive/speech impairment simulation.

Return ONLY valid JSON. No markdown. No explanation. Do not output <think>, reasoning, analysis, or any text before the JSON.

IMPORTANT LANGUAGE RULES:
- Use Traditional Chinese and natural Taiwanese Mandarin only.
- Do NOT use Simplified Chinese characters.
- Avoid Mainland phrasing. Prefer Taiwan terms such as 影片 not 視頻, 公車 not 公交, 捷運 not 地鐵 when relevant,計程車 not 出租車.
- Do NOT mention diagnosis names, disease names, rating-scale names, or clinical labels inside the transcript.
- Do NOT include "患者：" or "訪談者：" inside spoken_transcript, marked_transcript, conversation text, or event_script speech text.
- The patient speech should sound like an elderly Taiwanese Mandarin speaker.
- Keep impairment natural, not exaggerated.
- Do NOT include top-level acoustic_condition or acoustic_condition_info. Audio noise/effects will be handled later by the synthesis script, not by this JSON.

TARGET SAMPLE:
sample_id: {sample_id}
CDR level: {cdr}
CDR label: {rules["label"]}
task_group: {task_group}
scenario: {scenario}
task_type: {task_type}
interaction_type: {interaction_type}
speaker: {speaker["voice_id"]}
split: {split}

TASK PLAN:
{json.dumps(task_plan, ensure_ascii=False, indent=2)}

SPEECH COGNITIVE TASK REQUIREMENTS:
- Include top-level speech_cognitive_task exactly matching task_group and task_type.
- speech_cognitive_task.score must be between {score_lo} and {score_hi} for this severity.
- speech_cognitive_task.max_score must be {speech_cognitive_schema["max_score"]}.
- This is NOT an MMSE score. It is a speech-based cognitive task score.
- The transcript must visibly support the score. Do not claim missing items/errors that are not reflected in patient speech.
- For cognitive_interview: MUST create exactly three patient turns: immediate repetition answer, orientation answer, delayed recall answer.
- For cognitive_interview: the delayed recall answer must answer only the delayed recall question; do not mix delayed recall content into the orientation answer.
- For cognitive_interview: delayed recall must feel natural because earlier immediate repetition and orientation are in conversation_context/full_conversation_event_script.
- For picture_description: score using key information units from the fixed picture reference.
- For real_life_scenarios: score using expected steps, sequence errors, and memory gaps.

PICTURE REFERENCE, only if task_group is picture_description:
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
- If orientation_error_count > 0, use it only in cognitive_interview or natural clinic/medicine context.
- CDR 0 must be normal, complete, coherent, and score high.
- CDR 0.5 must be very mild. No obvious place/date confusion; minor hesitation only.
- CDR 1 should show mild but clear speech/cognitive difficulty.
- CDR 2 picture descriptions should be incomplete.
- CDR 3 should be fragmented but still valid Taiwanese Mandarin speech.

EVENT SCRIPT RULES:
- event_script is PATIENT-ONLY audio for model training.
- event_script must be a list of events.
- Allowed event types: speech, pause, long_pause, silence, sigh, cough.
- NEVER use filled_pause as an event type. Filled pauses such as 嗯, 呃, 那個 must be inside speech text.
- In event_script, speech event must have speaker="patient" and text.
- pause/long_pause/silence/sigh/cough must have duration_sec.
- marked_transcript should contain [停頓] for pause, [長停頓] for long_pause, [沉默] for silence, [嘆氣] for sigh, [咳嗽] for cough. The script will repair marker placement from event_script.
- event_stats must exactly match patient-only event_script counts.
- impairment_features.acoustic_fluency.pause_count must equal event_stats.pause_count.
- impairment_features.acoustic_fluency.long_pause_count must equal event_stats.long_pause_count.

CONVERSATION RULES:
- If interaction_type is two_person_conversation, conversation_context is REQUIRED.
- Use the task prompt as interviewer speech.
- conversation_context must include at least one interviewer turn and one patient turn.
- spoken_transcript must contain ONLY patient speech, not interviewer questions.
- marked_transcript must contain ONLY patient speech plus event markers.
- event_script must contain ONLY patient speech and patient pauses for training.
- full_conversation_event_script must contain interviewer speech plus patient speech for demo audio.
- For cognitive_interview, full_conversation_event_script must interleave the three interviewer prompts with patient responses. Do NOT put all interviewer questions as one first event.
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

    # For simple two-person tasks, update the single patient turn to match patient-only spoken_transcript.
    # Do NOT do this for cognitive_interview, which intentionally has three separate patient turns.
    if obj.get("interaction_type") == "two_person_conversation" and obj.get("task_group") != "cognitive_interview" and isinstance(obj.get("conversation_context"), list):
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

        # Evidence-based counts. Do not inflate features just to satisfy severity.
        acoustic["filled_pause_count"] = clamp_int(actual_filled, 0, rules["filled_pause"][1])

        ling["word_finding_count"] = clamp_int(actual_word_finding, 0, rules["word_finding"][1])
        ling["vague_reference_count"] = max(0, min(actual_vague, rules["word_finding"][1] + 3))
        ling["word_repetition_count"] = clamp_int(actual_repetition, 0, rules["repetition"][1])
        ling["self_correction_count"] = max(0, min(actual_self_correction, 3))
        ling["sentence_fragment_count"] = clamp_int(actual_fragments, 0, rules["fragment"][1])

        cog["memory_gap_count"] = clamp_int(actual_memory, 0, rules["memory_gap"][1])
        if obj.get("task_group") == "picture_description":
            cog["orientation_error_count"] = 0
        else:
            cog["orientation_error_count"] = clamp_int(actual_orientation, 0, rules["orientation"][1])
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



def strip_acoustic_condition_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Remove acoustic-condition metadata. Audio noise/effects belong to synthesis, not JSON generation."""
    if isinstance(obj, dict):
        obj.pop("acoustic_condition", None)
        obj.pop("acoustic_condition_info", None)
    return obj


def convert_known_simplified_text(obj: Any) -> Any:
    """Convert known simplified Chinese characters in JSON string values; still reject unmapped simplified chars later."""
    if isinstance(obj, str):
        for s, t in SIMPLIFIED_TO_TRADITIONAL.items():
            obj = obj.replace(s, t)
        return obj
    if isinstance(obj, list):
        return [convert_known_simplified_text(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_known_simplified_text(v) for k, v in obj.items()}
    return obj


def _contains_any(text: str, patterns: List[str]) -> bool:
    return any(p and p in text for p in patterns)


PICTURE_KEY_UNIT_PATTERNS = [
    ("客廳場景", ["客廳"]),
    ("阿公坐在沙發上看相簿", ["阿公", "老人", "老先生", "爺爺", "相簿", "照片", "沙發"]),
    ("老花眼鏡", ["老花眼鏡", "眼鏡"]),
    ("茶或茶杯", ["茶杯", "茶", "杯子"]),
    ("小孩玩積木", ["小孩", "孩子", "積木"]),
    ("媽媽在窗邊接電話", ["媽媽", "女兒", "女人", "窗邊", "接電話", "講電話"]),
    ("貓靠近茶几", ["貓"]),
    ("茶杯可能被碰倒", ["碰倒", "倒", "翻倒", "快要倒"]),
    ("牆上有時鐘", ["時鐘", "鐘"]),
    ("牆上有家庭照片", ["家庭照片", "照片", "相片"]),
]


def compute_picture_key_units(text: str) -> int:
    """Approximate key-unit count from patient speech. Conservative enough to catch score hallucinations."""
    text = normalize_text(text)
    count = 0
    # Require more specific combinations for some units to avoid overcounting one vague word.
    if "客廳" in text:
        count += 1
    if _contains_any(text, ["阿公", "老人", "老先生", "爺爺"]) and _contains_any(text, ["相簿", "照片", "沙發", "看東西"]):
        count += 1
    if _contains_any(text, ["老花眼鏡", "眼鏡"]):
        count += 1
    if _contains_any(text, ["茶杯", "茶", "杯子"]):
        count += 1
    if _contains_any(text, ["小孩", "孩子"]) and _contains_any(text, ["積木", "玩"]):
        count += 1
    if _contains_any(text, ["媽媽", "女人", "女兒"]) and _contains_any(text, ["窗邊", "接電話", "講電話", "電話"]):
        count += 1
    if "貓" in text:
        count += 1
    if _contains_any(text, ["碰倒", "快要倒", "翻倒", "倒了", "倒"]):
        count += 1
    if _contains_any(text, ["時鐘", "鐘"]):
        count += 1
    if _contains_any(text, ["家庭照片", "牆上有照片", "幾張照片", "相片"]):
        count += 1
    return max(0, min(10, count))


def delayed_recall_region(text: str) -> str:
    """Get the likely delayed-recall part so initial repetition does not inflate delayed recall score."""
    text = normalize_text(text)
    anchors = ["剛剛那三個", "剛才說的那三個", "剛剛", "剛才", "三個東西", "還記得"]
    pos = -1
    for a in anchors:
        idx = text.rfind(a)
        if idx > pos:
            pos = idx
    return text[pos:] if pos >= 0 else text[-60:]


def immediate_repetition_region(text: str) -> str:
    """Get likely immediate repetition part before orientation/date/place answers."""
    text = normalize_text(text)
    cut_positions = []
    for a in ["今天", "星期", "禮拜", "上午", "下午", "早上", "醫院", "診間", "這裡", "我們在"]:
        idx = text.find(a)
        if idx >= 0:
            cut_positions.append(idx)
    if cut_positions:
        return text[:min(cut_positions)]
    return text[:80]


def count_recalled_items(text: str, items: List[str]) -> List[str]:
    return [item for item in items if item in text]


UNCERTAINTY_PATTERNS = ["好像", "應該", "對不對", "不太清楚", "記不太清楚", "我不確定", "不確定", "還有...", "還有…", "?", "？"]


def has_uncertainty(text: str) -> bool:
    return any(p in normalize_text(text) for p in UNCERTAINTY_PATTERNS)


def cognitive_patient_turns(obj: Dict[str, Any]) -> List[str]:
    """Return [immediate repetition, orientation, delayed recall] patient turns when possible."""
    turns: List[str] = []
    ctx = obj.get("conversation_context")
    if isinstance(ctx, list):
        for turn in ctx:
            if isinstance(turn, dict) and turn.get("speaker") == "patient":
                txt = normalize_text(str(turn.get("text", "")))
                if txt:
                    turns.append(txt)
    if len(turns) >= 3:
        return [turns[0], turns[1], turns[-1]]

    # Fallback: split patient speech events. This is less reliable but better than full-transcript scoring.
    speech_texts = [normalize_text(str(e.get("text", ""))) for e in obj.get("event_script", []) if isinstance(e, dict) and e.get("type") == "speech" and str(e.get("text", "")).strip()]
    if len(speech_texts) >= 3:
        return [speech_texts[0], "".join(speech_texts[1:-1]), speech_texts[-1]]
    if len(speech_texts) == 2:
        return [speech_texts[0], "", speech_texts[1]]
    if len(speech_texts) == 1:
        all_text = speech_texts[0]
        tail = delayed_recall_region(all_text)
        head = all_text[:-len(tail)] if tail and all_text.endswith(tail) else all_text
        return [immediate_repetition_region(head), head[len(immediate_repetition_region(head)):], tail]
    return ["", "", ""]


def score_delayed_recall_turn(turn_text: str, items: List[str]) -> Tuple[int, List[str]]:
    """Score delayed recall only from the delayed-recall answer, with uncertainty penalty."""
    t = normalize_text(turn_text)
    found = count_recalled_items(t, items)
    score = len(found)
    # If the answer is unsure/question-like, do not allow a perfect score.
    if score == 3 and has_uncertainty(t):
        score = 2
    return score, found


def score_orientation_turn(turn_text: str, cdr: str, qwen_score: Any = None) -> int:
    """Turn-aware orientation scoring. Clamp by severity so CDR3 does not look too capable."""
    t = normalize_text(turn_text)
    if isinstance(qwen_score, int):
        score = max(0, min(3, qwen_score))
    else:
        score = 0
        if _contains_any(t, ["星期", "禮拜"]):
            score += 1
        if _contains_any(t, ["上午", "下午", "早上", "晚上", "白天"]):
            score += 1
        if _contains_any(t, ["醫院", "診間", "診所", "這裡"]):
            score += 1
    # Uncertainty should reduce confidence for non-normal groups.
    if cdr not in {"0", "0.5"} and has_uncertainty(t):
        score = min(score, 2)
    if cdr == "3":
        score = min(score, 1)
    return max(0, min(3, score))


def rebuild_cognitive_full_conversation(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Interleave interviewer questions and the three patient answers for cognitive_interview demo audio."""
    if obj.get("task_group") != "cognitive_interview" or obj.get("interaction_type") != "two_person_conversation":
        return obj
    task = obj.get("speech_cognitive_task", {}) if isinstance(obj.get("speech_cognitive_task"), dict) else {}
    prompt = task.get("prompt", "")
    prompts = [x.strip() for x in prompt.split("/") if x.strip()]
    if len(prompts) < 3:
        prompts = COGNITIVE_INTERVIEW_TASK["prompt_sequence"]
    prompts = prompts[:3]
    patient_chunks = cognitive_patient_turns(obj)

    events = []
    context = []
    for idx, q in enumerate(prompts):
        ans = patient_chunks[idx] if idx < len(patient_chunks) else ""
        events.append({"type": "speech", "speaker": "interviewer", "text": q})
        events.append({"type": "pause", "duration_sec": 0.5})
        context.append({"speaker": "interviewer", "text": q})
        if ans:
            events.append({"type": "speech", "speaker": "patient", "text": ans})
            if idx < len(prompts) - 1:
                events.append({"type": "pause", "duration_sec": 0.4})
            context.append({"speaker": "patient", "text": ans})
    obj["full_conversation_event_script"] = events
    obj["conversation_context"] = context
    return obj


def rebuild_simple_full_conversation(obj: Dict[str, Any]) -> Dict[str, Any]:
    """For non-cognitive two-person samples, make demo conversation match the one interviewer prompt then patient event_script."""
    if obj.get("interaction_type") != "two_person_conversation" or obj.get("task_group") == "cognitive_interview":
        return obj
    task = obj.get("speech_cognitive_task", {}) if isinstance(obj.get("speech_cognitive_task"), dict) else {}
    prompt = task.get("prompt") or "你可以多說一點嗎？"
    events = [{"type": "speech", "speaker": "interviewer", "text": prompt}, {"type": "pause", "duration_sec": 0.5}]
    for e in obj.get("event_script", []) if isinstance(obj.get("event_script"), list) else []:
        if isinstance(e, dict):
            copied = dict(e)
            if copied.get("type") == "speech":
                copied["speaker"] = "patient"
            events.append(copied)
    obj["full_conversation_event_script"] = events
    obj["conversation_context"] = [
        {"speaker": "interviewer", "text": prompt},
        {"speaker": "patient", "text": obj.get("spoken_transcript", "")},
    ]
    return obj


def repair_speech_cognitive_task_from_text(obj: Dict[str, Any], cdr: str) -> Dict[str, Any]:
    """Recompute speech_cognitive_task values from patient transcript so Qwen cannot hallucinate scores."""
    if not isinstance(obj, dict):
        return obj
    task = obj.get("speech_cognitive_task")
    if not isinstance(task, dict):
        return obj
    task_group = obj.get("task_group")
    spoken = normalize_text(obj.get("spoken_transcript", ""))
    if task_group == "picture_description":
        total = len(PICTURE_REFERENCE["key_units"])
        mentioned = compute_picture_key_units(spoken)
        incorrect = int(task.get("incorrect_detail_count", 0) or 0)
        incorrect = max(0, min(3, incorrect))
        score = max(0, mentioned - incorrect)
        task.update({
            "task_group": "picture_description",
            "task_type": "picture_description",
            "scoring_method": "key_information_units",
            "expected_key_units": PICTURE_REFERENCE["key_units"],
            "key_units_total": total,
            "key_units_mentioned": mentioned,
            "missing_key_information_count": total - mentioned,
            "incorrect_detail_count": incorrect,
            "score": score,
            "max_score": total,
            "score_interpretation": "speech-based cognitive task score, not clinical MMSE score",
        })
        # Mirror picture counts into impairment_features if present.
        try:
            cog = obj["impairment_features"]["cognitive_task_performance"]
            cog["key_information_units_total"] = total
            cog["key_information_units_mentioned"] = mentioned
            cog["missing_key_information_count"] = total - mentioned
            cog["incorrect_detail_count"] = incorrect
            cog["orientation_error_count"] = 0
        except Exception:
            pass
    elif task_group == "cognitive_interview":
        items = COGNITIVE_INTERVIEW_TASK["expected_items"]
        patient_turns = cognitive_patient_turns(obj)
        imm_text, orientation_text, delayed_text = patient_turns[0], patient_turns[1], patient_turns[2]
        immediate_items = count_recalled_items(imm_text, items)
        delayed_score, delayed_items = score_delayed_recall_turn(delayed_text, items)
        imm_score = len(immediate_items)
        old_orientation = task.get("orientation_correct_count", None)
        if not isinstance(old_orientation, int):
            old_orientation = task.get("subtasks", {}).get("orientation", {}).get("score", None) if isinstance(task.get("subtasks"), dict) else None
        orientation_score = score_orientation_turn(orientation_text, cdr, old_orientation)
        orientation_error_count = 3 - orientation_score
        total_score = imm_score + orientation_score + delayed_score
        task.update({
            "task_group": "cognitive_interview",
            "task_type": "cognitive_interview",
            "scoring_method": "orientation_plus_delayed_recall_speech_score",
            "expected_items": items,
            "orientation_questions": COGNITIVE_INTERVIEW_TASK["orientation_questions"],
            "subtasks": {
                "immediate_repetition": {"score": imm_score, "max_score": 3},
                "orientation": {"score": orientation_score, "max_score": 3},
                "delayed_recall": {"score": delayed_score, "max_score": 3},
            },
            "patient_recalled_items": delayed_items,
            "orientation_correct_count": orientation_score,
            "orientation_error_count": orientation_error_count,
            "score": total_score,
            "max_score": 9,
            "score_interpretation": "speech-based cognitive task score, not clinical MMSE score",
        })
        try:
            obj["impairment_features"]["cognitive_task_performance"]["orientation_error_count"] = orientation_error_count
        except Exception:
            pass
    elif task_group == "real_life_scenarios":
        expected_steps = task.get("expected_steps") if isinstance(task.get("expected_steps"), list) else []
        scenario = obj.get("scenario", "")
        step_patterns = {
            "medicine_routine": [["起床"], ["早餐", "早飯", "吃東西"], ["藥袋", "藥盒", "藥"], ["吃藥", "把藥吃"]],
            "daily_life": [["起床", "醒"], ["刷牙", "洗臉", "盥洗"], ["早餐", "早飯", "吃一點"], ["公園", "電視", "散步", "出門", "活動"]],
            "family": [["家人", "全家", "太太", "兒子", "女兒"], ["餐廳", "家裡", "家"], ["吃飯", "聊天", "點菜"], ["禮拜", "昨天", "上次", "然後", "後來"]],
            "home": [["家", "房間", "客廳"], ["早餐", "午餐", "晚餐", "吃飯"], ["電視", "散步", "公園", "休息"], ["早上", "下午", "傍晚", "然後"]],
            "market": [["出門", "市場"], ["買", "菜", "東西"], ["付錢", "付款", "錢"], ["回家", "回來"]],
        }
        patterns = step_patterns.get(scenario, [])
        steps_mentioned = sum(1 for group in patterns if _contains_any(spoken, group)) if patterns else int(task.get("steps_mentioned", 0) or 0)
        sequence_error = int(task.get("sequence_error_count", 0) or 0)
        memory_gap = count_occurrences(spoken, MEMORY_GAP_PATTERNS)
        max_score = int(task.get("max_score", len(expected_steps) or 4) or 4)
        steps_mentioned = max(0, min(max_score, steps_mentioned))
        sequence_error = max(0, min(2, sequence_error))
        score = max(0, min(max_score, steps_mentioned - sequence_error))
        task.update({
            "expected_steps": expected_steps,
            "steps_mentioned": steps_mentioned,
            "sequence_error_count": sequence_error,
            "memory_gap_count": memory_gap,
            "score": score,
            "max_score": max_score,
            "score_interpretation": "speech-based cognitive task score, not clinical MMSE score",
        })
    obj["speech_cognitive_task"] = task
    return obj


def light_repair_sample(obj: Dict[str, Any], cdr: str = "") -> Dict[str, Any]:
    """
    Conservative repair before validation.
    V6 uses event_script as the source of truth because Qwen often places
    markers incorrectly in marked_transcript.
    """
    obj.setdefault("repair_log", {})
    obj = strip_acoustic_condition_fields(obj)
    obj = convert_known_simplified_text(obj)
    before_spoken = obj.get("spoken_transcript")
    before_marked = obj.get("marked_transcript")
    before_features = json.dumps(obj.get("impairment_features", {}), ensure_ascii=False, sort_keys=True)
    obj = repair_invalid_filled_pause_events(obj)
    obj = ensure_minimum_disfluency_events(obj, cdr)
    obj = repair_marked_and_spoken_from_events(obj)
    obj = recompute_event_stats_and_acoustic(obj)
    obj = repair_feature_counts_from_text(obj, cdr)
    obj = repair_speech_cognitive_task_from_text(obj, cdr)
    obj = recompute_event_stats_and_acoustic(obj)
    obj = repair_marked_and_spoken_from_events(obj)
    obj = repair_speech_cognitive_task_from_text(obj, cdr)
    obj = rebuild_cognitive_full_conversation(obj)
    obj = rebuild_simple_full_conversation(obj)
    obj = strip_acoustic_condition_fields(obj)
    obj["repair_log"]["spoken_transcript_rebuilt_from_event_script"] = obj.get("spoken_transcript") != before_spoken
    obj["repair_log"]["marked_transcript_rebuilt_from_event_script"] = obj.get("marked_transcript") != before_marked
    obj["repair_log"]["feature_counts_repaired"] = json.dumps(obj.get("impairment_features", {}), ensure_ascii=False, sort_keys=True) != before_features
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
        "sample_id", "cdr_level", "cdr_label", "task_group", "scenario", "task_type", "interaction_type",
        "picture_reference", "speaker", "split",
        "speech_cognitive_task", "conversation_context", "spoken_transcript", "marked_transcript", "event_script",
        "full_conversation_event_script", "audio_outputs", "event_stats", "impairment_features", "repair_log"
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

    if obj.get("task_group") != expected.get("task_group"):
        errors.append(f"task_group mismatch: expected {expected.get('task_group')}, got {obj.get('task_group')}")

    if obj.get("scenario") != expected["scenario"]:
        errors.append("scenario mismatch")

    if obj.get("task_type") != expected.get("task_type"):
        errors.append(f"task_type mismatch: expected {expected.get('task_type')}, got {obj.get('task_type')}")

    if obj.get("split") != expected["split"]:
        errors.append("split mismatch")

    if "acoustic_condition" in obj or "acoustic_condition_info" in obj:
        errors.append("acoustic_condition fields must not be present; audio effects are handled during synthesis")

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

    mainland_hits = contains_mainland_phrasing(chinese_only_text)
    if mainland_hits:
        errors.append(f"Mainland/non-Taiwan phrasing found: {', '.join(mainland_hits[:10])}")

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
            patient_compact = re.sub(r"\s+", "", normalize_text(patient_turn_text))
            spoken_compact = re.sub(r"\s+", "", normalize_text(spoken))
            if patient_turn_text and patient_compact != spoken_compact:
                errors.append("spoken_transcript should match the joined patient answers in conversation_context")
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
            if expected.get("task_group") == "cognitive_interview":
                speech_speakers_order = [e.get("speaker") for e in full_conv_events if isinstance(e, dict) and e.get("type") == "speech"]
                interviewer_count = speech_speakers_order.count("interviewer")
                patient_count = speech_speakers_order.count("patient")
                if interviewer_count < 3:
                    errors.append("cognitive full_conversation_event_script must have three separate interviewer prompts")
                if patient_count < 2:
                    errors.append("cognitive full_conversation_event_script must interleave patient responses")
                if speech_speakers_order[:3] == ["interviewer", "interviewer", "interviewer"]:
                    errors.append("cognitive full_conversation_event_script puts all interviewer questions first; must be interleaved")
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

    # Speech cognitive task validation
    task = obj.get("speech_cognitive_task", {})
    if not isinstance(task, dict):
        errors.append("speech_cognitive_task must be object")
        task = {}
    else:
        if task.get("task_group") != expected.get("task_group"):
            errors.append("speech_cognitive_task.task_group mismatch")
        if task.get("task_type") != expected.get("task_type"):
            errors.append("speech_cognitive_task.task_type mismatch")
        if not isinstance(task.get("prompt"), str) or not task.get("prompt", "").strip():
            errors.append("speech_cognitive_task.prompt missing")
        if task.get("score_interpretation") != "speech-based cognitive task score, not clinical MMSE score":
            errors.append("speech_cognitive_task.score_interpretation must clarify it is not clinical MMSE")
        max_score = task.get("max_score")
        score = task.get("score")
        if not isinstance(max_score, int) or max_score <= 0:
            errors.append("speech_cognitive_task.max_score must be positive integer")
            max_score = 1
        if not isinstance(score, int):
            errors.append("speech_cognitive_task.score must be integer")
            score = 0
        else:
            lo, hi = expected_score_range_for_cdr(cdr, int(max_score))
            if not (lo <= score <= hi):
                errors.append(f"speech_cognitive_task.score={score} outside CDR envelope [{lo}, {hi}]")
        if expected.get("task_group") == "picture_description":
            if task.get("max_score") != len(PICTURE_REFERENCE["key_units"]):
                errors.append("picture speech_cognitive_task.max_score must equal picture key units total")
            for f in ["expected_key_units", "key_units_total", "key_units_mentioned", "missing_key_information_count"]:
                if f not in task:
                    errors.append(f"picture speech_cognitive_task missing {f}")
            if isinstance(task.get("key_units_mentioned"), int) and isinstance(task.get("missing_key_information_count"), int):
                if task["key_units_mentioned"] + task["missing_key_information_count"] != len(PICTURE_REFERENCE["key_units"]):
                    errors.append("picture key_units_mentioned + missing_key_information_count must equal total")
            try:
                if cog.get("orientation_error_count", 0) != 0:
                    errors.append("picture_description must have orientation_error_count = 0")
            except Exception:
                pass
        elif expected.get("task_group") == "cognitive_interview":
            if task.get("max_score") != 9:
                errors.append("cognitive_interview max_score must be 9")
            for f in ["expected_items", "orientation_questions", "subtasks", "patient_recalled_items", "orientation_correct_count", "orientation_error_count"]:
                if f not in task:
                    errors.append(f"cognitive_interview speech_cognitive_task missing {f}")
            if cdr in {"0", "0.5"} and isinstance(task.get("orientation_error_count"), int) and task.get("orientation_error_count") > 0:
                errors.append("CDR 0/0.5 cognitive_interview should not have orientation errors")
        elif expected.get("task_group") == "real_life_scenarios":
            for f in ["expected_steps", "steps_mentioned", "sequence_error_count", "memory_gap_count"]:
                if f not in task:
                    errors.append(f"real_life speech_cognitive_task missing {f}")

        # Strict score-consistency checks after Python-side recomputation.
        if expected.get("task_group") == "picture_description":
            actual_mentioned = compute_picture_key_units(spoken)
            if task.get("key_units_mentioned") != actual_mentioned:
                errors.append(f"picture key_units_mentioned={task.get('key_units_mentioned')} does not match transcript-derived {actual_mentioned}")
            expected_score = max(0, actual_mentioned - int(task.get("incorrect_detail_count", 0) or 0))
            if task.get("score") != expected_score:
                errors.append(f"picture score={task.get('score')} does not match computed {expected_score}")
        elif expected.get("task_group") == "cognitive_interview" and isinstance(task.get("subtasks"), dict):
            subtasks = task.get("subtasks", {})
            sub_sum = sum(int(subtasks.get(name, {}).get("score", 0) or 0) for name in ["immediate_repetition", "orientation", "delayed_recall"])
            if task.get("score") != sub_sum:
                errors.append(f"cognitive_interview score={task.get('score')} does not equal subtask sum {sub_sum}")
            delayed_turn = cognitive_patient_turns(obj)[2]
            expected_delayed_score, delayed_items = score_delayed_recall_turn(delayed_turn, COGNITIVE_INTERVIEW_TASK["expected_items"])
            delayed_score = int(subtasks.get("delayed_recall", {}).get("score", 0) or 0)
            if delayed_score != expected_delayed_score:
                errors.append(f"delayed_recall score={delayed_score} does not match turn-derived delayed recall score {expected_delayed_score}")
            if task.get("patient_recalled_items") != delayed_items:
                errors.append("patient_recalled_items must match turn-derived delayed recall items")
            # Severity-specific cognitive interview sanity checks.
            if cdr == "1" and delayed_score == 0 and computed_stats["long_pause_count"] >= 2:
                errors.append("CDR 1 cognitive_interview is too severe: delayed_recall=0 with 2+ long pauses")
            if cdr == "1" and computed_stats["total_pause_duration_sec"] > 4.5:
                errors.append("CDR 1 cognitive_interview total pause duration is too high")
            if cdr == "3" and int(task.get("orientation_correct_count", 0) or 0) > 1:
                errors.append("CDR 3 cognitive_interview orientation score should be 0-1")
        elif expected.get("task_group") == "real_life_scenarios":
            if isinstance(task.get("steps_mentioned"), int) and isinstance(task.get("score"), int):
                expected_score = max(0, min(int(task.get("max_score", 4) or 4), int(task.get("steps_mentioned", 0) or 0) - int(task.get("sequence_error_count", 0) or 0)))
                if task.get("score") != expected_score:
                    errors.append(f"real_life score={task.get('score')} does not match computed {expected_score}")

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
    int_in_range(acoustic.get("filled_pause_count"), 0, rules["filled_pause"][1], "acoustic_fluency.filled_pause_count", errors)
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

    # Linguistic/cognitive feature ranges: use evidence-based counts, so validate upper bounds strictly
    # without forcing every sample to contain every marker.
    int_in_range(ling.get("word_finding_count"), 0, rules["word_finding"][1], "linguistic_impairment.word_finding_count", errors)
    int_in_range(ling.get("word_repetition_count"), 0, rules["repetition"][1], "linguistic_impairment.word_repetition_count", errors)
    int_in_range(ling.get("sentence_fragment_count"), 0, rules["fragment"][1], "linguistic_impairment.sentence_fragment_count", errors)
    int_in_range(cog.get("memory_gap_count"), 0, rules["memory_gap"][1], "cognitive_task_performance.memory_gap_count", errors)
    int_in_range(cog.get("orientation_error_count"), 0, rules["orientation"][1], "cognitive_task_performance.orientation_error_count", errors)
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



# ---------------------------------------------------------------------
# V10 targeted overrides
# ---------------------------------------------------------------------
# These wrappers keep the V9 generator intact but add the remaining fixes found
# during manual review of generated samples:
# 1) reject interviewer prompt leakage inside patient speech,
# 2) stricter medicine-routine step scoring,
# 3) minimum length for normal real-life answers,
# 4) reject picture-reference copy/prompt leakage,
# 5) keep CDR3 daily-life drift related to the morning routine.

PATIENT_SPEECH_BANNED_PHRASES = [
    "請你看", "請你", "告訴我", "你可以", "我等一下會請你",
    "你知道今天", "剛剛那三個東西", "請描述", "請說明",
]

PICTURE_COPY_LEAKAGE_PHRASES = [
    "請你看這張圖",
    "一個客廳裡的家庭場景",
    "積木散在地毯上",
    "一隻貓跳到茶几旁邊",
    "好像快要碰倒茶杯",
]

DAILY_LIFE_OFF_TASK_DRIFT_PHRASES = [
    "公車", "捷運", "計程車", "車站", "火車站", "搭車", "坐車",
    "菜市場", "買菜", "買東西", "醫院", "診間",
]

# For CDR3 daily-life, these words are allowed if the sample is explicitly about going out.
DAILY_LIFE_ON_TASK_WORDS = [
    "起床", "起來", "刷牙", "洗臉", "洗澡", "廁所", "早餐", "吃", "喝",
    "牛奶", "蛋餅", "粥", "飯", "藥", "看電視", "沙發", "休息", "公園", "散步",
]

_original_repair_speech_cognitive_task_from_text = repair_speech_cognitive_task_from_text
_original_validate_sample = validate_sample


def patient_texts_for_leak_check(obj: Dict[str, Any]) -> List[str]:
    """Collect only patient-side text, never interviewer prompts."""
    texts: List[str] = []
    if isinstance(obj.get("spoken_transcript"), str):
        texts.append(obj["spoken_transcript"])
    for turn in obj.get("conversation_context") or []:
        if isinstance(turn, dict) and turn.get("speaker") == "patient" and isinstance(turn.get("text"), str):
            texts.append(turn["text"])
    for field in ["event_script", "full_conversation_event_script"]:
        for e in obj.get(field) or []:
            if isinstance(e, dict) and e.get("type") == "speech" and e.get("speaker") == "patient" and isinstance(e.get("text"), str):
                texts.append(e["text"])
    return texts


def contains_patient_prompt_leakage(obj: Dict[str, Any]) -> List[str]:
    hits: List[str] = []
    for txt in patient_texts_for_leak_check(obj):
        for phrase in PATIENT_SPEECH_BANNED_PHRASES:
            if phrase in txt:
                hits.append(phrase)
    return sorted(set(hits))


def compute_real_life_steps_v10(text: str, scenario: str) -> Tuple[int, int, int]:
    """
    Return (steps_mentioned, sequence_error_count, memory_gap_count).
    This is stricter than V9, especially for medicine_routine:
    - 拿藥 is not counted as 看藥袋/藥盒.
    - 拿藥 is not counted as 吃藥.
    - 看藥袋/看藥盒 and 吃藥 are separate steps.
    """
    t = normalize_text(text)

    if scenario == "medicine_routine":
        step_patterns = [
            ["起床", "起來", "早上起來", "醒來"],
            ["吃早餐", "早餐", "早飯", "蛋餅", "牛奶", "粥", "吃東西"],
            ["看藥袋", "看藥盒", "確認藥", "確認一下藥", "看一下藥", "藥袋上", "藥盒裡"],
            ["吃藥", "服藥", "把藥吃", "把藥吃了", "吞藥", "吃了藥"],
        ]
    elif scenario == "daily_life":
        step_patterns = [
            ["起床", "起來", "醒來"],
            ["刷牙", "洗臉", "洗澡", "盥洗", "去廁所", "上廁所"],
            ["吃早餐", "早餐", "早飯", "蛋餅", "牛奶", "粥", "吃東西"],
            ["看電視", "散步", "公園", "出門", "休息", "整理", "買菜", "活動"],
        ]
    elif scenario == "family":
        step_patterns = [
            ["爸爸", "媽媽", "妹妹", "哥哥", "姐姐", "家人", "全家", "阿公", "阿嬤", "兒子", "女兒"],
            ["餐廳", "家裡", "老家", "家", "板橋", "外面"],
            ["吃飯", "聊天", "火鍋", "聚餐", "看電視", "散步"],
            ["上禮拜", "昨天", "晚上", "早上", "中午", "吃完", "之後", "最後", "回家"],
        ]
    elif scenario == "home":
        step_patterns = [
            ["整理", "家務", "房間", "床", "洗衣", "曬", "打掃"],
            ["吃飯", "早餐", "午餐", "晚餐", "煮"],
            ["看電視", "散步", "公園", "休息", "聊天"],
            ["早上", "下午", "晚上", "之後", "最後", "傍晚"],
        ]
    elif scenario == "market":
        step_patterns = [
            ["出門", "從家裡", "走到", "去市場", "菜市場"],
            ["買", "青菜", "肉", "水果", "東西"],
            ["付錢", "錢包", "結帳", "付款"],
            ["回家", "提回來", "拿回家"],
        ]
    else:
        step_patterns = []

    steps = sum(1 for pats in step_patterns if any(p in t for p in pats))
    memory_gap = count_occurrences(t, MEMORY_GAP_PATTERNS)

    sequence_error = 0
    if any(p in t for p in ["先吃藥再吃早餐", "吃藥之後才吃早餐", "先吃藥，然後吃早餐"]):
        sequence_error += 1
    if any(p in t for p in ["忘記順序", "順序不太記得", "不知道先後", "先後忘了"]):
        sequence_error += 1
    # If explicit confusion about whether an activity happened, count one sequence/recall error.
    if any(p in t for p in ["忘記有沒有", "不記得有沒有", "不知道有沒有"]):
        sequence_error += 1

    return max(0, min(4, steps)), max(0, min(3, sequence_error)), max(0, min(5, memory_gap))


def repair_speech_cognitive_task_from_text(obj: Dict[str, Any], cdr: str) -> Dict[str, Any]:
    """V10 wrapper: keep V9 repair, then apply stricter real-life step scoring."""
    obj = _original_repair_speech_cognitive_task_from_text(obj, cdr)
    if not isinstance(obj, dict) or obj.get("task_group") != "real_life_scenarios":
        return obj

    scenario = str(obj.get("scenario", ""))
    plan = REAL_LIFE_TASK_BANK.get(scenario)
    if not plan:
        return obj

    spoken = normalize_text(obj.get("spoken_transcript", ""))
    steps, sequence_error, memory_gap = compute_real_life_steps_v10(spoken, scenario)
    max_score = int(plan.get("max_score", 4))
    score = max(0, min(max_score, steps - sequence_error))

    task = obj.get("speech_cognitive_task") if isinstance(obj.get("speech_cognitive_task"), dict) else {}
    task.update({
        "task_group": "real_life_scenarios",
        "task_type": plan["task_type"],
        "prompt": plan["prompt"],
        "scoring_method": "expected_steps_and_sequence",
        "score": score,
        "max_score": max_score,
        "score_interpretation": "speech-based cognitive task score, not clinical MMSE score",
        "expected_steps": plan["expected_steps"],
        "steps_mentioned": steps,
        "sequence_error_count": sequence_error,
        "memory_gap_count": memory_gap,
    })
    obj["speech_cognitive_task"] = task

    try:
        obj["impairment_features"]["cognitive_task_performance"]["memory_gap_count"] = clamp_int(memory_gap, 0, CDR_RULES[cdr]["memory_gap"][1])
    except Exception:
        pass

    return obj


def is_picture_reference_copy_like(text: str) -> bool:
    t = normalize_text(text)
    if any(p in t for p in PICTURE_COPY_LEAKAGE_PHRASES):
        return True
    # Copying the picture reference tends to mention these exact written-description chunks.
    copied_chunks = [
        "客廳場景", "阿公坐在沙發上看相簿", "旁邊有一副老花眼鏡和一杯茶",
        "媽媽站在窗邊接電話", "牆上有時鐘和家庭照片", "窗外看起來像是下午",
    ]
    return sum(1 for p in copied_chunks if p in t) >= 4


def daily_life_drift_is_off_task(text: str, cdr: str) -> bool:
    """For severe daily-life samples, allow drift but keep it related to morning routine/home."""
    if cdr != "3":
        return False
    t = normalize_text(text)
    if not any(p in t for p in DAILY_LIFE_OFF_TASK_DRIFT_PHRASES):
        return False
    # 公園/散步 is acceptable for planned activity; transport/market/clinic drift is not.
    bad = [p for p in DAILY_LIFE_OFF_TASK_DRIFT_PHRASES if p in t and p not in {"公園"}]
    return len(bad) > 0


def validate_sample(obj: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """V10 validation wrapper adding the remaining strict rules."""
    ok, errors = _original_validate_sample(obj, expected)
    cdr = str(expected.get("cdr_level"))
    task_group = expected.get("task_group")
    scenario = expected.get("scenario")
    spoken = normalize_text(obj.get("spoken_transcript", "")) if isinstance(obj, dict) else ""

    # 1) No interviewer prompt leakage in patient speech.
    leaks = contains_patient_prompt_leakage(obj)
    if leaks:
        errors.append("patient speech contains interviewer prompt leakage: " + ", ".join(leaks))

    # 2) Picture description must not copy prompt/reference wording.
    if task_group == "picture_description":
        if is_picture_reference_copy_like(spoken):
            errors.append("picture_description appears to copy prompt/reference wording too closely")
        if spoken.startswith("請你") or "請你看這張圖" in spoken:
            errors.append("picture_description patient speech starts with interviewer prompt")

    # 3) Real-life stricter scoring consistency using V10 rules.
    if task_group == "real_life_scenarios":
        task = obj.get("speech_cognitive_task", {}) if isinstance(obj.get("speech_cognitive_task"), dict) else {}
        steps, sequence_error, memory_gap = compute_real_life_steps_v10(spoken, str(scenario))
        max_score = int(task.get("max_score", 4) or 4)
        expected_score = max(0, min(max_score, steps - sequence_error))
        if task.get("steps_mentioned") != steps:
            errors.append(f"real_life steps_mentioned={task.get('steps_mentioned')} does not match V10 step detector {steps}")
        if task.get("sequence_error_count") != sequence_error:
            errors.append(f"real_life sequence_error_count={task.get('sequence_error_count')} does not match V10 detector {sequence_error}")
        if task.get("score") != expected_score:
            errors.append(f"real_life score={task.get('score')} does not match V10 computed {expected_score}")

        # CDR 0 real-life should be complete enough to be useful; avoid one-line normal samples.
        if cdr == "0" and len(re.sub(r"\s+", "", spoken)) < 35:
            errors.append("CDR 0 real-life sample is too short; require at least 35 Chinese characters")

        # CDR 3 daily-life drift should stay related to morning/home routine.
        if scenario == "daily_life" and daily_life_drift_is_off_task(spoken, cdr):
            errors.append("CDR 3 daily_life_narrative has off-task transport/market/clinic drift")

    # 4) Medicine-specific: 拿藥 alone cannot count as 看藥袋 or 吃藥.
    if scenario == "medicine_routine" and "拿藥" in spoken:
        task = obj.get("speech_cognitive_task", {}) if isinstance(obj.get("speech_cognitive_task"), dict) else {}
        if "看藥袋" not in spoken and "看藥盒" not in spoken and "確認藥" not in spoken and task.get("steps_mentioned", 0) >= 3:
            errors.append("medicine_routine counts 拿藥 as 看藥袋/藥盒; require explicit 看藥袋/看藥盒/確認藥")
        if "吃藥" not in spoken and "服藥" not in spoken and "把藥吃" not in spoken and task.get("score", 0) >= 3:
            errors.append("medicine_routine counts 拿藥 as 吃藥; require explicit 吃藥/服藥")

    # Deduplicate while preserving order.
    deduped: List[str] = []
    for e in errors:
        if e not in deduped:
            deduped.append(e)
    return len(deduped) == 0, deduped



# -----------------------------
# V11 targeted patch: make CDR 3 non-cognitive tasks generatable
# -----------------------------
# Why:
# V10 was too strict for CDR 3 picture/real-life tasks. Qwen often produced
# fragmented but keyword-rich descriptions. The raw keyword scorer then gave
# high scores, so picture_description and real_life_scenarios were rejected,
# leaving only cognitive_interview samples for CDR 3.
#
# V11 changes:
# - For CDR 3 picture: count "functional key information", not isolated object words.
#   Fragmented uncertain descriptions are capped at 4/10 so they can fit severe CDR.
# - For CDR 3 real-life: count reliable steps only; uncertain/forgotten steps do not
#   become full credit.
# - Relax CDR 3 non-cognitive pause-count edge cases slightly.
# - Repair missing duration_sec for sigh/breath/cough events.
# - Repair missing top-level split from speaker.split when possible.

_original_v10_light_repair_sample = light_repair_sample
_original_v10_repair_speech_cognitive_task_from_text = repair_speech_cognitive_task_from_text
_original_v10_validate_sample = validate_sample


def repair_nonverbal_event_durations_v11(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Add default duration_sec for nonverbal events if Qwen forgot them."""
    defaults = {"sigh": 0.9, "breath": 0.45, "cough": 0.55}

    def fix(events):
        if not isinstance(events, list):
            return events
        for e in events:
            if not isinstance(e, dict):
                continue
            typ = e.get("type")
            if typ in defaults:
                try:
                    dur = float(e.get("duration_sec"))
                except Exception:
                    dur = 0.0
                if dur <= 0:
                    e["duration_sec"] = defaults[typ]
        return events

    if isinstance(obj, dict):
        obj["event_script"] = fix(obj.get("event_script"))
        obj["full_conversation_event_script"] = fix(obj.get("full_conversation_event_script"))
    return obj


def cdr3_fragmentation_strength(text: str) -> int:
    t = normalize_text(text)
    score = 0
    score += count_occurrences(t, ["忘記", "想不起來", "不知道", "不確定", "不記得", "記不太清楚"])
    score += count_occurrences(t, ["那個", "這個", "呃", "嗯", "好像", "應該"])
    score += t.count("...") + t.count("…")
    return score


def compute_picture_key_units_v11(text: str, cdr: str) -> int:
    """
    CDR 0–2: use the existing V10 key-unit logic.
    CDR 3: do not give full credit for isolated keyword listing.
    Severe picture descriptions can mention objects but still fail to build
    complete visual propositions, so cap functional score.
    """
    raw = compute_picture_key_units(text)
    if str(cdr) != "3":
        return raw

    t = normalize_text(text)
    frag = cdr3_fragmentation_strength(t)

    # Count only the clearest complete propositions for severe samples.
    complete = 0
    if "客廳" in t and _contains_any(t, ["看到", "這裡", "裡面", "圖"]):
        complete += 1
    if _contains_any(t, ["阿公", "老人", "老先生", "爺爺"]) and _contains_any(t, ["看相簿", "看照片", "看東西"]):
        complete += 1
    if _contains_any(t, ["小孩", "孩子"]) and "積木" in t and _contains_any(t, ["玩", "地上"]):
        complete += 1
    if _contains_any(t, ["媽媽", "女人", "阿姨"]) and _contains_any(t, ["電話", "窗邊"]):
        complete += 1
    if "貓" in t and _contains_any(t, ["茶几", "杯", "倒", "碰"]):
        complete += 1
    if _contains_any(t, ["時鐘", "照片", "相片"]):
        complete += 1

    # If it is fragmented/uncertain, score should represent usable information,
    # not just object names.
    if frag >= 5:
        return max(0, min(4, complete))
    return max(0, min(4, raw))


def compute_real_life_steps_v11(text: str, scenario: str, cdr: str) -> Tuple[int, int, int]:
    """
    V11 CDR 3 fix:
    V10 counted mentioned activities even when the patient said they forgot,
    were unsure, or could not sequence them. For severe CDR we count reliable
    steps only.
    """
    raw_steps, sequence_error, memory_gap = compute_real_life_steps_v10(text, scenario)
    if str(cdr) != "3":
        return raw_steps, sequence_error, memory_gap

    t = normalize_text(text)
    uncertainty = cdr3_fragmentation_strength(t)

    # For CDR 3, a step mentioned inside "好像/忘記/不知道/不記得" should not
    # count like a confident completed step.
    if scenario in {"daily_life", "medicine_routine", "home_activity_recall", "home"}:
        if memory_gap >= 2 or uncertainty >= 5:
            reliable_steps = min(raw_steps, 1)
        else:
            reliable_steps = min(raw_steps, 2)
    else:
        reliable_steps = min(raw_steps, 1 if memory_gap >= 2 else 2)

    # If Qwen lists many steps but also says it does not remember, treat the
    # excess as sequence/recall failure.
    sequence_error = max(sequence_error, max(0, raw_steps - reliable_steps))
    return max(0, min(4, reliable_steps)), max(0, min(3, sequence_error)), max(0, min(5, memory_gap))


def repair_speech_cognitive_task_from_text(obj: Dict[str, Any], cdr: str) -> Dict[str, Any]:
    obj = _original_v10_repair_speech_cognitive_task_from_text(obj, cdr)
    if not isinstance(obj, dict):
        return obj

    task_group = obj.get("task_group")
    spoken = normalize_text(obj.get("spoken_transcript", ""))

    if task_group == "picture_description":
        total = len(PICTURE_REFERENCE["key_units"])
        mentioned = compute_picture_key_units_v11(spoken, cdr)
        task = obj.get("speech_cognitive_task") if isinstance(obj.get("speech_cognitive_task"), dict) else {}
        incorrect = int(task.get("incorrect_detail_count", 0) or 0)
        incorrect = max(0, min(3, incorrect))
        score = max(0, mentioned - incorrect)
        if str(cdr) == "3":
            score = min(score, 4)
            mentioned = min(mentioned, 4)

        task.update({
            "task_group": "picture_description",
            "task_type": "picture_description",
            "scoring_method": "key_information_units",
            "expected_key_units": PICTURE_REFERENCE["key_units"],
            "key_units_total": total,
            "key_units_mentioned": mentioned,
            "missing_key_information_count": total - mentioned,
            "incorrect_detail_count": incorrect,
            "score": score,
            "max_score": total,
            "score_interpretation": "speech-based cognitive task score, not clinical MMSE score",
        })
        obj["speech_cognitive_task"] = task
        try:
            cog = obj["impairment_features"]["cognitive_task_performance"]
            cog["key_information_units_total"] = total
            cog["key_information_units_mentioned"] = mentioned
            cog["missing_key_information_count"] = total - mentioned
            cog["incorrect_detail_count"] = incorrect
            cog["orientation_error_count"] = 0
        except Exception:
            pass

    elif task_group == "real_life_scenarios":
        scenario = str(obj.get("scenario", ""))
        plan = REAL_LIFE_TASK_BANK.get(scenario)
        if plan:
            steps, sequence_error, memory_gap = compute_real_life_steps_v11(spoken, scenario, cdr)
            max_score = int(plan.get("max_score", 4))
            score = max(0, min(max_score, steps - sequence_error))
            if str(cdr) == "3":
                score = min(score, 1)

            task = obj.get("speech_cognitive_task") if isinstance(obj.get("speech_cognitive_task"), dict) else {}
            task.update({
                "task_group": "real_life_scenarios",
                "task_type": plan["task_type"],
                "prompt": plan["prompt"],
                "scoring_method": "expected_steps_and_sequence",
                "score": score,
                "max_score": max_score,
                "score_interpretation": "speech-based cognitive task score, not clinical MMSE score",
                "expected_steps": plan["expected_steps"],
                "steps_mentioned": steps,
                "sequence_error_count": sequence_error,
                "memory_gap_count": memory_gap,
            })
            obj["speech_cognitive_task"] = task
            try:
                obj["impairment_features"]["cognitive_task_performance"]["memory_gap_count"] = clamp_int(memory_gap, 0, CDR_RULES[str(cdr)]["memory_gap"][1])
            except Exception:
                pass

    return obj


def light_repair_sample(obj: Dict[str, Any], cdr: str = "") -> Dict[str, Any]:
    if isinstance(obj, dict) and "split" not in obj:
        sp = obj.get("speaker", {})
        if isinstance(sp, dict) and sp.get("split"):
            obj["split"] = sp.get("split")
    obj = repair_nonverbal_event_durations_v11(obj)
    obj = _original_v10_light_repair_sample(obj, cdr)
    obj = repair_nonverbal_event_durations_v11(obj)
    obj = recompute_event_stats_and_acoustic(obj)
    obj = repair_speech_cognitive_task_from_text(obj, cdr)
    obj = repair_marked_and_spoken_from_events(obj)
    obj = strip_acoustic_condition_fields(obj)
    return obj


def validate_sample(obj: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, List[str]]:
    ok, errors = _original_v10_validate_sample(obj, expected)
    cdr = str(expected.get("cdr_level"))
    task_group = expected.get("task_group")
    scenario = str(expected.get("scenario"))
    spoken = normalize_text(obj.get("spoken_transcript", "")) if isinstance(obj, dict) else ""

    filtered: List[str] = []
    for e in errors:
        # V10 real-life consistency errors use V10 raw step detector; V11 uses reliable-step
        # scoring for CDR 3, so remove those and re-check below.
        if cdr == "3" and task_group == "real_life_scenarios" and e.startswith("real_life "):
            continue

        # CDR 3 non-cognitive speech may legitimately have 2 pauses or slightly more
        # than 12 short fragments. Do not let this alone block all non-cog CDR 3.
        if cdr == "3" and "acoustic_fluency.pause_count=" in e and "outside allowed range [3, 12]" in e:
            m = re.search(r"pause_count=(\d+)", e)
            n = int(m.group(1)) if m else None
            if task_group == "real_life_scenarios" and n is not None and 2 <= n <= 12:
                continue
            if task_group == "picture_description" and n is not None and 3 <= n <= 16:
                continue

        filtered.append(e)

    errors = filtered

    # V11 real-life consistency check.
    if task_group == "real_life_scenarios":
        task = obj.get("speech_cognitive_task", {}) if isinstance(obj.get("speech_cognitive_task"), dict) else {}
        steps, sequence_error, memory_gap = compute_real_life_steps_v11(spoken, scenario, cdr)
        max_score = int(task.get("max_score", 4) or 4)
        expected_score = max(0, min(max_score, steps - sequence_error))
        if cdr == "3":
            expected_score = min(expected_score, 1)

        if task.get("steps_mentioned") != steps:
            errors.append(f"real_life steps_mentioned={task.get('steps_mentioned')} does not match V11 reliable-step detector {steps}")
        if task.get("sequence_error_count") != sequence_error:
            errors.append(f"real_life sequence_error_count={task.get('sequence_error_count')} does not match V11 detector {sequence_error}")
        if task.get("score") != expected_score:
            errors.append(f"real_life score={task.get('score')} does not match V11 computed {expected_score}")

    # V11 picture consistency check for CDR 3 functional key information.
    if task_group == "picture_description" and cdr == "3":
        task = obj.get("speech_cognitive_task", {}) if isinstance(obj.get("speech_cognitive_task"), dict) else {}
        mentioned = compute_picture_key_units_v11(spoken, cdr)
        incorrect = int(task.get("incorrect_detail_count", 0) or 0)
        expected_score = max(0, min(4, mentioned - incorrect))
        if task.get("key_units_mentioned") != mentioned:
            errors.append(f"picture key_units_mentioned={task.get('key_units_mentioned')} does not match V11 functional detector {mentioned}")
        if task.get("missing_key_information_count") != 10 - mentioned:
            errors.append(f"picture missing_key_information_count={task.get('missing_key_information_count')} does not match V11 {10 - mentioned}")
        if task.get("score") != expected_score:
            errors.append(f"picture score={task.get('score')} does not match V11 computed {expected_score}")

    # Deduplicate while preserving order.
    deduped: List[str] = []
    for e in errors:
        if e not in deduped:
            deduped.append(e)
    return len(deduped) == 0, deduped




# -----------------------------
# V11.1 edge-case patch for CDR 3 picture retries
# -----------------------------
_original_v11_light_repair_sample = light_repair_sample
_original_v11_validate_sample = validate_sample


def repair_event_duration_ranges_v11(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp small duration mistakes that are safe to repair."""
    def fix(events, patient_only: bool = False):
        if not isinstance(events, list):
            return events
        for e in events:
            if not isinstance(e, dict):
                continue
            typ = e.get("type")
            if typ == "speech" and patient_only:
                e["speaker"] = "patient"
            if typ == "pause":
                try:
                    d = float(e.get("duration_sec", 0.8))
                except Exception:
                    d = 0.8
                e["duration_sec"] = round(max(0.3, min(1.5, d)), 2)
            elif typ == "long_pause":
                try:
                    d = float(e.get("duration_sec", 2.0))
                except Exception:
                    d = 2.0
                e["duration_sec"] = round(max(1.5, min(4.5, d)), 2)
            elif typ == "silence":
                try:
                    d = float(e.get("duration_sec", 3.5))
                except Exception:
                    d = 3.5
                e["duration_sec"] = round(max(1.5, min(7.0, d)), 2)
        return events

    if isinstance(obj, dict):
        obj["event_script"] = fix(obj.get("event_script"), patient_only=True)
        obj["full_conversation_event_script"] = fix(obj.get("full_conversation_event_script"), patient_only=False)
    return obj


def light_repair_sample(obj: Dict[str, Any], cdr: str = "") -> Dict[str, Any]:
    obj = repair_event_duration_ranges_v11(obj)
    obj = _original_v11_light_repair_sample(obj, cdr)
    obj = repair_event_duration_ranges_v11(obj)
    obj = repair_nonverbal_event_durations_v11(obj)
    obj = recompute_event_stats_and_acoustic(obj)
    obj = repair_speech_cognitive_task_from_text(obj, cdr)
    obj = repair_marked_and_spoken_from_events(obj)
    obj = strip_acoustic_condition_fields(obj)
    return obj


def validate_sample(obj: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, List[str]]:
    ok, errors = _original_v11_validate_sample(obj, expected)
    cdr = str(expected.get("cdr") or expected.get("cdr_level"))
    task_group = expected.get("task_group")

    filtered: List[str] = []
    for e in errors:
        # Base validator uses old raw picture key-unit detector. For CDR 3 we now
        # intentionally use V11 functional key-units, so suppress old mismatch errors.
        if cdr == "3" and task_group == "picture_description":
            if e.startswith("picture key_units_mentioned=") and "transcript-derived" in e:
                continue
            if e.startswith("picture score=") and "computed" in e and "V11" not in e:
                continue
            if "acoustic_fluency.pause_count=" in e and "outside allowed range [3, 12]" in e:
                m = re.search(r"pause_count=(\d+)", e)
                n = int(m.group(1)) if m else None
                if n is not None and 2 <= n <= 16:
                    continue
        filtered.append(e)

    deduped: List[str] = []
    for e in filtered:
        if e not in deduped:
            deduped.append(e)
    return len(deduped) == 0, deduped



# -----------------------------
# Generation loop
# -----------------------------

def build_sample_id(cdr: str, voice_id: str, task_group: str, scenario: str, index: int) -> str:
    cdr_str = cdr.replace(".", "_")
    short_group = {
        "picture_description": "pic",
        "cognitive_interview": "cog",
        "real_life_scenarios": "real",
    }.get(task_group, task_group)
    return f"cdr_{cdr_str}_{voice_id}_{short_group}_{scenario}_{index:04d}"


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
    task_group: str,
    scenario: str,
    task_plan: Dict[str, Any],
    speaker: Dict[str, str],
    split: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    rejected_attempts = []
    previous_errors: Optional[List[str]] = None

    expected = {
        "sample_id": sample_id,
        "cdr": cdr,
        "task_group": task_group,
        "scenario": scenario,
        "task_type": task_plan.get("task_type"),
        "split": split,
    }

    for attempt in range(1, args.max_retries + 1):
        prompt = build_prompt(
            sample_id=sample_id,
            cdr=cdr,
            task_group=task_group,
            scenario=scenario,
            task_plan=task_plan,
            speaker=speaker,
            split=split,
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
        "task_group_weights": TASK_GROUP_WEIGHTS,
        "folder_structure": "output_dir/split/cdr_level/task_group/sample.json",
        "manifest": "dataset_manifest.csv",
        "accepted": 0,
        "failed": 0,
        "failed_samples": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    global_index = 0

    for cdr, count in counts.items():
        for i in range(count):
            global_index += 1

            task_group = choose_task_group(i, count)
            scenario = choose_scenario_for_task_group(task_group)
            task_plan = build_task_plan(task_group, scenario)
            speaker = random.choice(VOICE_POOL)
            split = choose_split(i, count)
            sample_id = build_sample_id(cdr, speaker["voice_id"], task_group, scenario, i)
            sample_path = out_dir / split / cdr_folder_name(cdr) / task_group / f"{sample_id}.json"

            if args.resume and sample_path.exists():
                print(f"[SKIP] {sample_id}")
                continue

            print(f"[GENERATE] {sample_id} | CDR {cdr} | {task_group} | {scenario} | {speaker['voice_id']}")

            try:
                obj, rejected = generate_one(
                    args=args,
                    sample_id=sample_id,
                    cdr=cdr,
                    task_group=task_group,
                    scenario=scenario,
                    task_plan=task_plan,
                    speaker=speaker,
                    split=split,
                )
            except Exception as e:
                obj = None
                rejected = [{"attempt": "exception", "errors": [str(e)]}]

            if obj is not None:
                save_json(sample_path, obj)
                write_manifest_row(out_dir / "dataset_manifest.csv", obj, sample_path, out_dir)
                summary["accepted"] += 1
                summary.setdefault("accepted_by_task_group", {}).setdefault(task_group, 0)
                summary["accepted_by_task_group"][task_group] += 1
                print(f"  -> ACCEPTED: {sample_path}")
            else:
                summary["failed"] += 1
                summary["failed_samples"].append(sample_id)
                fail_path = rejected_dir / split / cdr_folder_name(cdr) / task_group / f"{sample_id}.rejected.json"
                save_json(fail_path, {
                    "sample_id": sample_id,
                    "cdr": cdr,
                    "task_group": task_group,
                    "scenario": scenario,
                    "task_plan": task_plan,
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


# ---------------------------------------------------------------------
# V12 final targeted patch: CDR2/3 non-cognitive reliable scoring
# ---------------------------------------------------------------------
# Problem found from real rejected logs:
# - CDR2 picture_description was rejected because raw keyword matching gave
#   7-8/10 even when the transcript was hesitant, vague, and partially wrong.
# - CDR2 real-life market was rejected because raw step matching gave 4/4 even
#   when the transcript explicitly contained forgetting/uncertainty.
# - CDR3 picture/family were mostly repaired by V11, but V12 keeps the same
#   reliable scoring behavior and filters old raw-score validator errors.
#
# Principle:
# The JSON score should measure RELIABLE task information, not mere keyword
# mentions. For CDR2/3, uncertain fragments like 「好像」、「忘記」、「不知道」、
# 「那個」 should reduce functional score.

# Keep references to the last active V11.1 wrappers.
_original_v11_1_light_repair_sample = light_repair_sample
_original_v11_1_validate_sample = validate_sample
_original_v11_1_repair_speech_cognitive_task_from_text = repair_speech_cognitive_task_from_text

COMMON_TYPO_REPLACEMENTS_V12 = {
    "菜市埸": "菜市場",
    "市埸": "市場",
    "诊": "診",
    "脸": "臉",
}


def replace_common_typos_v12(obj: Any) -> Any:
    if isinstance(obj, str):
        for a, b in COMMON_TYPO_REPLACEMENTS_V12.items():
            obj = obj.replace(a, b)
        return obj
    if isinstance(obj, list):
        return [replace_common_typos_v12(x) for x in obj]
    if isinstance(obj, dict):
        return {k: replace_common_typos_v12(v) for k, v in obj.items()}
    return obj


def impairment_evidence_strength_v12(text: str) -> int:
    """Visible uncertainty/impairment evidence used to reduce functional scores."""
    t = normalize_text(text)
    score = 0
    score += count_occurrences(t, ["忘記", "想不起來", "不知道", "不確定", "記不太清楚", "不太記得", "不記得"])
    score += count_occurrences(t, ["好像", "應該", "可能", "對不對", "那個", "這個", "呃", "嗯"])
    score += count_occurrences(t, ["叫什麼", "怎麼講", "那個東西", "紅色的東西", "會跑的東西"])
    score += t.count("...") + t.count("…")
    return score


def compute_picture_key_units_v12(text: str, cdr: str) -> int:
    """
    Reliable picture key-unit scoring.
    CDR0-1: mostly raw functional key-unit scoring.
    CDR2: cap keyword-rich but impaired descriptions to the moderate envelope.
    CDR3: keep V11 severe functional scoring.
    """
    cdr = str(cdr)
    raw = compute_picture_key_units(text)
    if cdr == "3":
        return compute_picture_key_units_v11(text, cdr)
    if cdr != "2":
        return raw

    t = normalize_text(text)
    evidence = impairment_evidence_strength_v12(t)
    has_memory_or_uncertainty = any(p in t for p in [
        "忘記", "想不起來", "不知道", "不確定", "好像", "可能", "那個", "這個", "呃", "嗯", "不記得", "記不太清楚"
    ])

    # If CDR2 text is clean and complete, do not rescue it; let validation reject.
    if raw >= 8 and evidence < 3:
        return raw

    # If it is keyword-rich but clearly impaired, score usable information, not object listing.
    if raw >= 7 and has_memory_or_uncertainty:
        return 6
    if raw == 6 and has_memory_or_uncertainty:
        return 6
    return max(0, min(6, raw))


# Override the V11 name too, because older V11 validate_sample dynamically looks it up.
def compute_picture_key_units_v11(text: str, cdr: str) -> int:  # type: ignore[no-redef]
    cdr = str(cdr)
    if cdr == "2":
        return compute_picture_key_units_v12(text, cdr)
    # Original severe behavior, copied from V11 to avoid recursion.
    raw = compute_picture_key_units(text)
    if cdr != "3":
        return raw
    t = normalize_text(text)
    frag = cdr3_fragmentation_strength(t)
    complete = 0
    if "客廳" in t and _contains_any(t, ["看到", "這裡", "裡面", "圖"]):
        complete += 1
    if _contains_any(t, ["阿公", "老人", "老先生", "爺爺"]) and _contains_any(t, ["看相簿", "看照片", "看東西"]):
        complete += 1
    if _contains_any(t, ["小孩", "孩子"]) and "積木" in t and _contains_any(t, ["玩", "地上"]):
        complete += 1
    if _contains_any(t, ["媽媽", "女人", "阿姨"]) and _contains_any(t, ["電話", "窗邊"]):
        complete += 1
    if "貓" in t and _contains_any(t, ["茶几", "杯", "倒", "碰"]):
        complete += 1
    if _contains_any(t, ["時鐘", "照片", "相片"]):
        complete += 1
    if frag >= 5:
        return max(0, min(4, complete))
    return max(0, min(4, raw))


def compute_real_life_steps_v12(text: str, scenario: str, cdr: str) -> Tuple[int, int, int]:
    """
    Reliable real-life scoring.
    For CDR2/3, do not give full 4/4 just because all step keywords appear.
    Memory gaps, uncertainty, and sequence confusion reduce reliable steps.
    """
    raw_steps, sequence_error, memory_gap = compute_real_life_steps_v10(text, scenario)
    cdr = str(cdr)
    if cdr == "3":
        return compute_real_life_steps_v11(text, scenario, cdr)
    if cdr != "2":
        return raw_steps, sequence_error, memory_gap

    t = normalize_text(text)
    evidence = impairment_evidence_strength_v12(t)
    has_uncertain_recall = memory_gap >= 1 or any(p in t for p in [
        "忘記", "想不起來", "不記得", "不太記得", "不確定", "好像", "對不對", "順序有點亂", "忘了"
    ])

    if has_uncertain_recall or evidence >= 4:
        reliable_steps = min(raw_steps, 2)
    else:
        reliable_steps = raw_steps

    # Keep score in CDR2 envelope while preserving visible memory gap evidence.
    # Do not invent sequence errors unless the text indicates ordering confusion.
    if any(p in t for p in ["順序有點亂", "先後", "不知道先", "忘記順序", "不記得順序"]):
        sequence_error = max(sequence_error, 1)
    else:
        sequence_error = min(sequence_error, max(0, reliable_steps - 1))

    reliable_steps = max(0, min(4, reliable_steps))
    sequence_error = max(0, min(3, sequence_error))
    memory_gap = max(0, min(5, memory_gap))
    return reliable_steps, sequence_error, memory_gap


# Override the V11 name because V11 validate_sample dynamically calls it.
def compute_real_life_steps_v11(text: str, scenario: str, cdr: str) -> Tuple[int, int, int]:  # type: ignore[no-redef]
    cdr = str(cdr)
    if cdr == "2":
        return compute_real_life_steps_v12(text, scenario, cdr)
    if cdr == "3":
        raw_steps, sequence_error, memory_gap = compute_real_life_steps_v10(text, scenario)
        t = normalize_text(text)
        uncertainty = cdr3_fragmentation_strength(t)
        if scenario in {"daily_life", "medicine_routine", "home_activity_recall", "home"}:
            reliable_steps = min(raw_steps, 1) if (memory_gap >= 2 or uncertainty >= 5) else min(raw_steps, 2)
        else:
            reliable_steps = min(raw_steps, 1 if memory_gap >= 2 else 2)
        sequence_error = max(sequence_error, max(0, raw_steps - reliable_steps))
        return max(0, min(4, reliable_steps)), max(0, min(3, sequence_error)), max(0, min(5, memory_gap))
    return compute_real_life_steps_v10(text, scenario)


def clamp_acoustic_values_v12(obj: Dict[str, Any], cdr: str) -> Dict[str, Any]:
    """Clamp acoustic scalar fields that are metadata, not transcript content."""
    if not isinstance(obj, dict):
        return obj
    rules = CDR_RULES.get(str(cdr))
    if not rules:
        return obj
    try:
        acoustic = obj["impairment_features"]["acoustic_fluency"]
        lo, hi = rules["speech_rate_cps"]
        val = acoustic.get("speech_rate_target_cps")
        if not isinstance(val, (int, float)):
            acoustic["speech_rate_target_cps"] = round((lo + hi) / 2, 2)
        else:
            acoustic["speech_rate_target_cps"] = round(max(lo, min(hi, float(val))), 2)
    except Exception:
        pass
    return obj


def repair_speech_cognitive_task_from_text(obj: Dict[str, Any], cdr: str) -> Dict[str, Any]:  # type: ignore[no-redef]
    obj = _original_v11_1_repair_speech_cognitive_task_from_text(obj, cdr)
    if not isinstance(obj, dict):
        return obj

    task_group = obj.get("task_group")
    spoken = normalize_text(obj.get("spoken_transcript", ""))

    if task_group == "picture_description" and str(cdr) == "2":
        total = len(PICTURE_REFERENCE["key_units"])
        mentioned = compute_picture_key_units_v12(spoken, cdr)
        task = obj.get("speech_cognitive_task") if isinstance(obj.get("speech_cognitive_task"), dict) else {}
        incorrect = int(task.get("incorrect_detail_count", 0) or 0)
        incorrect = max(0, min(3, incorrect))
        score = max(0, min(6, mentioned - incorrect))
        mentioned = max(0, min(6, mentioned))
        task.update({
            "task_group": "picture_description",
            "task_type": "picture_description",
            "scoring_method": "key_information_units",
            "expected_key_units": PICTURE_REFERENCE["key_units"],
            "key_units_total": total,
            "key_units_mentioned": mentioned,
            "missing_key_information_count": total - mentioned,
            "incorrect_detail_count": incorrect,
            "score": score,
            "max_score": total,
            "score_interpretation": "speech-based cognitive task score, not clinical MMSE score",
        })
        obj["speech_cognitive_task"] = task
        try:
            cog = obj["impairment_features"]["cognitive_task_performance"]
            cog["key_information_units_total"] = total
            cog["key_information_units_mentioned"] = mentioned
            cog["missing_key_information_count"] = total - mentioned
            cog["incorrect_detail_count"] = incorrect
            cog["orientation_error_count"] = 0
        except Exception:
            pass

    elif task_group == "real_life_scenarios" and str(cdr) == "2":
        scenario = str(obj.get("scenario", ""))
        plan = REAL_LIFE_TASK_BANK.get(scenario)
        if plan:
            steps, sequence_error, memory_gap = compute_real_life_steps_v12(spoken, scenario, cdr)
            max_score = int(plan.get("max_score", 4))
            score = max(0, min(2, steps - sequence_error))
            task = obj.get("speech_cognitive_task") if isinstance(obj.get("speech_cognitive_task"), dict) else {}
            task.update({
                "task_group": "real_life_scenarios",
                "task_type": plan["task_type"],
                "prompt": plan["prompt"],
                "scoring_method": "expected_steps_and_sequence",
                "score": score,
                "max_score": max_score,
                "score_interpretation": "speech-based cognitive task score, not clinical MMSE score",
                "expected_steps": plan["expected_steps"],
                "steps_mentioned": steps,
                "sequence_error_count": sequence_error,
                "memory_gap_count": memory_gap,
            })
            obj["speech_cognitive_task"] = task
            try:
                obj["impairment_features"]["cognitive_task_performance"]["memory_gap_count"] = clamp_int(memory_gap, 0, CDR_RULES[str(cdr)]["memory_gap"][1])
            except Exception:
                pass

    return obj


def light_repair_sample(obj: Dict[str, Any], cdr: str = "") -> Dict[str, Any]:  # type: ignore[no-redef]
    obj = replace_common_typos_v12(obj)
    obj = _original_v11_1_light_repair_sample(obj, cdr)
    obj = replace_common_typos_v12(obj)
    obj = repair_event_duration_ranges_v11(obj)
    obj = repair_nonverbal_event_durations_v11(obj)
    obj = recompute_event_stats_and_acoustic(obj)
    obj = repair_speech_cognitive_task_from_text(obj, cdr)
    obj = clamp_acoustic_values_v12(obj, cdr)
    obj = repair_marked_and_spoken_from_events(obj)
    obj = strip_acoustic_condition_fields(obj)
    return obj


def validate_sample(obj: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, List[str]]:  # type: ignore[no-redef]
    ok, errors = _original_v11_1_validate_sample(obj, expected)
    cdr = str(expected.get("cdr") or expected.get("cdr_level"))
    task_group = expected.get("task_group")
    spoken = normalize_text(obj.get("spoken_transcript", "")) if isinstance(obj, dict) else ""

    filtered: List[str] = []
    for e in errors:
        # Old raw picture validator overcounts keyword-rich impaired CDR2/3 picture speech.
        if cdr in {"2", "3"} and task_group == "picture_description":
            if e.startswith("picture key_units_mentioned=") and "transcript-derived" in e:
                continue
            if e.startswith("picture score=") and "computed" in e and "V11" not in e:
                continue
            if "cognitive_task_performance.missing_key_information_count=" in e and "outside allowed range" in e:
                # We re-check below using V12 reliable key units.
                continue
        # Old V11 real-life consistency checker is now replaced for CDR2 as well.
        if cdr == "2" and task_group == "real_life_scenarios" and e.startswith("real_life "):
            continue
        filtered.append(e)

    errors = filtered

    # V12 picture consistency for CDR2/3.
    if task_group == "picture_description" and cdr in {"2", "3"}:
        task = obj.get("speech_cognitive_task", {}) if isinstance(obj.get("speech_cognitive_task"), dict) else {}
        mentioned = compute_picture_key_units_v12(spoken, cdr)
        total = len(PICTURE_REFERENCE["key_units"])
        incorrect = int(task.get("incorrect_detail_count", 0) or 0)
        max_score = 6 if cdr == "2" else 4
        if cdr == "2":
            mentioned = min(mentioned, 6)
        expected_score = max(0, min(max_score, mentioned - incorrect))
        if task.get("key_units_mentioned") != mentioned:
            errors.append(f"picture key_units_mentioned={task.get('key_units_mentioned')} does not match V12 reliable detector {mentioned}")
        if task.get("missing_key_information_count") != total - mentioned:
            errors.append(f"picture missing_key_information_count={task.get('missing_key_information_count')} does not match V12 {total - mentioned}")
        if task.get("score") != expected_score:
            errors.append(f"picture score={task.get('score')} does not match V12 computed {expected_score}")

    # V12 real-life consistency for CDR2.
    if task_group == "real_life_scenarios" and cdr == "2":
        scenario = str(expected.get("scenario") or obj.get("scenario", ""))
        task = obj.get("speech_cognitive_task", {}) if isinstance(obj.get("speech_cognitive_task"), dict) else {}
        steps, sequence_error, memory_gap = compute_real_life_steps_v12(spoken, scenario, cdr)
        expected_score = max(0, min(2, steps - sequence_error))
        if task.get("steps_mentioned") != steps:
            errors.append(f"real_life steps_mentioned={task.get('steps_mentioned')} does not match V12 reliable-step detector {steps}")
        if task.get("sequence_error_count") != sequence_error:
            errors.append(f"real_life sequence_error_count={task.get('sequence_error_count')} does not match V12 detector {sequence_error}")
        if task.get("score") != expected_score:
            errors.append(f"real_life score={task.get('score')} does not match V12 computed {expected_score}")

    deduped: List[str] = []
    for e in errors:
        if e not in deduped:
            deduped.append(e)
    return len(deduped) == 0, deduped

# V12.1 cleanup: one-person samples must not carry demo full-conversation turns.
_original_v12_light_repair_sample = light_repair_sample


def light_repair_sample(obj: Dict[str, Any], cdr: str = "") -> Dict[str, Any]:  # type: ignore[no-redef]
    obj = _original_v12_light_repair_sample(obj, cdr)
    if isinstance(obj, dict) and obj.get("interaction_type") == "one_person_description":
        obj["conversation_context"] = None
        obj["full_conversation_event_script"] = None
    return obj


# -----------------------------------------------------------------------------
# V13 final targeted patch: uncertainty-window scoring for CDR2/3 non-cognitive
# -----------------------------------------------------------------------------
# V12 still allowed old validators / raw keyword logic to over-count:
# - CDR2 market: "不太清楚到底有沒有付錢" counted as a reliable payment step.
# - CDR3 picture: fragmented object lists counted as full picture understanding.
# - CDR3 family: vague/contradictory person/place/time mentions counted as full steps.
# V13 scores RELIABLE information only: a keyword near 忘記/好像/不清楚/對不對/etc.
# is not treated as a confident task unit.

VALIDATOR_VERSION = "V13_UNCERTAINTY_WINDOW_RELIABLE_SCORING"

_original_v12_1_light_repair_sample = light_repair_sample
_original_v12_validate_sample = validate_sample
_original_v12_repair_speech_cognitive_task_from_text = repair_speech_cognitive_task_from_text
_original_expected_score_range_for_cdr = expected_score_range_for_cdr

UNCERTAIN_CONTEXT_PATTERNS_V13 = [
    "忘記", "忘了", "想不起來", "不記得", "不太記得", "記不清楚", "記不太清楚",
    "不知道", "不清楚", "不太清楚", "不確定", "好像", "可能", "應該",
    "對不對", "是不是", "有沒有", "還是", "什麼來著", "叫什麼", "怎麼講",
    "那個", "這個", "呃", "嗯", "?", "？", "...", "…",
]

STRONG_UNCERTAIN_PATTERNS_V13 = [
    "忘記", "忘了", "想不起來", "不記得", "不太記得", "記不清楚", "記不太清楚",
    "不知道", "不清楚", "不太清楚", "不確定", "什麼來著", "叫什麼", "有沒有",
]


def _v13_context(text: str, idx: int, pattern_len: int, window: int = 14) -> str:
    lo = max(0, idx - window)
    hi = min(len(text), idx + pattern_len + window)
    return text[lo:hi]


def _v13_find_pattern_context(text: str, patterns: List[str], window: int = 14) -> Optional[str]:
    t = normalize_text(text)
    best_idx = None
    best_pat = None
    for p in patterns:
        idx = t.find(p)
        if idx >= 0 and (best_idx is None or idx < best_idx):
            best_idx = idx
            best_pat = p
    if best_idx is None or best_pat is None:
        return None
    return _v13_context(t, best_idx, len(best_pat), window)


def _v13_is_uncertain_context(ctx: str) -> bool:
    if not ctx:
        return False
    return any(p in ctx for p in UNCERTAIN_CONTEXT_PATTERNS_V13)


def _v13_has_strong_uncertainty(ctx: str) -> bool:
    if not ctx:
        return False
    return any(p in ctx for p in STRONG_UNCERTAIN_PATTERNS_V13)


def _v13_visible_memory_gap_count(text: str) -> int:
    t = normalize_text(text)
    return count_occurrences(t, [
        "忘記", "忘了", "想不起來", "不記得", "不太記得", "記不清楚", "記不太清楚",
        "不知道", "不清楚", "不太清楚", "想不起來", "沒有印象"
    ])


def _v13_fragment_strength(text: str) -> int:
    t = normalize_text(text)
    score = 0
    score += t.count("...") + t.count("…")
    score += count_occurrences(t, ["那個", "這個", "呃", "嗯", "好像", "可能", "對不對", "忘記", "想不起來", "不清楚"])
    # Many very short speech chunks joined by punctuation/space are a sign of object-listing.
    short_chunks = [c for c in re.split(r"[，。！？?\s]+", t) if 0 < len(c) <= 4]
    if len(short_chunks) >= 8:
        score += 3
    elif len(short_chunks) >= 5:
        score += 2
    return score


def _v13_has_phrase(text: str, patterns: List[str], require_no_strong_uncertainty: bool = True) -> bool:
    ctx = _v13_find_pattern_context(text, patterns)
    if ctx is None:
        return False
    if require_no_strong_uncertainty and _v13_has_strong_uncertainty(ctx):
        return False
    return True


def compute_picture_key_units_v13(text: str, cdr: str) -> int:
    """
    V13 picture scoring:
    - CDR0/0.5/1: use the existing mostly raw detector.
    - CDR2: count reliable visual information; uncertainty caps score.
    - CDR3: isolated object lists are not full credit. Require relations/actions.
    """
    cdr = str(cdr)
    t = normalize_text(text)
    raw = compute_picture_key_units(t)
    if cdr not in {"2", "3"}:
        return raw

    relation_score = 0
    # 客廳 context: can be a reliable scene unit unless clearly uncertain.
    if _v13_has_phrase(t, ["客廳"], require_no_strong_uncertainty=False):
        relation_score += 1
    # 阿公 unit requires person + action/object context.
    if _contains_any(t, ["阿公", "老人", "老先生", "爺爺"]):
        if _contains_any(t, ["沙發", "坐", "看相簿", "看照片", "看東西", "相簿"]):
            ctx = _v13_find_pattern_context(t, ["阿公", "老人", "老先生", "爺爺"], 18) or ""
            if not _v13_has_strong_uncertainty(ctx):
                relation_score += 1
    # Old glasses / tea: count for CDR2, but for CDR3 only if not pure object list.
    if _v13_has_phrase(t, ["老花眼鏡", "眼鏡"], require_no_strong_uncertainty=True):
        if cdr == "2" or _contains_any(t, ["旁邊", "在那邊", "放"]):
            relation_score += 1
    if _v13_has_phrase(t, ["茶杯", "茶", "杯子"], require_no_strong_uncertainty=True):
        if cdr == "2" or _contains_any(t, ["旁邊", "茶几", "倒", "碰", "放"]):
            relation_score += 1
    # Child blocks.
    if _contains_any(t, ["小孩", "孩子", "小朋友"]) and _contains_any(t, ["積木", "玩", "地上"]):
        ctx = _v13_find_pattern_context(t, ["小孩", "孩子", "小朋友"], 18) or ""
        if not _v13_has_strong_uncertainty(ctx):
            relation_score += 1
    # Mother phone.
    if _contains_any(t, ["媽媽", "女人", "阿姨", "女的"]) and _contains_any(t, ["電話", "接電話", "講電話", "窗邊"]):
        ctx = _v13_find_pattern_context(t, ["媽媽", "女人", "阿姨", "女的"], 18) or ""
        if not _v13_has_strong_uncertainty(ctx):
            relation_score += 1
    # Cat danger unit combines cat with table/cup/danger.
    if "貓" in t and _contains_any(t, ["茶几", "杯", "倒", "碰", "跳"]):
        ctx = _v13_find_pattern_context(t, ["貓"], 18) or ""
        if not _v13_has_strong_uncertainty(ctx):
            relation_score += 1
    # Cup falling is its own event.
    if _v13_has_phrase(t, ["碰倒", "快要倒", "翻倒", "倒了", "要倒", "快倒"], require_no_strong_uncertainty=True):
        relation_score += 1
    # Clock/photos are weaker; count them in CDR2, but not both as full CDR3 score unless coherent.
    weak = 0
    if _v13_has_phrase(t, ["時鐘", "鐘"], require_no_strong_uncertainty=True):
        weak += 1
    if _v13_has_phrase(t, ["家庭照片", "照片", "相片"], require_no_strong_uncertainty=True):
        weak += 1

    frag = _v13_fragment_strength(t)
    mem = _v13_visible_memory_gap_count(t)

    if cdr == "2":
        # CDR2 can contain fairly many details, but uncertainty/object listing should cap it.
        score = relation_score + weak
        if raw >= 8 and (frag >= 4 or mem >= 1):
            score = min(score, 6)
        return max(0, min(6, score))

    # CDR3: severe fragmented object listing. Score coherent relations only and cap hard.
    score = relation_score
    if frag >= 6 or mem >= 1:
        score = min(score, 3)
    # allow one weak environmental unit at most if the answer is not just a list
    if weak and frag < 6:
        score += 1
    return max(0, min(4, score))


# Override older dynamic names used by previous validators.
def compute_picture_key_units_v12(text: str, cdr: str) -> int:  # type: ignore[no-redef]
    return compute_picture_key_units_v13(text, cdr)


def compute_picture_key_units_v11(text: str, cdr: str) -> int:  # type: ignore[no-redef]
    return compute_picture_key_units_v13(text, cdr)


REAL_LIFE_STEP_PATTERNS_V13 = {
    "medicine_routine": [
        ["起床", "起來", "醒來", "早上起來"],
        ["吃早餐", "早餐", "早飯", "蛋餅", "牛奶", "粥", "吃東西"],
        ["看藥袋", "看藥盒", "確認藥", "確認一下藥", "看一下藥", "藥袋上", "藥盒裡"],
        ["吃藥", "服藥", "把藥吃", "吞藥", "吃了藥"],
    ],
    "daily_life": [
        ["起床", "起來", "醒來"],
        ["刷牙", "洗臉", "洗澡", "盥洗", "去廁所", "上廁所"],
        ["吃早餐", "早餐", "早飯", "蛋餅", "牛奶", "粥", "吃東西"],
        ["看電視", "散步", "公園", "出門", "休息", "整理", "活動"],
    ],
    "family": [
        ["爸爸", "媽媽", "妹妹", "哥哥", "姐姐", "家人", "全家", "阿公", "阿嬤", "兒子", "女兒", "孫子", "孫女"],
        ["餐廳", "家裡", "老家", "阿嬤家", "舅舅家", "家", "台北", "板橋", "外面"],
        ["吃飯", "聊天", "火鍋", "聚餐", "吃麵", "吃", "看電視", "散步"],
        ["上禮拜", "上週", "昨天", "晚上", "早上", "中午", "下午", "吃完", "之後", "最後", "回家", "上個月", "前週"],
    ],
    "home": [
        ["整理", "家務", "房間", "床", "洗衣", "曬", "打掃"],
        ["吃飯", "早餐", "午餐", "晚餐", "煮"],
        ["看電視", "散步", "公園", "休息", "聊天"],
        ["早上", "下午", "晚上", "中午", "之後", "最後", "傍晚"],
    ],
    "market": [
        ["出門", "從家裡", "走到", "去市場", "菜市場", "市場", "出去"],
        ["買菜", "買東西", "買", "青菜", "肉", "水果"],
        ["付錢", "結帳", "付款"],
        ["回家", "提回來", "拿回家", "帶回家"],
    ],
}


def _v13_step_is_reliable(text: str, patterns: List[str], cdr: str) -> bool:
    ctx = _v13_find_pattern_context(text, patterns, window=16)
    if ctx is None:
        return False
    # In CDR2/3, uncertain mentions such as 有沒有付錢, 好像去阿嬤家, 中午吧或者晚上
    # are not reliable task performance.
    if str(cdr) in {"2", "3"} and _v13_is_uncertain_context(ctx):
        return False
    return True


def compute_real_life_steps_v13(text: str, scenario: str, cdr: str) -> Tuple[int, int, int]:
    """Reliable real-life step scoring with uncertainty windows."""
    cdr = str(cdr)
    t = normalize_text(text)
    scenario = str(scenario)
    patterns = REAL_LIFE_STEP_PATTERNS_V13.get(scenario) or REAL_LIFE_STEP_PATTERNS_V13.get(str(scenario).replace("_recall", ""), [])
    if not patterns:
        return compute_real_life_steps_v10(t, scenario)

    raw_steps = sum(1 for pats in patterns if any(p in t for p in pats))
    reliable_steps = sum(1 for pats in patterns if _v13_step_is_reliable(t, pats, cdr))
    memory_gap = _v13_visible_memory_gap_count(t)

    sequence_error = 0
    if any(p in t for p in ["順序有點亂", "忘記順序", "順序不太記得", "不知道先後", "先後忘了", "不太清楚到底有沒有", "不知道有沒有", "不記得有沒有", "忘記有沒有"]):
        sequence_error += 1
    if scenario == "medicine_routine" and any(p in t for p in ["忘記看藥袋", "沒看藥袋", "直接把藥", "不知道什麼時候吃"]):
        sequence_error += 1
    if scenario == "market" and any(p in t for p in ["忘記帶錢包", "不清楚到底有沒有付錢", "不知道有沒有付錢", "沒辦法付錢"]):
        sequence_error += 1
    if scenario == "family" and any(p in t for p in ["中午吧，或者", "晚上？反正", "家裡，又好像不是", "兒子，他也在場吧？還是女兒"]):
        sequence_error += 1

    frag = _v13_fragment_strength(t)

    if cdr == "2":
        # Moderate impairment: allow 1-3 reliable steps, but cap if visible uncertainty is high.
        if memory_gap >= 2 or frag >= 7:
            reliable_steps = min(reliable_steps, 2)
        else:
            reliable_steps = min(reliable_steps, 3)
        # Do not let a single sequence error erase all moderately impaired content.
        sequence_error = min(sequence_error, 1)
    elif cdr == "3":
        # Severe impairment: even if keywords appear, only very clear info counts.
        if memory_gap >= 2 or frag >= 6:
            reliable_steps = min(reliable_steps, 1)
        else:
            reliable_steps = min(reliable_steps, 2)
        # Excess raw mentions beyond reliable info are recall/sequence failures.
        sequence_error = max(sequence_error, max(0, raw_steps - reliable_steps))
        sequence_error = min(sequence_error, 3)

    return max(0, min(4, reliable_steps)), max(0, min(3, sequence_error)), max(0, min(5, memory_gap))


# Override earlier dynamic names.
def compute_real_life_steps_v12(text: str, scenario: str, cdr: str) -> Tuple[int, int, int]:  # type: ignore[no-redef]
    return compute_real_life_steps_v13(text, scenario, cdr)


def compute_real_life_steps_v11(text: str, scenario: str, cdr: str) -> Tuple[int, int, int]:  # type: ignore[no-redef]
    return compute_real_life_steps_v13(text, scenario, cdr)


def expected_score_range_for_cdr(cdr: str, max_score: int) -> Tuple[int, int]:  # type: ignore[no-redef]
    """V13: keep old ranges except CDR2 4-point real-life tasks may validly score up to 3."""
    cdr = str(cdr)
    if cdr == "2" and int(max_score) == 4:
        return 1, 3
    return _original_expected_score_range_for_cdr(cdr, max_score)


def repair_speech_cognitive_task_from_text(obj: Dict[str, Any], cdr: str) -> Dict[str, Any]:  # type: ignore[no-redef]
    # Run the previous repair first, then overwrite the non-cognitive scoring with V13.
    obj = _original_v12_repair_speech_cognitive_task_from_text(obj, cdr)
    if not isinstance(obj, dict):
        return obj

    task_group = obj.get("task_group")
    spoken = normalize_text(obj.get("spoken_transcript", ""))
    cdr = str(cdr)

    if task_group == "picture_description" and cdr in {"2", "3"}:
        total = len(PICTURE_REFERENCE["key_units"])
        mentioned = compute_picture_key_units_v13(spoken, cdr)
        task = obj.get("speech_cognitive_task") if isinstance(obj.get("speech_cognitive_task"), dict) else {}
        incorrect = int(task.get("incorrect_detail_count", 0) or 0)
        incorrect = max(0, min(3, incorrect))
        max_score = 6 if cdr == "2" else 4
        score = max(0, min(max_score, mentioned - incorrect))
        task.update({
            "task_group": "picture_description",
            "task_type": "picture_description",
            "scoring_method": "key_information_units",
            "expected_key_units": PICTURE_REFERENCE["key_units"],
            "key_units_total": total,
            "key_units_mentioned": mentioned,
            "missing_key_information_count": total - mentioned,
            "incorrect_detail_count": incorrect,
            "score": score,
            "max_score": total,
            "score_interpretation": "speech-based cognitive task score, not clinical MMSE score",
        })
        obj["speech_cognitive_task"] = task
        try:
            cog = obj["impairment_features"]["cognitive_task_performance"]
            cog["key_information_units_total"] = total
            cog["key_information_units_mentioned"] = mentioned
            cog["missing_key_information_count"] = total - mentioned
            cog["incorrect_detail_count"] = incorrect
            cog["orientation_error_count"] = 0
        except Exception:
            pass

    elif task_group == "real_life_scenarios" and cdr in {"2", "3"}:
        scenario = str(obj.get("scenario", ""))
        plan = REAL_LIFE_TASK_BANK.get(scenario)
        if plan:
            steps, sequence_error, memory_gap = compute_real_life_steps_v13(spoken, scenario, cdr)
            max_score = int(plan.get("max_score", 4))
            score_hi = 3 if cdr == "2" else 1
            score = max(0, min(score_hi, steps - sequence_error))
            task = obj.get("speech_cognitive_task") if isinstance(obj.get("speech_cognitive_task"), dict) else {}
            task.update({
                "task_group": "real_life_scenarios",
                "task_type": plan["task_type"],
                "prompt": plan["prompt"],
                "scoring_method": "expected_steps_and_sequence",
                "score": score,
                "max_score": max_score,
                "score_interpretation": "speech-based cognitive task score, not clinical MMSE score",
                "expected_steps": plan["expected_steps"],
                "steps_mentioned": steps,
                "sequence_error_count": sequence_error,
                "memory_gap_count": memory_gap,
            })
            obj["speech_cognitive_task"] = task
            try:
                cog = obj["impairment_features"]["cognitive_task_performance"]
                cog["memory_gap_count"] = memory_gap
            except Exception:
                pass
    return obj


def light_repair_sample(obj: Dict[str, Any], cdr: str = "") -> Dict[str, Any]:  # type: ignore[no-redef]
    # Use V12.1 repair first, then recompute V13 scores AFTER spoken transcript has been rebuilt.
    obj = _original_v12_1_light_repair_sample(obj, cdr)
    obj = repair_speech_cognitive_task_from_text(obj, cdr)
    obj = repair_event_duration_ranges_v11(obj)
    obj = recompute_event_stats_and_acoustic(obj)
    obj = clamp_acoustic_values_v12(obj, cdr)
    obj = strip_acoustic_condition_fields(obj)
    if isinstance(obj, dict) and obj.get("interaction_type") == "one_person_description":
        obj["conversation_context"] = None
        obj["full_conversation_event_script"] = None
    return obj


def validate_sample(obj: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, List[str]]:  # type: ignore[no-redef]
    ok, errors = _original_v12_validate_sample(obj, expected)
    cdr = str(expected.get("cdr") or expected.get("cdr_level"))
    task_group = expected.get("task_group")
    scenario = str(expected.get("scenario") or (obj.get("scenario", "") if isinstance(obj, dict) else ""))
    spoken = normalize_text(obj.get("spoken_transcript", "")) if isinstance(obj, dict) else ""

    filtered: List[str] = []
    for e in errors:
        # Suppress older raw validators; V13 re-checks below.
        if task_group == "picture_description" and cdr in {"2", "3"}:
            if e.startswith("picture key_units_mentioned="):
                continue
            if e.startswith("picture missing_key_information_count="):
                continue
            if e.startswith("picture score="):
                continue
            if "speech_cognitive_task.score=" in e and "outside CDR envelope" in e:
                continue
            if "cognitive_task_performance.missing_key_information_count=" in e and "outside allowed range" in e:
                continue
            if "pause event" in e and "outside" in e and cdr == "3":
                # CDR3 fragmented picture can contain borderline pauses; duration repair handles the audio side.
                continue
        if task_group == "real_life_scenarios" and cdr in {"2", "3"}:
            if e.startswith("real_life steps_mentioned=") or e.startswith("real_life sequence_error_count=") or e.startswith("real_life score="):
                continue
            if "speech_cognitive_task.score=" in e and "outside CDR envelope" in e:
                continue
        filtered.append(e)
    errors = filtered

    task = obj.get("speech_cognitive_task", {}) if isinstance(obj.get("speech_cognitive_task"), dict) else {}

    if task_group == "picture_description" and cdr in {"2", "3"}:
        total = len(PICTURE_REFERENCE["key_units"])
        mentioned = compute_picture_key_units_v13(spoken, cdr)
        incorrect = int(task.get("incorrect_detail_count", 0) or 0)
        incorrect = max(0, min(3, incorrect))
        max_score = 6 if cdr == "2" else 4
        expected_score = max(0, min(max_score, mentioned - incorrect))
        if task.get("key_units_mentioned") != mentioned:
            errors.append(f"picture key_units_mentioned={task.get('key_units_mentioned')} does not match V13 reliable detector {mentioned}")
        if task.get("missing_key_information_count") != total - mentioned:
            errors.append(f"picture missing_key_information_count={task.get('missing_key_information_count')} does not match V13 {total - mentioned}")
        if task.get("score") != expected_score:
            errors.append(f"picture score={task.get('score')} does not match V13 computed {expected_score}")
        # CDR3 should not be rejected merely because it mentioned isolated nouns; detector caps this.
        lo, hi = (2, 6) if cdr == "2" else (0, 4)
        if not (lo <= int(task.get("score", -999)) <= hi):
            errors.append(f"picture score={task.get('score')} outside V13 envelope [{lo}, {hi}]")

    if task_group == "real_life_scenarios" and cdr in {"2", "3"}:
        steps, sequence_error, memory_gap = compute_real_life_steps_v13(spoken, scenario, cdr)
        score_hi = 3 if cdr == "2" else 1
        expected_score = max(0, min(score_hi, steps - sequence_error))
        if task.get("steps_mentioned") != steps:
            errors.append(f"real_life steps_mentioned={task.get('steps_mentioned')} does not match V13 reliable-step detector {steps}")
        if task.get("sequence_error_count") != sequence_error:
            errors.append(f"real_life sequence_error_count={task.get('sequence_error_count')} does not match V13 detector {sequence_error}")
        if task.get("score") != expected_score:
            errors.append(f"real_life score={task.get('score')} does not match V13 computed {expected_score}")
        lo, hi = (1, 3) if cdr == "2" else (0, 1)
        if not (lo <= int(task.get("score", -999)) <= hi):
            errors.append(f"real_life score={task.get('score')} outside V13 envelope [{lo}, {hi}]")

    deduped: List[str] = []
    for e in errors:
        if e not in deduped:
            deduped.append(e)
    return len(deduped) == 0, deduped


# V13.1: do not treat fillers (嗯/呃/那個) as invalidating a whole real-life step.
# They are fluency markers, not proof that the step is unreliable. Strong uncertainty
# near the step still blocks credit: 忘記/不知道/不清楚/有沒有/etc.
VALIDATOR_VERSION = "V13_1_UNCERTAINTY_WINDOW_RELIABLE_SCORING"


def _v13_step_is_reliable(text: str, patterns: List[str], cdr: str) -> bool:  # type: ignore[no-redef]
    t = normalize_text(text)
    best_idx = None
    best_pat = None
    for p in patterns:
        idx = t.find(p)
        if idx >= 0 and (best_idx is None or idx < best_idx):
            best_idx = idx
            best_pat = p
    if best_idx is None or best_pat is None:
        return False
    if str(cdr) in {"2", "3"}:
        ctx = _v13_context(t, best_idx, len(best_pat), window=8)
        if _v13_has_strong_uncertainty(ctx):
            return False
    return True

