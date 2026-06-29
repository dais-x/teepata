#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Taiwanese Mandarin dementia-style transcript JSON files using Qwen via Ollama.

This script generates JSON samples for audio generation.

Main features:
- Taiwanese Mandarin only
- CDR levels: 0, 0.5, 1, 2, 3
- One-person description tasks
- Two-person conversation tasks
- One fixed picture-reference task
- Four available voices:
  voice1_male, voice2_female, voice3_male, voice4_female
- Output JSON structure compatible with later BreezyVoice/audio generation

Example:
python generate_tw_mandarin_cdr_json.py \
  --model qwen2.5:14b \
  --output-dir dataset/tw_mandarin_cdr_json \
  --cdr-counts "0:350,0.5:300,1:200,2:100,3:50"
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import requests


# ============================================================
# Dataset configuration
# ============================================================

CDR_LABELS = {
    "0": "normal",
    "0.5": "very_mild",
    "1": "mild",
    "2": "moderate",
    "3": "severe",
}

DEFAULT_CDR_COUNTS = {
    "0": 350,
    "0.5": 300,
    "1": 200,
    "2": 100,
    "3": 50,
}

VOICE_POOL = [
    {
        "voice_id": "voice1_male",
        "speaker_group": "male",
        "gender": "male",
        "description": "男性長輩，台灣華語，自然清楚。"
    },
    {
        "voice_id": "voice2_female",
        "speaker_group": "female",
        "gender": "female",
        "description": "女性長輩，台灣華語，自然清楚。"
    },
    {
        "voice_id": "voice3_male",
        "speaker_group": "male",
        "gender": "male",
        "description": "男性長輩，台灣華語，語速稍慢。"
    },
    {
        "voice_id": "voice4_female",
        "speaker_group": "female",
        "gender": "female",
        "description": "女性長輩，台灣華語，語氣柔和。"
    },
]

INTERVIEWER_VOICE = {
    "voice_id": "interviewer_neutral",
    "speaker_group": "interviewer",
    "gender": "neutral",
    "description": "訪談者，語氣清楚自然。"
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
    {
        "task_type": "picture_description",
        "interaction_type": "one_person_description",
        "weight": 35,
    },
    {
        "task_type": "daily_life_description",
        "interaction_type": "one_person_description",
        "weight": 25,
    },
    {
        "task_type": "orientation_conversation",
        "interaction_type": "two_person_conversation",
        "weight": 15,
    },
    {
        "task_type": "memory_recall_conversation",
        "interaction_type": "two_person_conversation",
        "weight": 15,
    },
    {
        "task_type": "structured_cognitive_interview",
        "interaction_type": "two_person_conversation",
        "weight": 10,
    },
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
    )
}

ACOUSTIC_CONDITIONS = [
    {
        "name": "clean",
        "weight": 45,
        "description": "乾淨近距離錄音，低背景噪音。"
    },
    {
        "name": "clinic_room_noise",
        "weight": 25,
        "description": "診間環境，輕微背景聲與空調聲。"
    },
    {
        "name": "phone_mic_degraded",
        "weight": 20,
        "description": "手機麥克風錄音，音質稍微壓縮。"
    },
    {
        "name": "home_background_noise",
        "weight": 10,
        "description": "家中背景聲，例如電風扇、遠處人聲或碗盤聲。"
    },
]


# ============================================================
# Utility
# ============================================================

def weighted_choice(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    weights = [item.get("weight", 1) for item in items]
    return random.choices(items, weights=weights, k=1)[0]


def parse_cdr_counts(text: str) -> Dict[str, int]:
    result = {}

    for part in text.split(","):
        if ":" not in part:
            raise ValueError(f"Invalid CDR count format: {part}")

        key, value = part.split(":", 1)
        key = key.strip()
        value = int(value.strip())

        if key not in CDR_LABELS:
            raise ValueError(f"Invalid CDR level: {key}")

        result[key] = value

    return result


def make_split_list(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> List[str]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    train_n = int(total * train_ratio)
    val_n = int(total * val_ratio)
    test_n = total - train_n - val_n

    splits = ["train"] * train_n + ["val"] * val_n + ["test"] * test_n

    rng = random.Random(seed)
    rng.shuffle(splits)

    return splits


def cdr_to_filename_part(cdr_level: str) -> str:
    return cdr_level.replace(".", "_")


def cdr_to_json_value(cdr_level: str):
    if cdr_level == "0.5":
        return 0.5
    return int(cdr_level)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    replacements = {
        "这": "這",
        "个": "個",
        "们": "們",
        "说": "說",
        "来": "來",
        "为": "為",
        "会": "會",
        "过": "過",
        "后": "後",
        "东": "東",
        "车": "車",
        "门": "門",
        "买": "買",
        "卖": "賣",
        "饭": "飯",
        "医": "醫",
        "药": "藥",
        "头": "頭",
        "发": "發",
        "没": "沒",
        "听": "聽",
        "话": "話",
        "点": "點",
        "边": "邊",
        "里": "裡",
        "嗎": "嗎",
        "(停頓)": "[停頓]",
        "（停頓）": "[停頓]",
        "(長停頓)": "[長停頓]",
        "（長停頓）": "[長停頓]",
        "(嘆氣)": "[嘆氣]",
        "（嘆氣）": "[嘆氣]",
        "(咳嗽)": "[咳嗽]",
        "（咳嗽）": "[咳嗽]",
        "(吸氣)": "[吸氣]",
        "（吸氣）": "[吸氣]",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_event_markers(text: str) -> str:
    text = re.sub(r"\[停頓\]", "……", text)
    text = re.sub(r"\[長停頓\]", "…………", text)
    text = re.sub(r"\[嘆氣\]", "……", text)
    text = re.sub(r"\[咳嗽\]", "……", text)
    text = re.sub(r"\[吸氣\]", "……", text)
    return clean_text(text)


def chinese_char_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


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
        raise ValueError("No JSON object found in model output.")

    return json.loads(match.group(0))


# ============================================================
# CDR behavior prompt
# ============================================================

def get_cdr_rules(cdr_level: str) -> str:
    if cdr_level == "0":
        return """
CDR 0 / normal:
- 語句自然、清楚、連貫。
- 日期、地點、人物關係大多正確。
- 幾乎沒有找詞困難。
- 可以有非常少量自然口語詞，例如「嗯」、「喔」。
- 不要故意製造明顯失智症狀。
- pause_event_count 建議 0 到 1。
- hesitation_count 建議 0 到 2。
- topic_drift_score 0.00 到 0.08。
- coherence_score 0.90 到 1.00。
- speech_rate_target: normal。
"""

    if cdr_level == "0.5":
        return """
CDR 0.5 / very mild:
- 整體仍然連貫，但有輕微猶豫。
- 偶爾出現「那個」、「我想一下」、「有點忘記」。
- 可以有輕微日期不確定，但不要嚴重混亂。
- 可以有一點自我修正。
- pause_event_count 建議 1 到 3。
- hesitation_count 建議 3 到 6。
- repetition_count 建議 0 到 2。
- topic_drift_score 0.08 到 0.20。
- coherence_score 0.75 到 0.90。
- speech_rate_target: slightly_slow。
"""

    if cdr_level == "1":
        return """
CDR 1 / mild:
- 明顯比正常更慢，有找詞困難。
- 可以重複問題或重複短語。
- 對日期、早餐、最近事情可能有小錯誤或不確定。
- 句子仍可理解，但會有停頓、自我修正、空泛詞。
- pause_event_count 建議 2 到 5。
- hesitation_count 建議 5 到 9。
- repetition_count 建議 1 到 4。
- word_finding_count 建議 1 到 4。
- topic_drift_score 0.18 到 0.35。
- coherence_score 0.60 到 0.80。
- speech_rate_target: slow。
"""

    if cdr_level == "2":
        return """
CDR 2 / moderate:
- 記憶困難明顯，回答需要訪談者引導。
- 常出現長停頓、重複、答非所問一點點。
- 可以把時間、地點或事件弄混。
- 句子較破碎，但還要能看出大概意思。
- pause_event_count 建議 4 到 8。
- hesitation_count 建議 8 到 14。
- repetition_count 建議 3 到 7。
- word_finding_count 建議 3 到 7。
- topic_drift_score 0.35 到 0.60。
- coherence_score 0.35 到 0.65。
- speech_rate_target: very_slow。
"""

    if cdr_level == "3":
        return """
CDR 3 / severe:
- 語句非常破碎，回答短、慢、不完整。
- 常常不知道問題重點，可能只回答片段。
- 很多長停頓、重複、找詞失敗。
- 可以出現明顯定向錯誤，但不要把內容寫成搞笑或誇張。
- pause_event_count 建議 6 到 10。
- hesitation_count 建議 10 到 18。
- repetition_count 建議 5 到 10。
- word_finding_count 建議 5 到 10。
- topic_drift_score 0.55 到 0.85。
- coherence_score 0.15 到 0.45。
- speech_rate_target: very_slow。
"""

    raise ValueError(f"Unknown CDR level: {cdr_level}")


# ============================================================
# Prompt builder
# ============================================================

def build_prompt(
    sample_id: str,
    cdr_level: str,
    cdr_label: str,
    scenario: str,
    task_type: str,
    interaction_type: str,
    patient_voice: Dict[str, str],
    split: str,
    acoustic_condition: Dict[str, Any],
) -> str:
    cdr_rules = get_cdr_rules(cdr_level)

    if task_type == "picture_description":
        task_instruction = f"""
TASK: one-person picture description.

The patient describes this ONE fixed picture reference.

picture_reference:
- picture_id: {PICTURE_REFERENCE["picture_id"]}
- name: {PICTURE_REFERENCE["name"]}
- description: {PICTURE_REFERENCE["description"]}

Important:
- Do not mention that this is a test.
- The patient should describe the picture naturally in Taiwanese Mandarin.
- The same picture reference is used for all picture-description samples.
- For CDR 0, the patient should describe the main actions clearly.
- For higher CDR levels, the patient may miss details, repeat, hesitate, or confuse actions.
"""
    elif task_type == "daily_life_description":
        task_instruction = """
TASK: one-person daily-life description.

The patient describes a daily-life topic based on the scenario.
Possible topics:
- morning routine
- going to the market
- taking medicine
- family visit
- clinic visit
- preparing breakfast
- looking for something at home

Important:
- It should sound like natural Taiwanese Mandarin from an older adult.
- It should not sound like formal textbook Mandarin.
"""
    elif task_type == "orientation_conversation":
        task_instruction = """
TASK: two-person orientation conversation.

The interviewer asks simple questions about:
- today's date or day
- current place
- why the patient came here
- recent daily events

Important:
- event_script must include both interviewer and patient speech events.
- Use speaker field: "interviewer" or "patient".
- The interviewer should speak briefly and naturally.
- The patient response should reflect the assigned CDR level.
"""
    elif task_type == "memory_recall_conversation":
        task_instruction = """
TASK: two-person memory recall conversation.

The interviewer asks about:
- breakfast
- recent family event
- medicine
- something the patient was asked to remember
- what happened yesterday or this morning

Important:
- event_script must include both interviewer and patient speech events.
- Use speaker field: "interviewer" or "patient".
- The interviewer should ask short questions.
- The patient response should reflect the assigned CDR level.
"""
    elif task_type == "structured_cognitive_interview":
        task_instruction = """
TASK: two-person structured cognitive interview.

The interviewer asks a short sequence of cognitive-style questions:
- orientation
- memory recall
- daily life
- simple picture/daily description

Important:
- This is MMSE-inspired, but do NOT copy the real MMSE.
- event_script must include both interviewer and patient speech events.
- Use speaker field: "interviewer" or "patient".
- Keep it realistic and respectful.
"""
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    picture_reference_json = "null"
    if task_type == "picture_description":
        picture_reference_json = json.dumps(PICTURE_REFERENCE, ensure_ascii=False)

    prompt = f"""
You are Qwen generating a synthetic Taiwanese Mandarin dementia speech transcript for audio dataset creation.

VERY IMPORTANT OUTPUT RULES:
1. Output ONLY one valid JSON object.
2. Do NOT wrap the JSON in markdown.
3. Do NOT add explanation before or after JSON.
4. Use Traditional Chinese only.
5. Use natural Taiwanese Mandarin.
6. Do NOT use Simplified Chinese.
7. Do NOT use English in the transcript except JSON keys.
8. Do NOT mention AI, dataset, synthetic, prompt, Qwen, or model.
9. Do NOT copy any real clinical test text.
10. The JSON must follow the exact schema requested below.

Dataset metadata:
- sample_id: {sample_id}
- cdr_level: {cdr_level}
- cdr_label: {cdr_label}
- scenario: {scenario}
- task_type: {task_type}
- interaction_type: {interaction_type}
- patient voice_id: {patient_voice["voice_id"]}
- patient speaker_group: {patient_voice["speaker_group"]}
- split: {split}
- acoustic_condition: {acoustic_condition["name"]}

CDR behavior rules:
{cdr_rules}

Task instruction:
{task_instruction}

Taiwanese Mandarin style:
- Use natural phrases like: 嗯、欸、啊、那個、我想一下、記不得、好像、應該是、差不多、啦、喔、咧、嘛.
- Do not overuse these phrases.
- Keep it realistic for older Taiwanese Mandarin speakers.
- Avoid too much Taigi. This dataset is Taiwanese Mandarin, not full Taigi.
- It is okay to include small Taiwan-style words like 囝仔, 阿母, 菜市場, 診所, 拿藥.
- Do not make the speech too dramatic.

Audio-generation rules:
- marked_transcript should contain markers:
  [停頓], [長停頓], [嘆氣], [咳嗽], [吸氣]
- event_script must separate speech and non-speech events.
- pause duration:
  [停頓] usually 400-900 ms
  [長停頓] usually 1000-1800 ms
  [嘆氣] usually 700-1500 ms
  [咳嗽] usually 300-900 ms
  [吸氣] usually 300-800 ms
- For now, sigh/cough/breath can be inserted as separate events.
- spoken_transcript should NOT contain square-bracket event markers.
- marked_transcript should contain event markers.
- event_script text should match the marked_transcript order.
- For two_person_conversation, event_script speech events must include speaker:
  "speaker": "interviewer" or "speaker": "patient".
- For one_person_description, event_script speech events can use:
  "speaker": "patient".

Required JSON schema:

{{
  "sample_id": "{sample_id}",
  "cdr_level": {cdr_to_json_value(cdr_level)},
  "cdr_label": "{cdr_label}",
  "scenario": "{scenario}",
  "task_type": "{task_type}",
  "interaction_type": "{interaction_type}",
  "picture_reference": {picture_reference_json},
  "speaker": {{
    "speaker_id": "{patient_voice["voice_id"]}",
    "voice_id": "{patient_voice["voice_id"]}",
    "speaker_group": "{patient_voice["speaker_group"]}",
    "gender": "{patient_voice["gender"]}",
    "split": "{split}"
  }},
  "interviewer": {{
    "speaker_id": "{INTERVIEWER_VOICE["voice_id"]}",
    "voice_id": "{INTERVIEWER_VOICE["voice_id"]}",
    "speaker_group": "interviewer"
  }},
  "split": "{split}",
  "spoken_transcript": "台灣華語逐字稿，不含事件標記。two-person conversation 要包含訪談者與患者的文字，例如：訪談者：...\\n患者：...",
  "marked_transcript": "含 [停頓] [長停頓] [嘆氣] [咳嗽] [吸氣] 的逐字稿。",
  "event_script": [
    {{
      "type": "speech",
      "speaker": "patient",
      "text": "..."
    }},
    {{
      "type": "pause",
      "duration_ms": 800
    }},
    {{
      "type": "sigh",
      "duration_ms": 1000
    }}
  ],
  "event_stats": {{
    "speech_chunk_count": 0,
    "pause_event_count": 0,
    "sigh_event_count": 0,
    "cough_event_count": 0,
    "breath_event_count": 0,
    "total_pause_ms": 0,
    "chinese_char_count": 0
  }},
  "impairment_features": {{
    "hesitation_count": 0,
    "repetition_count": 0,
    "word_finding_count": 0,
    "orientation_error_count": 0,
    "memory_error_count": 0,
    "topic_drift_score": 0.0,
    "coherence_score": 1.0,
    "speech_rate_target": "normal",
    "pause_event_count": 0
  }},
  "acoustic_condition": "{acoustic_condition["name"]}",
  "acoustic_condition_info": {{
    "weight": {acoustic_condition["weight"]},
    "description": "{acoustic_condition["description"]}"
  }}
}}

Before finalizing JSON, internally check:
- Does the CDR severity match the transcript?
- Does the transcript sound like Taiwanese Mandarin?
- Does the event_script order match the marked_transcript?
- Are pause counts and stats reasonable?
- Are all required keys present?

Now output ONLY the JSON object.
"""
    return prompt.strip()


# ============================================================
# Ollama call
# ============================================================

def call_ollama(
    prompt: str,
    model: str,
    ollama_host: str,
    temperature: float,
    top_p: float,
    timeout_sec: int,
) -> str:
    url = f"{ollama_host.rstrip('/')}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        }
    }

    response = requests.post(url, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    data = response.json()

    return data.get("response", "")


# ============================================================
# Validation / repair
# ============================================================

def repair_event_stats(sample: Dict[str, Any]) -> Dict[str, Any]:
    event_script = sample.get("event_script", [])

    speech_chunk_count = 0
    pause_event_count = 0
    sigh_event_count = 0
    cough_event_count = 0
    breath_event_count = 0
    total_pause_ms = 0

    for event in event_script:
        event_type = event.get("type")

        if event_type == "speech":
            speech_chunk_count += 1

        elif event_type == "pause":
            pause_event_count += 1
            total_pause_ms += int(event.get("duration_ms", 0) or 0)

        elif event_type == "sigh":
            sigh_event_count += 1
            total_pause_ms += int(event.get("duration_ms", 0) or 0)

        elif event_type == "cough":
            cough_event_count += 1
            total_pause_ms += int(event.get("duration_ms", 0) or 0)

        elif event_type == "breath":
            breath_event_count += 1
            total_pause_ms += int(event.get("duration_ms", 0) or 0)

    spoken_transcript = sample.get("spoken_transcript", "")

    sample["event_stats"] = {
        "speech_chunk_count": speech_chunk_count,
        "pause_event_count": pause_event_count,
        "sigh_event_count": sigh_event_count,
        "cough_event_count": cough_event_count,
        "breath_event_count": breath_event_count,
        "total_pause_ms": total_pause_ms,
        "chinese_char_count": chinese_char_count(spoken_transcript),
    }

    if "impairment_features" not in sample:
        sample["impairment_features"] = {}

    sample["impairment_features"]["pause_event_count"] = pause_event_count

    return sample


def repair_sample_metadata(
    sample: Dict[str, Any],
    sample_id: str,
    cdr_level: str,
    cdr_label: str,
    scenario: str,
    task_type: str,
    interaction_type: str,
    patient_voice: Dict[str, str],
    split: str,
    acoustic_condition: Dict[str, Any],
) -> Dict[str, Any]:

    sample["sample_id"] = sample_id
    sample["cdr_level"] = cdr_to_json_value(cdr_level)
    sample["cdr_label"] = cdr_label
    sample["scenario"] = scenario
    sample["task_type"] = task_type
    sample["interaction_type"] = interaction_type

    if task_type == "picture_description":
        sample["picture_reference"] = PICTURE_REFERENCE
    else:
        sample["picture_reference"] = None

    sample["speaker"] = {
        "speaker_id": patient_voice["voice_id"],
        "voice_id": patient_voice["voice_id"],
        "speaker_group": patient_voice["speaker_group"],
        "gender": patient_voice["gender"],
        "split": split,
    }

    sample["interviewer"] = {
        "speaker_id": INTERVIEWER_VOICE["voice_id"],
        "voice_id": INTERVIEWER_VOICE["voice_id"],
        "speaker_group": "interviewer",
    }

    sample["split"] = split

    sample["spoken_transcript"] = clean_text(sample.get("spoken_transcript", ""))
    sample["marked_transcript"] = clean_text(sample.get("marked_transcript", ""))

    if not sample["spoken_transcript"] and sample["marked_transcript"]:
        sample["spoken_transcript"] = remove_event_markers(sample["marked_transcript"])

    if not sample["marked_transcript"] and sample["spoken_transcript"]:
        sample["marked_transcript"] = sample["spoken_transcript"]

    if "event_script" not in sample or not isinstance(sample["event_script"], list):
        sample["event_script"] = [
            {
                "type": "speech",
                "speaker": "patient",
                "text": sample["spoken_transcript"]
            }
        ]

    # Clean event script
    repaired_events = []
    for event in sample["event_script"]:
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")

        if event_type == "speech":
            speaker = event.get("speaker", "patient")
            if speaker not in ["patient", "interviewer"]:
                speaker = "patient"

            repaired_events.append({
                "type": "speech",
                "speaker": speaker,
                "text": clean_text(event.get("text", ""))
            })

        elif event_type in ["pause", "sigh", "cough", "breath"]:
            duration_ms = event.get("duration_ms", 600)
            try:
                duration_ms = int(duration_ms)
            except Exception:
                duration_ms = 600

            repaired_events.append({
                "type": event_type,
                "duration_ms": duration_ms
            })

    sample["event_script"] = repaired_events

    if "impairment_features" not in sample or not isinstance(sample["impairment_features"], dict):
        sample["impairment_features"] = {}

    default_features = {
        "hesitation_count": 0,
        "repetition_count": 0,
        "word_finding_count": 0,
        "orientation_error_count": 0,
        "memory_error_count": 0,
        "topic_drift_score": 0.0,
        "coherence_score": 1.0,
        "speech_rate_target": "normal",
        "pause_event_count": 0,
    }

    for key, value in default_features.items():
        sample["impairment_features"].setdefault(key, value)

    sample["acoustic_condition"] = acoustic_condition["name"]
    sample["acoustic_condition_info"] = {
        "weight": acoustic_condition["weight"],
        "description": acoustic_condition["description"],
    }

    sample = repair_event_stats(sample)

    return sample


def validate_sample(sample: Dict[str, Any]) -> Tuple[bool, str]:
    required_keys = [
        "sample_id",
        "cdr_level",
        "cdr_label",
        "scenario",
        "task_type",
        "interaction_type",
        "speaker",
        "split",
        "spoken_transcript",
        "marked_transcript",
        "event_script",
        "event_stats",
        "impairment_features",
        "acoustic_condition",
        "acoustic_condition_info",
    ]

    for key in required_keys:
        if key not in sample:
            return False, f"Missing key: {key}"

    if not isinstance(sample["event_script"], list):
        return False, "event_script must be list"

    if len(sample["spoken_transcript"]) < 20:
        return False, "spoken_transcript too short"

    if sample["interaction_type"] == "two_person_conversation":
        has_interviewer = any(
            e.get("type") == "speech" and e.get("speaker") == "interviewer"
            for e in sample["event_script"]
        )
        has_patient = any(
            e.get("type") == "speech" and e.get("speaker") == "patient"
            for e in sample["event_script"]
        )

        if not has_interviewer or not has_patient:
            return False, "two_person_conversation must include both interviewer and patient speech events"

    return True, "ok"


# ============================================================
# Main generation
# ============================================================

def generate_one_sample(
    sample_id: str,
    cdr_level: str,
    cdr_label: str,
    scenario: str,
    task_type: str,
    interaction_type: str,
    patient_voice: Dict[str, str],
    split: str,
    acoustic_condition: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:

    prompt = build_prompt(
        sample_id=sample_id,
        cdr_level=cdr_level,
        cdr_label=cdr_label,
        scenario=scenario,
        task_type=task_type,
        interaction_type=interaction_type,
        patient_voice=patient_voice,
        split=split,
        acoustic_condition=acoustic_condition,
    )

    last_error = None

    for attempt in range(1, args.max_retries + 1):
        try:
            raw_text = call_ollama(
                prompt=prompt,
                model=args.model,
                ollama_host=args.ollama_host,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout_sec=args.timeout_sec,
            )

            sample = extract_json_from_model_output(raw_text)

            sample = repair_sample_metadata(
                sample=sample,
                sample_id=sample_id,
                cdr_level=cdr_level,
                cdr_label=cdr_label,
                scenario=scenario,
                task_type=task_type,
                interaction_type=interaction_type,
                patient_voice=patient_voice,
                split=split,
                acoustic_condition=acoustic_condition,
            )

            ok, message = validate_sample(sample)
            if not ok:
                raise ValueError(message)

            return sample

        except Exception as e:
            last_error = e
            print(f"[WARN] {sample_id} attempt {attempt}/{args.max_retries} failed: {e}")
            time.sleep(args.retry_sleep_sec)

    raise RuntimeError(f"Failed to generate {sample_id}: {last_error}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="qwen2.5:14b")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434")
    parser.add_argument("--output-dir", type=str, default="dataset/tw_mandarin_cdr_json")

    parser.add_argument(
        "--cdr-counts",
        type=str,
        default="0:350,0.5:300,1:200,2:100,3:50"
    )

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-sec", type=int, default=5)

    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cdr_counts = parse_cdr_counts(args.cdr_counts)
    total_samples = sum(cdr_counts.values())

    splits = make_split_list(
        total=total_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print("=" * 80)
    print("Taiwanese Mandarin CDR JSON Generator")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Ollama host: {args.ollama_host}")
    print(f"Output dir: {output_dir}")
    print(f"CDR counts: {cdr_counts}")
    print(f"Total samples: {total_samples}")
    print(f"Voices: {[v['voice_id'] for v in VOICE_POOL]}")
    print("=" * 80)

    if args.dry_run:
        print("Dry run only. No files generated.")
        return

    manifest = []
    global_index = 0

    for cdr_level, count in cdr_counts.items():
        cdr_label = CDR_LABELS[cdr_level]

        for local_index in range(1, count + 1):
            global_index += 1

            split = splits[global_index - 1]
            patient_voice = random.choice(VOICE_POOL)
            task_info = weighted_choice(TASK_TYPES)
            task_type = task_info["task_type"]
            interaction_type = task_info["interaction_type"]
            acoustic_condition = weighted_choice(ACOUSTIC_CONDITIONS)

            if task_type == "picture_description":
                scenario = "picture_description"
            else:
                scenario = random.choice(SCENARIOS)

            safe_cdr = cdr_to_filename_part(cdr_level)
            sample_id = (
                f"cdr_{safe_cdr}_"
                f"{patient_voice['voice_id']}_"
                f"{scenario}_"
                f"{global_index:04d}"
            )

            cdr_folder_name = f"cdr_{cdr_level.replace('.', '_')}"
            interaction_folder_name = interaction_type

            sample_output_dir = output_dir / cdr_folder_name / interaction_folder_name
            sample_output_dir.mkdir(parents=True, exist_ok=True)

            output_path = sample_output_dir / f"{sample_id}.json"

            if output_path.exists():
                print(f"[SKIP] {global_index}/{total_samples} {sample_id}")
                continue

            start_time = time.time()

            try:
                sample = generate_one_sample(
                    sample_id=sample_id,
                    cdr_level=cdr_level,
                    cdr_label=cdr_label,
                    scenario=scenario,
                    task_type=task_type,
                    interaction_type=interaction_type,
                    patient_voice=patient_voice,
                    split=split,
                    acoustic_condition=acoustic_condition,
                    args=args,
                )

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(sample, f, ensure_ascii=False, indent=2)

                elapsed = time.time() - start_time

                chars = sample["event_stats"]["chinese_char_count"]
                pauses = sample["event_stats"]["pause_event_count"]

                print(
                    f"[OK] {global_index}/{total_samples} | "
                    f"{sample_id} | "
                    f"cdr={cdr_level} | "
                    f"split={split} | "
                    f"voice={patient_voice['voice_id']} | "
                    f"task={task_type} | "
                    f"{elapsed:.2f}s | "
                    f"chars={chars} | "
                    f"pauses={pauses}"
                )

                manifest.append({
                    "sample_id": sample_id,
                    "path": str(output_path),
                    "cdr_level": cdr_to_json_value(cdr_level),
                    "cdr_label": cdr_label,
                    "scenario": scenario,
                    "task_type": task_type,
                    "interaction_type": interaction_type,
                    "voice_id": patient_voice["voice_id"],
                    "speaker_group": patient_voice["speaker_group"],
                    "split": split,
                    "acoustic_condition": acoustic_condition["name"],
                })

            except Exception as e:
                print(f"[ERROR] {global_index}/{total_samples} | {sample_id} | {e}")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Generation finished.")
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()