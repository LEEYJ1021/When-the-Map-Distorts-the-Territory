"""
================================================================================
STAGE 0 통합 파이프라인 v1.1  (버그픽스: parquet 타입오류, unk 언어 처리)
================================================================================
연구계획서 v5.0 - Month 1 즉시 착수 작업

포함 내용:
  [1] 언어감지 + 번역대상 산정 (fasttext)
  [2] Ollama 번역 파이프라인 (qwen2.5:14b, 배치/캐싱/체크포인트, Pilot QA)
  [3] 출원인 그룹분류 (llama3.1:8b, few-shot, STAGE 0-B)

사전 요구사항:
  - Ollama 설치 및 실행 중 (ollama serve)
  - 모델 다운로드:
      ollama pull qwen2.5:14b
      ollama pull llama3.1:8b
  - pip install fasttext langdetect pandas openpyxl requests tqdm pyarrow --break-system-packages
  - fasttext 언어감지 모델 다운로드:
      wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

실행 방법:
  python stage0_pipeline.py --step 1              # 언어감지/산정만
  python stage0_pipeline.py --step 2 --pilot-only # Pilot QA만
  python stage0_pipeline.py --step 2              # 전체 번역
  python stage0_pipeline.py --step 3              # 출원인 그룹분류
  python stage0_pipeline.py --step all            # 전체 순차 실행

v1.1 변경사항:
  - [FIX] parquet 저장 시 pyarrow ArrowTypeError 해결
    (Publication Date 등 혼합타입 object 컬럼을 저장 전 string으로 통일)
  - [FIX] fasttext 'unk' 반환 시 휴리스틱으로 영어 판정
    (ASCII 비율 기반 → 영어특허 오분류 방지)
  - [FIX] Step2 체크포인트/최종파일의 동일한 타입 통일 처리 적용
================================================================================
"""

import os
import re
import json
import time
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

# ------------------------------------------------------------------
# 공통 설정
# ------------------------------------------------------------------
BASE_DIR = Path.cwd()
INPUT_FILE = BASE_DIR / "b60w_data.xlsx"
WORK_DIR = BASE_DIR / "stage0_work"
WORK_DIR.mkdir(exist_ok=True)

# 산출물 경로
LANG_REPORT_PATH = WORK_DIR / "step1_language_report.json"
LANG_TAGGED_PATH = WORK_DIR / "step1_lang_tagged.parquet"
TRANSLATION_CACHE_PATH = WORK_DIR / "step2_translation_cache.json"
TRANSLATION_OUTPUT_PATH = WORK_DIR / "step2_translated.parquet"
TRANSLATION_CHECKPOINT_PATH = WORK_DIR / "step2_checkpoint.parquet"
PILOT_QA_PATH = WORK_DIR / "step2_pilot_qa.xlsx"
APPLICANT_GROUP_OUTPUT_PATH = WORK_DIR / "step3_applicant_groups.parquet"
APPLICANT_REVIEW_PATH = WORK_DIR / "step3_manual_review.xlsx"
TOP10_REPORT_PATH = WORK_DIR / "step3_top10_comparison.xlsx"

OLLAMA_URL = "http://localhost:11434/api/generate"
TRANSLATION_MODEL = "qwen2.5:14b"
CLASSIFICATION_MODEL = "llama3.1:8b"

FASTTEXT_MODEL_PATH = BASE_DIR / "lid.176.bin"


# ------------------------------------------------------------------
# [공통 유틸] parquet 저장 전 타입 안전화
# ------------------------------------------------------------------
def safe_to_parquet(df: pd.DataFrame, path, index=False):
    """
    pyarrow가 object 컬럼을 잘못된 타입으로 추론하는 문제를 방지한다.
    - list/dict 등 복합객체 컬럼은 JSON 문자열로 변환
    - 나머지 object 컬럼은 str(None→pd.NA)로 통일
    저장 후 원본 df는 변경하지 않는다(복사본으로 처리).
    """
    df_save = df.copy()
    for col in df_save.columns:
        col_dtype = df_save[col].dtype
        if col_dtype == object:
            # 첫 번째 non-null 값으로 타입 판별
            sample = df_save[col].dropna()
            if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict)):
                # 복합 객체 → JSON 문자열
                df_save[col] = df_save[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False) if x is not None and not (
                        isinstance(x, float) and np.isnan(x)) else None
                )
            else:
                # 일반 object → string (None 유지)
                df_save[col] = df_save[col].where(df_save[col].notna(), None)
                df_save[col] = df_save[col].astype(str).where(df_save[col].notna(), None)
    df_save.to_parquet(path, index=index)


# ==============================================================================
# [1] 언어감지 + 번역대상 산정
# ==============================================================================

def step1_language_detection():
    """
    각 특허의 Title / Abstract 언어를 감지하고,
    번역이 필요한 행과 그 분량을 정확히 산정한다.
    """
    print("=" * 80)
    print("[STEP 1] 언어감지 및 번역대상 산정")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"입력 파일 없음: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE, sheet_name="Sheet1")
    print(f"총 {len(df):,}건 로드")

    # ---- fasttext 로드 ----
    use_fasttext = FASTTEXT_MODEL_PATH.exists()
    ft_model = None
    if use_fasttext:
        import fasttext
        fasttext.FastText.eprint = lambda x: None  # warning 숨김
        ft_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))
        print(f"fasttext 모델 로드 완료: {FASTTEXT_MODEL_PATH}")
    else:
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 42
            print(f"⚠ fasttext 모델 파일 없음 ({FASTTEXT_MODEL_PATH})")
            print("  langdetect로 대체 실행 (정확도 다소 낮음).")
            print("  fasttext 권장: wget https://dl.fbaipublicfiles.com/fasttext/"
                  "supervised-models/lid.176.bin")
        except ImportError:
            print("⚠ fasttext, langdetect 모두 없음. ASCII 휴리스틱으로만 동작.")
        use_fasttext = False

    def _ascii_ratio(text: str) -> float:
        """ASCII 문자 비율 반환 (공백 제외)"""
        no_space = text.replace(" ", "")
        if not no_space:
            return 1.0
        return sum(1 for c in no_space if ord(c) < 128) / len(no_space)

    def detect_lang(text):
        """
        텍스트의 언어코드(ISO 639-1) 반환.
        - None/빈값 → 'none'
        - 너무 짧음 → 'unk'
        - fasttext 저신뢰(conf<=0.5) → 'unk'
        - 최종 'unk'이지만 ASCII 비율 >= 0.90 → 'en' 으로 재판정
          (특허 영문 텍스트에서 fasttext가 짧은 텍스트를 unk 반환하는 케이스 보완)
        """
        if pd.isna(text):
            return "none"
        text = str(text).strip()
        if len(text) < 3:
            return "unk"
        clean = text.replace("\n", " ").replace("\r", " ")
        lang = "unk"
        try:
            if use_fasttext and ft_model is not None:
                pred = ft_model.predict(clean, k=1)
                detected = pred[0][0].replace("__label__", "")
                conf = float(pred[1][0])
                if conf > 0.5:
                    lang = detected
                else:
                    lang = f"{detected}?"   # 저신뢰 표시
            else:
                from langdetect import detect as ld_detect
                lang = ld_detect(clean)
        except Exception:
            lang = "unk"

        # ---- unk 보완: ASCII 비율로 영어 재판정 ----
        if lang in ("unk", "unk?"):
            if _ascii_ratio(clean) >= 0.90:
                lang = "en"

        return lang

    print("\nTitle 언어감지 중...")
    tqdm.pandas(desc="Title")
    df["title_lang"] = df["Title"].progress_apply(detect_lang)

    print("\nAbstract 언어감지 중...")
    tqdm.pandas(desc="Abstract")
    df["abstract_lang"] = df["Abstract"].progress_apply(detect_lang)

    # ---- 영어 판정 (신뢰도 표시 '?' 도 영어면 영어로 처리) ----
    def is_english(lang):
        return lang.replace("?", "") == "en"

    df["title_is_en"] = df["title_lang"].apply(is_english)
    df["abstract_is_en"] = df["abstract_lang"].apply(is_english)

    # ---- Tier 분류 (연구계획서 v5.0 STAGE 0-C 기준) ----
    def determine_tier_and_translation_need(row):
        """
        Tier 1  : Abstract 존재 + 영어       → 번역 불필요
        Tier 1-T: Abstract 존재 + 비영어     → Abstract 번역 필요
        Tier 2  : Abstract 결측, Title 존재
                  Title 영어 → 번역 불필요 (CPC 보완은 STAGE0-C)
                  Title 비영어 → Title 번역 필요
        Tier 3  : Abstract·Title 모두 결측/짧음 → Claims 수집 대상
        """
        has_abstract = pd.notna(row["Abstract"]) and len(str(row["Abstract"]).split()) >= 5
        has_title = pd.notna(row["Title"]) and len(str(row["Title"]).strip()) >= 3

        if has_abstract:
            if row["abstract_is_en"]:
                return "tier1", "none"
            else:
                return "tier1", "abstract"
        elif has_title:
            if row["title_is_en"]:
                return "tier2", "none"
            else:
                return "tier2", "title"
        else:
            return "tier3", "pending_claims"

    results = df.apply(determine_tier_and_translation_need, axis=1, result_type="expand")
    df["text_tier"] = results[0]
    df["translation_target"] = results[1]

    # ---- 산정 리포트 ----
    print("\n" + "-" * 80)
    print("[산정 리포트]")
    print("-" * 80)

    tier_counts = df["text_tier"].value_counts()
    print("\nText Tier 분포:")
    print(tier_counts)

    trans_counts = df["translation_target"].value_counts()
    print("\n번역 대상 분포:")
    print(trans_counts)

    n_abstract_translate = (df["translation_target"] == "abstract").sum()
    n_title_translate = (df["translation_target"] == "title").sum()
    n_claims_pending = (df["translation_target"] == "pending_claims").sum()

    # 언어별 분포 (번역 대상만)
    abstract_lang_dist = df.loc[df["translation_target"] == "abstract", "abstract_lang"] \
        .value_counts()
    title_lang_dist = df.loc[df["translation_target"] == "title", "title_lang"] \
        .value_counts()

    print("\nAbstract 번역대상 언어 분포 (상위 10):")
    print(abstract_lang_dist.head(10))
    print("\nTitle 번역대상 언어 분포 (상위 10):")
    print(title_lang_dist.head(10))

    # ---- 작업량(단어수) 산정 ----
    abstract_words = df.loc[df["translation_target"] == "abstract", "Abstract"] \
        .dropna().str.split().apply(len)
    title_words = df.loc[df["translation_target"] == "title", "Title"] \
        .dropna().str.split().apply(len)

    total_words = abstract_words.sum() + title_words.sum()

    # 시간 추정 (14B 모델, 약 15-25 tokens/sec 가정, 토큰≈단어*1.3)
    est_tokens = total_words * 1.3
    est_seconds_low = est_tokens / 25
    est_seconds_high = est_tokens / 15
    est_hours_low = est_seconds_low / 3600
    est_hours_high = est_seconds_high / 3600

    print("\n" + "-" * 80)
    print("[작업량 추정]")
    print("-" * 80)
    print(f"번역대상 Abstract: {n_abstract_translate:,}건 (총 {abstract_words.sum():,} 단어)")
    print(f"번역대상 Title:    {n_title_translate:,}건 (총 {title_words.sum():,} 단어)")
    print(f"Claims 수집대상(Tier3): {n_claims_pending:,}건")
    print(f"\n총 번역 단어수: {total_words:,.0f} 단어")
    print(f"예상 토큰수: {est_tokens:,.0f}")
    print(f"예상 작업시간: {est_hours_low:.1f} ~ {est_hours_high:.1f} 시간 "
          f"(qwen2.5:14b, 단일 GPU 기준, 병렬화 미적용)")
    print("→ 병렬화(2-3 인스턴스) 시 위 시간의 1/2~1/3로 단축 가능")

    # ---- [FIX] 타입 안전화 후 저장 ----
    safe_to_parquet(df, LANG_TAGGED_PATH)
    print(f"\n언어태깅 결과 저장: {LANG_TAGGED_PATH}")

    report = {
        "total_patents": len(df),
        "tier_counts": tier_counts.to_dict(),
        "translation_target_counts": trans_counts.to_dict(),
        "abstract_translate_count": int(n_abstract_translate),
        "title_translate_count": int(n_title_translate),
        "claims_pending_count": int(n_claims_pending),
        "abstract_lang_distribution": abstract_lang_dist.to_dict(),
        "title_lang_distribution": title_lang_dist.to_dict(),
        "total_words_to_translate": int(total_words),
        "estimated_tokens": float(est_tokens),
        "estimated_hours_single_gpu": [round(est_hours_low, 1), round(est_hours_high, 1)],
        "fasttext_used": use_fasttext,
    }
    with open(LANG_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"산정 리포트(JSON) 저장: {LANG_REPORT_PATH}")

    return df


# ==============================================================================
# [2] Ollama 번역 파이프라인
# ==============================================================================

TRANSLATION_SYSTEM_PROMPT = (
    "You are a patent translation specialist with expertise in automotive "
    "and autonomous driving technology (IPC class B60W). "
    "Translate the following patent {field_type} from {lang_name} to English. "
    "Rules:\n"
    "1. Preserve ALL technical terminology exactly and consistently.\n"
    "2. Do NOT paraphrase, summarize, or omit any information.\n"
    "3. Keep the same sentence structure where possible.\n"
    "4. Output ONLY the English translation. No explanations, no notes, "
    "no quotation marks around the output.\n"
)

LANG_NAME_MAP = {
    "ja": "Japanese", "ko": "Korean", "de": "German", "fr": "French",
    "zh": "Chinese", "zh-cn": "Chinese", "ru": "Russian", "es": "Spanish",
    "it": "Italian", "nl": "Dutch", "pt": "Portuguese", "sv": "Swedish",
    "fi": "Finnish", "da": "Danish", "no": "Norwegian", "pl": "Polish",
    "tr": "Turkish", "cs": "Czech", "hu": "Hungarian",
    # fasttext 저신뢰/미감지 → 모델이 언어를 직접 판단
    "unk": "auto-detected",
    "none": "auto-detected",
}

# unk 언어용 프롬프트 (언어 명시 없이 자동판단 번역)
TRANSLATION_SYSTEM_PROMPT_AUTO = (
    "You are a patent translation specialist with expertise in automotive "
    "and autonomous driving technology (IPC class B60W). "
    "The following patent {field_type} is written in a non-English language. "
    "First identify the language, then translate it into English. "
    "Rules:\n"
    "1. Preserve ALL technical terminology exactly and consistently.\n"
    "2. Do NOT paraphrase, summarize, or omit any information.\n"
    "3. Keep the same sentence structure where possible.\n"
    "4. Output ONLY the English translation. No explanations, no language label, "
    "no quotation marks around the output.\n"
)


def clean_lang_code(code):
    """'ja?' → 'ja' 등 신뢰도 표시 제거"""
    return code.replace("?", "")


def ollama_generate(prompt, model, system=None, temperature=0.1,
                    max_retries=3, timeout=120):
    """Ollama REST API 호출. 실패 시 재시도."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except Exception as e:
            if attempt == max_retries:
                print(f"  [오류] Ollama 호출 실패 (시도 {attempt}/{max_retries}): {e}")
                return None
            time.sleep(2 * attempt)
    return None


def translate_text(text, lang_code, field_type="abstract", model=TRANSLATION_MODEL):
    """단일 텍스트 번역. lang_code: ISO 639-1.
    'unk'/'none' 인 경우 모델이 언어를 자동 감지하여 번역한다.
    """
    lang_code = clean_lang_code(lang_code)

    if lang_code in ("unk", "none"):
        # 언어 불명 → 자동판단 프롬프트 사용
        system_prompt = TRANSLATION_SYSTEM_PROMPT_AUTO.format(field_type=field_type)
    else:
        lang_name = LANG_NAME_MAP.get(lang_code, lang_code.upper())
        system_prompt = TRANSLATION_SYSTEM_PROMPT.format(
            field_type=field_type, lang_name=lang_name
        )

    result = ollama_generate(text, model=model, system=system_prompt, temperature=0.1)
    if result is None:
        return None
    result = result.strip().strip('"').strip("'")
    result = re.sub(r"^(Translation|English translation)\s*:\s*", "", result, flags=re.I)
    # "Language: Korean\n" 같은 언어 레이블 접두사 제거
    result = re.sub(r"^(Language\s*:\s*\w+\s*\n+)", "", result, flags=re.I)
    return result.strip()


def text_hash(text, lang_code):
    """캐시 키 생성"""
    return hashlib.md5(f"{lang_code}::{text}".encode("utf-8")).hexdigest()


def load_cache():
    if TRANSLATION_CACHE_PATH.exists():
        with open(TRANSLATION_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(TRANSLATION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def step2_pilot_qa(df, n_samples=20):
    """
    각 주요 언어별 샘플 번역 + 품질확인용 엑셀 출력.
    """
    print("\n" + "=" * 80)
    print("[STEP 2-A] Pilot 번역 QA (언어별 샘플)")
    print("=" * 80)

    pilot_rows = []
    targets = df[df["translation_target"] != "none"].copy()

    for target_col, lang_col, field_name in [
        ("abstract", "abstract_lang", "Abstract"),
        ("title", "title_lang", "Title"),
    ]:
        subset = targets[targets["translation_target"] == target_col]
        for lang, group in subset.groupby(lang_col):
            lang_clean = clean_lang_code(lang)
            if lang_clean not in LANG_NAME_MAP:
                continue  # 진짜 희귀언어(맵 미등록)만 제외; unk/none은 포함됨
            sample = group.head(min(n_samples, len(group)))
            for idx, row in sample.iterrows():
                src_text = row[field_name]
                if pd.isna(src_text) or len(str(src_text).strip()) < 3:
                    continue
                pilot_rows.append({
                    "idx": idx,
                    "field": field_name,
                    "lang": lang_clean,
                    "source_text": src_text,
                    "Display Key": row.get("Display Key", ""),
                })

    print(f"Pilot 대상: {len(pilot_rows)}건")
    print("Ollama 번역 실행 중...")

    for item in tqdm(pilot_rows, desc="Pilot 번역"):
        translated = translate_text(
            item["source_text"], item["lang"],
            field_type=item["field"].lower()
        )
        item["local_llm_translation"] = translated if translated else "[번역실패]"
        item["google_translate_reference"] = ""
        item["technical_term_match_pct"] = ""
        item["notes"] = ""

    pilot_df = pd.DataFrame(pilot_rows)
    pilot_df.to_excel(PILOT_QA_PATH, index=False)
    print(f"\nPilot QA 엑셀 저장: {PILOT_QA_PATH}")
    print("\n다음 단계:")
    print("  1. 'google_translate_reference' 컬럼에 Google Translate 결과 입력")
    print("  2. 'technical_term_match_pct' 컬럼에 핵심기술용어 일치율(%) 입력 (인간평가)")
    print("  3. H0a 검정: 평균 일치율 >= 80% 확인")

    return pilot_df


def step2_full_translation(df, batch_checkpoint_every=200, model=TRANSLATION_MODEL):
    """
    전체 배치 번역 실행. 캐시 + 체크포인트 지원.
    """
    print("\n" + "=" * 80)
    print(f"[STEP 2-B] 전체 배치번역 실행 (model={model})")
    print("=" * 80)

    cache = load_cache()
    print(f"기존 캐시: {len(cache):,}건")

    abstract_targets = df[df["translation_target"] == "abstract"].copy()
    title_targets = df[df["translation_target"] == "title"].copy()

    print(f"Abstract 번역대상: {len(abstract_targets):,}건")
    print(f"Title 번역대상: {len(title_targets):,}건")

    df["abstract_en_translated"] = None
    df["title_en_translated"] = None
    df["translation_status"] = "not_needed"

    # 체크포인트 복원
    if TRANSLATION_CHECKPOINT_PATH.exists():
        ckpt = pd.read_parquet(TRANSLATION_CHECKPOINT_PATH)
        print(f"체크포인트 발견: {len(ckpt):,}건 처리 완료 상태에서 재시작")
        df.update(ckpt[["abstract_en_translated", "title_en_translated",
                         "translation_status"]])

    jobs = []
    for idx, row in abstract_targets.iterrows():
        if df.at[idx, "translation_status"] == "done":
            continue
        jobs.append((idx, "abstract", row["Abstract"], row["abstract_lang"], "Abstract"))
    for idx, row in title_targets.iterrows():
        if df.at[idx, "translation_status"] == "done":
            continue
        jobs.append((idx, "title", row["Title"], row["title_lang"], "Title"))

    print(f"\n실행대상(미완료): {len(jobs):,}건")

    if len(jobs) == 0:
        print("모든 번역이 이미 완료됨.")
        return df

    processed = 0
    for idx, target_col, src_text, lang_code, field_name in tqdm(jobs, desc="번역진행"):
        if pd.isna(src_text) or len(str(src_text).strip()) < 3:
            df.at[idx, "translation_status"] = "skip_empty"
            continue

        lang_clean = clean_lang_code(lang_code)
        # LANG_NAME_MAP에 없는 진짜 희귀언어만 스킵 ('unk'/'none'은 맵에 포함됨)
        if lang_clean not in LANG_NAME_MAP:
            df.at[idx, "translation_status"] = "skip_unsupported_lang"
            continue

        cache_key = text_hash(src_text, lang_clean)

        if cache_key in cache:
            translated = cache[cache_key]
        else:
            translated = translate_text(
                src_text, lang_clean,
                field_type=field_name.lower(), model=model
            )
            if translated is None:
                df.at[idx, "translation_status"] = "failed"
                continue
            cache[cache_key] = translated

        if target_col == "abstract":
            df.at[idx, "abstract_en_translated"] = translated
        else:
            df.at[idx, "title_en_translated"] = translated
        df.at[idx, "translation_status"] = "done"

        processed += 1
        if processed % batch_checkpoint_every == 0:
            safe_to_parquet(df, TRANSLATION_CHECKPOINT_PATH)   # [FIX]
            save_cache(cache)
            tqdm.write(f"  [체크포인트] {processed:,}건 처리 → 저장완료 "
                       f"(캐시 {len(cache):,}건)")

    # 최종 저장
    save_cache(cache)
    safe_to_parquet(df, TRANSLATION_OUTPUT_PATH)               # [FIX]
    if TRANSLATION_CHECKPOINT_PATH.exists():
        TRANSLATION_CHECKPOINT_PATH.unlink()

    status_counts = df["translation_status"].value_counts()
    print("\n" + "-" * 80)
    print("[번역 완료 리포트]")
    print("-" * 80)
    print(status_counts)
    print(f"\n최종결과 저장: {TRANSLATION_OUTPUT_PATH}")
    print(f"캐시 크기: {len(cache):,}건 (재실행시 재사용)")

    return df


# ==============================================================================
# [3] 출원인 그룹분류 (STAGE 0-B)
# ==============================================================================

KNOWN_GROUP_MAPPING = {
    # Toyota Group
    "TOYOTA MOTOR CO LTD": "Toyota Group",
    "TOYOTA MOTOR CORP": "Toyota Group",
    "TOYOTA ENG & MFG NORTH AMERICA": "Toyota Group",
    "TOYOTA JIDOSHA KABUSHIKI KAISHA": "Toyota Group",
    "AISIN AW CO": "Toyota Group",
    "AISIN SEIKI": "Toyota Group",
    "AISIN CORP": "Toyota Group",
    "DENSO CORP": "Toyota Group",
    "DAIHATSU MOTOR": "Toyota Group",
    "HINO MOTORS": "Toyota Group",
    # Honda
    "HONDA MOTOR CO LTD": "Honda Group",
    "HONDA R&D CO LTD": "Honda Group",
    # Nissan
    "NISSAN MOTOR": "Nissan Group",
    "NISSAN MOTOR CO LTD": "Nissan Group",
    # Hyundai Group
    "HYUNDAI MOTOR CO LTD": "Hyundai Motor Group",
    "HYUNDAI MOTOR COMPANY": "Hyundai Motor Group",
    "HYUNDAI MOBIS": "Hyundai Motor Group",
    "현대자동차주식회사": "Hyundai Motor Group",
    "현대자동차 주식회사": "Hyundai Motor Group",
    "현대모비스 주식회사": "Hyundai Motor Group",
    "KIA MOTORS CORP": "Hyundai Motor Group",
    "KIA CORP": "Hyundai Motor Group",
    "기아 주식회사": "Hyundai Motor Group",
    "엘지전자 주식회사": "LG Group",
    # Ford
    "FORD GLOBAL TECH LLC": "Ford Group",
    "FORD MOTOR CO": "Ford Group",
    # GM
    "GM GLOBAL TECH OPERATIONS INC": "GM Group",
    "GM GLOBAL TECH OPERATIONS LLC": "GM Group",
    "GENERAL MOTORS": "GM Group",
    # Bosch
    "BOSCH GMBH ROBERT": "Bosch Group",
    "ROBERT BOSCH GMBH": "Bosch Group",
    # Volkswagen
    "VOLKSWAGEN AG": "Volkswagen Group",
    "AUDI AG": "Volkswagen Group",
    "PORSCHE": "Volkswagen Group",
    # BMW
    "BAYERISCHE MOTOREN WERKE AG": "BMW Group",
    "BMW AG": "BMW Group",
    # Waymo / Google
    "WAYMO LLC": "Waymo/Alphabet",
    "GOOGLE LLC": "Waymo/Alphabet",
    # Others
    "STATE FARM MUTUAL AUTOMOBILE INSURANCE CO": "State Farm (Non-OEM)",
    "DEERE & CO": "Deere & Co (Non-OEM)",
    "ZAHNRADFABRIK FRIEDRICHSHAFEN": "ZF Group",
    "SCANIA CV AB": "Scania/Traton Group",
    "RENAULT SAS": "Renault Group",
    "RENAULT SA": "Renault Group",
    "PEUGEOT CITROEN AUTOMOBILES SA": "Stellantis Group",
    "IBM": "IBM",
    "GEN ELECTRIC": "GE",
    "LUK LAMELLEN & KUPPLUNGSBAU": "Schaeffler Group",
    "HERE GLOBAL BV": "HERE/Nokia",
}

FEW_SHOT_EXAMPLES = [
    ("TOYOTA MOTOR CO LTD", "Toyota Group"),
    ("AISIN SEIKI", "Toyota Group"),
    ("トヨタ自動車株式会社", "Toyota Group"),
    ("HYUNDAI MOTOR CO LTD", "Hyundai Motor Group"),
    ("현대모비스 주식회사", "Hyundai Motor Group"),
    ("KIA MOTORS CORP", "Hyundai Motor Group"),
    ("GM GLOBAL TECH OPERATIONS LLC", "GM Group"),
    ("FORD GLOBAL TECH LLC", "Ford Group"),
    ("ROBERT BOSCH GMBH", "Bosch Group"),
    ("AUDI AG", "Volkswagen Group"),
    ("WAYMO LLC", "Waymo/Alphabet"),
    ("STATE FARM MUTUAL AUTOMOBILE INSURANCE CO", "State Farm (Non-OEM)"),
    ("HEAP ANTHONY H", "Individual Inventor (Non-Group)"),
]

CLASSIFICATION_SYSTEM_PROMPT = """You are a corporate ownership analyst specializing in the automotive industry.
Given an applicant/assignee name from a patent (which may be in English, Japanese, Korean, or German),
classify it into its parent corporate GROUP for automotive patent analysis.

Rules:
1. If the name belongs to a known automotive OEM or its subsidiaries/affiliates, output the
   parent group name in the format "X Group" (e.g., "Toyota Group", "Hyundai Motor Group").
2. If the name is a major auto-parts supplier (e.g., Bosch, ZF, Continental, Denso), classify
   it as its own group (e.g., "Bosch Group") unless it is a known subsidiary of an OEM.
3. If the name appears to be an individual person's name (not a company), output
   "Individual Inventor (Non-Group)".
4. If the name is a non-automotive company (insurance, tech, agriculture, etc.), output
   the company's own name followed by " (Non-OEM)", e.g., "State Farm (Non-OEM)".
5. If you are NOT confident about the classification, output "UNCERTAIN: <your best guess>".
6. Output ONLY the classification label. No explanation.

Examples:
""" + "\n".join([f'Input: "{name}"\nOutput: {group}' for name, group in FEW_SHOT_EXAMPLES])


def classify_applicant(name, model=CLASSIFICATION_MODEL):
    """단일 출원인명을 그룹으로 분류"""
    name_clean = name.strip()

    # 1단계: 사전매칭 (대소문자 무시 정확일치)
    name_upper = name_clean.upper()
    for k, v in KNOWN_GROUP_MAPPING.items():
        if k.upper() == name_upper:
            return v, "dictionary"

    # 2단계: LLM 분류
    prompt = f'Input: "{name_clean}"\nOutput:'
    result = ollama_generate(
        prompt, model=model,
        system=CLASSIFICATION_SYSTEM_PROMPT, temperature=0.0
    )
    if result is None:
        return "ERROR", "llm_failed"

    result = result.strip().strip('"')
    if result.upper().startswith("UNCERTAIN"):
        return result, "llm_uncertain"
    return result, "llm"


def split_applicants(val):
    """';;' 구분 멀티값 분리"""
    if pd.isna(val):
        return []
    return [x.strip() for x in str(val).split(";;") if x.strip()]


def step3_applicant_grouping(df):
    """
    출원인(Applicants) + 소유자(Owners) 필드를 결합하여
    개별 엔티티를 추출하고, 사전매칭 우선 + LLM fallback으로 그룹 분류.
    """
    print("\n" + "=" * 80)
    print("[STEP 3] 출원인 그룹 분류 (STAGE 0-B)")
    print("=" * 80)

    df["applicants_list"] = df["Applicants"].apply(split_applicants)
    df["owners_list"] = df["Owners"].apply(split_applicants)

    def strip_date_suffix(name):
        return re.sub(r"\s*\(\d{4}-\d{2}-\d{2}\)\s*$", "", name).strip()

    df["owners_list_clean"] = df["owners_list"].apply(
        lambda lst: [strip_date_suffix(x) for x in lst]
    )

    # 분류 대상 엔티티 집합
    all_entities = set()
    for lst in df["applicants_list"]:
        all_entities.update(lst)
    for lst in df["owners_list_clean"]:
        all_entities.update(lst)
    all_entities.discard("")

    print(f"고유 엔티티 수: {len(all_entities):,}")

    # 빈도 계산
    freq = defaultdict(int)
    for lst in df["applicants_list"]:
        for e in lst:
            freq[e] += 1
    for lst in df["owners_list_clean"]:
        for e in lst:
            freq[e] += 1

    sorted_entities = sorted(all_entities, key=lambda x: -freq[x])

    # ---- 분류 실행 (캐시 활용) ----
    classification_cache_path = WORK_DIR / "step3_classification_cache.json"
    if classification_cache_path.exists():
        with open(classification_cache_path, "r", encoding="utf-8") as f:
            classification_cache = json.load(f)
    else:
        classification_cache = {}

    print(f"기존 분류캐시: {len(classification_cache):,}건")

    uncertain_entities = []
    print("엔티티 분류 진행 중 (빈도 높은 순)...")
    for entity in tqdm(sorted_entities, desc="분류"):
        if entity in classification_cache:
            continue
        group, method = classify_applicant(entity)
        classification_cache[entity] = {
            "group": group,
            "method": method,
            "frequency": freq[entity],
        }
        if method in ("llm_uncertain", "llm_failed"):
            uncertain_entities.append(entity)

        if len(classification_cache) % 100 == 0:
            with open(classification_cache_path, "w", encoding="utf-8") as f:
                json.dump(classification_cache, f, ensure_ascii=False, indent=2)

    with open(classification_cache_path, "w", encoding="utf-8") as f:
        json.dump(classification_cache, f, ensure_ascii=False, indent=2)

    # ---- 특허 단위 그룹 매핑 ----
    def get_groups_for_patent(row):
        entities = list(row["applicants_list"]) + list(row["owners_list_clean"])
        entities = [e for e in entities if e]
        if not entities:
            return None, []

        groups = []
        for e in entities:
            info = classification_cache.get(e, {"group": "UNKNOWN"})
            g = info["group"]
            if not g.upper().startswith("UNCERTAIN") and g != "UNKNOWN":
                groups.append(g)

        if not groups:
            return "UNKNOWN", entities

        group_counts = pd.Series(groups).value_counts()
        primary_group = group_counts.index[0]
        return primary_group, groups

    print("\n특허별 그룹 매핑 중...")
    tqdm.pandas(desc="매핑")
    mapping_results = df.progress_apply(
        get_groups_for_patent, axis=1, result_type="expand"
    )
    df["applicant_group_primary"] = mapping_results[0]
    df["applicant_groups_all"] = mapping_results[1]

    # ---- Top 15 비교 리포트 ----
    print("\n" + "-" * 80)
    print("[Top 15 출원인 비교: 원본명 vs 그룹통합]")
    print("-" * 80)

    raw_top = pd.Series(
        [lst[0] for lst in df["applicants_list"] if lst]
    ).value_counts().head(15)

    grouped_top = df["applicant_group_primary"].value_counts().head(15)

    print("\n[원본명 기준 Top 15]")
    print(raw_top)
    print("\n[그룹통합 기준 Top 15]")
    print(grouped_top)

    comparison_df = pd.DataFrame({
        "원본명_순위": range(1, len(raw_top) + 1),
        "원본명": raw_top.index,
        "원본명_건수": raw_top.values,
    })
    grouped_df = pd.DataFrame({
        "그룹통합_순위": range(1, len(grouped_top) + 1),
        "그룹명": grouped_top.index,
        "그룹_건수": grouped_top.values,
    })
    comparison_combined = pd.concat([comparison_df, grouped_df], axis=1)

    with pd.ExcelWriter(TOP10_REPORT_PATH) as writer:
        comparison_combined.to_excel(writer, sheet_name="Top10_비교", index=False)

        detail_rows = []
        for entity, info in classification_cache.items():
            detail_rows.append({
                "entity": entity,
                "frequency": info["frequency"],
                "classified_group": info["group"],
                "method": info["method"],
            })
        detail_df = pd.DataFrame(detail_rows).sort_values("frequency", ascending=False)
        detail_df.to_excel(writer, sheet_name="전체분류상세", index=False)

    print(f"\nTop15 비교 + 전체분류상세 저장: {TOP10_REPORT_PATH}")

    # ---- 수동검토 리스트 ----
    review_rows = []
    for entity, info in classification_cache.items():
        if (info["method"] in ("llm_uncertain", "llm_failed")
                or info["group"] == "UNKNOWN"):
            review_rows.append({
                "entity": entity,
                "frequency": info["frequency"],
                "classified_group": info["group"],
                "method": info["method"],
                "manual_group_override": "",
            })
    review_df = pd.DataFrame(review_rows).sort_values("frequency", ascending=False)
    review_df.to_excel(APPLICANT_REVIEW_PATH, index=False)

    n_review = len(review_df)
    n_review_high_freq = (review_df["frequency"] >= 5).sum() if n_review > 0 else 0
    print(f"\n수동검토 필요 엔티티: {n_review:,}건 "
          f"(빈도≥5: {n_review_high_freq:,}건 — 우선검토 대상)")
    print(f"수동검토 리스트 저장: {APPLICANT_REVIEW_PATH}")

    # ---- 최종 저장 ----
    safe_to_parquet(df, APPLICANT_GROUP_OUTPUT_PATH)           # [FIX]
    print(f"\n최종결과 저장: {APPLICANT_GROUP_OUTPUT_PATH}")

    return df


# ==============================================================================
# 메인 실행
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="STAGE 0 통합 파이프라인")
    parser.add_argument(
        "--step", choices=["1", "2", "3", "all"], default="all",
        help="실행할 단계 (1=언어감지, 2=번역, 3=출원인분류, all=전체)"
    )
    parser.add_argument(
        "--pilot-only", action="store_true",
        help="step 2에서 Pilot QA만 실행 (전체번역 생략)"
    )
    parser.add_argument(
        "--n-pilot-samples", type=int, default=20,
        help="언어별 pilot 샘플 수"
    )
    args = parser.parse_args()

    df = None

    if args.step in ("1", "all"):
        df = step1_language_detection()

    if args.step in ("2", "all"):
        if df is None:
            if not LANG_TAGGED_PATH.exists():
                raise FileNotFoundError("Step 1 결과 없음. 먼저 --step 1 실행 필요.")
            df = pd.read_parquet(LANG_TAGGED_PATH)

        step2_pilot_qa(df, n_samples=args.n_pilot_samples)

        if not args.pilot_only:
            print("\n" + "!" * 80)
            print("전체 배치번역을 시작합니다. 데이터 규모에 따라 수십 시간 소요될 수 있습니다.")
            print("중단해도 체크포인트에서 재시작 가능합니다.")
            print("!" * 80)
            df = step2_full_translation(df)

    if args.step in ("3", "all"):
        if df is None:
            if TRANSLATION_OUTPUT_PATH.exists():
                df = pd.read_parquet(TRANSLATION_OUTPUT_PATH)
            elif LANG_TAGGED_PATH.exists():
                df = pd.read_parquet(LANG_TAGGED_PATH)
            else:
                df = pd.read_excel(INPUT_FILE, sheet_name="Sheet1")

        df = step3_applicant_grouping(df)

    print("\n" + "=" * 80)
    print("파이프라인 실행 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()