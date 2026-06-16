"""
================================================================================
STAGE 1 언어그래프 구축 파이프라인 v1.0
================================================================================
연구계획서 v5.0 - Month 3 (Stage 0 완료 후 즉시 착수)

입력:
  - stage0_work/step2_translated.parquet   (번역 완료 텍스트)
  - stage0_work/step3_applicant_groups.parquet  (출원인 그룹 분류)
  ※ step2 미완료 시 step1_lang_tagged.parquet 자동 fallback

산출물:
  - stage1_work/
      s1_corpus.parquet          # 통합 텍스트(final_text) + text_tier + 분기 라벨
      s1_keywords.json           # 전체 키워드 1,000개 + 임베딩
      s1_node_features.parquet   # (T × N × 8) 노드 특징 행렬 (long format)
      s1_edges.parquet           # (T) 분기별 엣지 리스트 (PMI 기반 공출현)
      s1_graph_meta.json         # 분기 목록, 노드 목록, 파이프라인 메타
      s1_quarter_summary.xlsx    # 분기별 특허수·노드수·엣지수 요약표

포함 단계:
  [A] 코퍼스 통합  — Stage 0 산출물 결합, final_text / text_tier 확정
  [B] 분기 라벨링  — Application/Publication Date → 연도Q 분기 코드
  [C] 임베딩       — Ollama nomic-embed-text (로컬) 또는 sentence-transformers fallback
  [D] 키워드 추출  — KeyBERT (MMR), 전체 코퍼스 기준 Top 1,000 노드 선정
  [E] 분기 그래프  — PMI 기반 공출현 엣지 + 인용 엣지
  [F] 노드 특징    — f1~f8 8개 특징 행렬 구축
  [G] 요약 리포트  — 분기별 통계 + 분기 범위 확정(100 vs 120분기)

실행 방법:
  python stage1_graph_pipeline.py --step A       # 코퍼스 통합만
  python stage1_graph_pipeline.py --step B       # 분기 라벨링
  python stage1_graph_pipeline.py --step C       # 임베딩 (시간 소요)
  python stage1_graph_pipeline.py --step D       # 키워드 추출
  python stage1_graph_pipeline.py --step E       # 그래프 구축
  python stage1_graph_pipeline.py --step F       # 노드 특징 행렬
  python stage1_graph_pipeline.py --step all     # 전체 순차 실행
  python stage1_graph_pipeline.py --step all --embed-model sentence-transformers
  python stage1_graph_pipeline.py --step all --n-keywords 1000 --start-year 1995

사전 요구사항:
  pip install keybert sentence-transformers scikit-learn networkx scipy \
              openpyxl pyarrow tqdm pandas numpy requests --break-system-packages
  (임베딩 Ollama 사용 시) ollama pull nomic-embed-text
================================================================================
"""

import os
import re
import json
import time
import argparse
import hashlib
import warnings
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==============================================================================
# 공통 설정
# ==============================================================================
BASE_DIR = Path.cwd()
WORK_DIR = BASE_DIR / "stage0_work"
OUT_DIR = BASE_DIR / "stage1_work"
OUT_DIR.mkdir(exist_ok=True)

# 입력 파일 (우선순위 순)
INPUT_CANDIDATES = [
    WORK_DIR / "step3_applicant_groups.parquet",   # Stage 0 완전 완료
    WORK_DIR / "step2_translated.parquet",          # 번역까지만 완료
    WORK_DIR / "step1_lang_tagged.parquet",         # 언어태깅만 완료
]

# 산출물 경로
CORPUS_PATH = OUT_DIR / "s1_corpus.parquet"
KEYWORDS_PATH = OUT_DIR / "s1_keywords.json"
EMBED_CACHE_PATH = OUT_DIR / "s1_embed_cache.json"
NODE_FEATURES_PATH = OUT_DIR / "s1_node_features.parquet"
EDGES_PATH = OUT_DIR / "s1_edges.parquet"
GRAPH_META_PATH = OUT_DIR / "s1_graph_meta.json"
QUARTER_SUMMARY_PATH = OUT_DIR / "s1_quarter_summary.xlsx"

# 분기 범위 기본값 (--start-year 인자로 변경 가능)
DEFAULT_START_YEAR = 1995
DEFAULT_END_YEAR = 2024

# 노드 수 (KeyBERT Top-K 키워드)
DEFAULT_N_KEYWORDS = 1000

# Ollama 설정
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# PMI 엣지 최소 공출현 횟수 / PMI 임계값
MIN_COOCCUR = 3
PMI_THRESHOLD = 0.0     # PMI > 0 이면 양의 연관

# 노드 특징 정의 (f1~f8)
# f1: 분기 내 출현 빈도 (TF)
# f2: 전체 코퍼스 IDF
# f3: 평균 forward citation (피인용수)
# f4: 평균 backward citation (인용수)
# f5: 특허 나이 (출원연도 대비 분기)
# f6: CPC 서브그룹 다양성 (Shannon entropy)
# f7: 출원인 그룹 다양성 (고유 그룹 수)
# f8: text_tier (1=Abstract, 2=Title+CPC, 3=Claims)


# ==============================================================================
# 유틸리티
# ==============================================================================

def safe_to_parquet(df: pd.DataFrame, path):
    """pyarrow 타입 오류 방지 저장"""
    df_save = df.copy()
    for col in df_save.columns:
        if df_save[col].dtype == object:
            sample = df_save[col].dropna()
            if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict)):
                df_save[col] = df_save[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False)
                    if x is not None and not (isinstance(x, float) and np.isnan(x))
                    else None
                )
            else:
                df_save[col] = df_save[col].where(df_save[col].notna(), None)
                df_save[col] = df_save[col].astype(str).where(df_save[col].notna(), None)
    df_save.to_parquet(path, index=False)
    return path


def quarter_label(year: int, q: int) -> str:
    """(2020, 3) → '2020Q3'"""
    return f"{year}Q{q}"


def date_to_quarter(date_val) -> tuple:
    """
    다양한 형식의 날짜 값을 (year, quarter) 로 변환.
    지원 형식: '20200315', '2020-03-15', '2020/03', datetime, int(YYYYMMDD)
    실패 시 (None, None) 반환.
    """
    if pd.isna(date_val):
        return None, None
    s = str(date_val).strip()
    # 숫자 8자리: YYYYMMDD
    if re.match(r"^\d{8}$", s):
        try:
            year = int(s[:4])
            month = int(s[4:6])
            return year, (month - 1) // 3 + 1
        except Exception:
            return None, None
    # YYYY-MM-DD 또는 YYYY/MM/DD
    m = re.match(r"^(\d{4})[-/](\d{2})", s)
    if m:
        try:
            year = int(m.group(1))
            month = int(m.group(2))
            return year, (month - 1) // 3 + 1
        except Exception:
            return None, None
    # YYYY 만 있는 경우 → Q1 기본
    m = re.match(r"^(\d{4})$", s)
    if m:
        return int(m.group(1)), 1
    return None, None


# ==============================================================================
# [A] 코퍼스 통합
# ==============================================================================

def step_A_corpus(start_year=DEFAULT_START_YEAR, end_year=DEFAULT_END_YEAR):
    """
    Stage 0 산출물을 로드하여 final_text / text_tier 컬럼을 확정한다.

    final_text 우선순위:
      Tier 1: abstract_en_translated (비영어 번역본) > Abstract (영어 원문)
      Tier 2: title_en_translated > Title  (Abstract 없을 때)
      Tier 3: [Claims 수집 전이면 Title 유지, 추후 대체]

    text_tier:
      1 = Abstract 사용
      2 = Title (+CPC 보완은 Step G에서 pseudo-abstract로 확장 예정)
      3 = 매우 짧음 / Claims 수집 필요
    """
    print("=" * 80)
    print("[STEP A] 코퍼스 통합 및 final_text 확정")
    print("=" * 80)

    # 입력 파일 선택
    src = None
    for candidate in INPUT_CANDIDATES:
        if candidate.exists():
            src = candidate
            break
    if src is None:
        raise FileNotFoundError(
            "Stage 0 결과 파일 없음. stage0_pipeline.py를 먼저 실행하세요."
        )
    print(f"입력 파일: {src}")
    df = pd.read_parquet(src)
    print(f"로드: {len(df):,}건, 컬럼: {list(df.columns)}")

    # ── final_text 확정 ──────────────────────────────────────────────
    def build_final_text(row):
        # Abstract 계열
        abstract_en = None
        for col in ("abstract_en_translated", "Abstract"):
            if col in row and pd.notna(row[col]) and len(str(row[col]).split()) >= 5:
                abstract_en = str(row[col]).strip()
                break
        if abstract_en:
            return abstract_en, 1

        # Title 계열
        title_en = None
        for col in ("title_en_translated", "Title"):
            if col in row and pd.notna(row[col]) and len(str(row[col]).strip()) >= 3:
                title_en = str(row[col]).strip()
                break
        if title_en:
            return title_en, 2

        return "", 3

    print("final_text 구성 중...")
    results = df.apply(build_final_text, axis=1, result_type="expand")
    df["final_text"] = results[0]
    df["text_tier"] = results[1].astype(int)

    tier_dist = df["text_tier"].value_counts().sort_index()
    print("\ntext_tier 분포:")
    for t, cnt in tier_dist.items():
        print(f"  Tier {t}: {cnt:,}건 ({cnt/len(df)*100:.1f}%)")

    # ── 분기 라벨 부여 ────────────────────────────────────────────────
    # 날짜 컬럼 자동 탐지
    date_col = None
    for candidate_col in ("Application Date", "Priority Date",
                           "Publication Date", "Filing Date"):
        if candidate_col in df.columns:
            date_col = candidate_col
            break
    if date_col is None:
        # 날짜처럼 보이는 컬럼 자동 탐지
        for col in df.columns:
            sample = df[col].dropna().head(10).astype(str)
            if sample.str.match(r"\d{4}").any():
                date_col = col
                break

    print(f"\n날짜 컬럼 사용: {date_col}")

    if date_col:
        parsed = df[date_col].apply(date_to_quarter)
        df["year"] = parsed.apply(lambda x: x[0])
        df["quarter"] = parsed.apply(lambda x: x[1])
    else:
        print("⚠ 날짜 컬럼 없음 — year/quarter = None")
        df["year"] = None
        df["quarter"] = None

    df["quarter_label"] = df.apply(
        lambda r: quarter_label(int(r["year"]), int(r["quarter"]))
        if pd.notna(r["year"]) and pd.notna(r["quarter"])
        else "UNKNOWN",
        axis=1
    )

    # 분기 범위 필터
    mask = (
        df["year"].notna()
        & (df["year"] >= start_year)
        & (df["year"] <= end_year)
    )
    df_filtered = df[mask].copy()
    n_out = len(df) - len(df_filtered)
    print(f"\n분기 필터 ({start_year}~{end_year}): {len(df_filtered):,}건 유지, "
          f"{n_out:,}건 제외")

    # 빈 final_text 경고
    empty_text = (df_filtered["final_text"].str.strip() == "").sum()
    print(f"final_text 빈값: {empty_text:,}건 (Tier3, Claims 수집 필요)")

    # 저장
    safe_to_parquet(df_filtered, CORPUS_PATH)
    print(f"\n코퍼스 저장: {CORPUS_PATH}  ({len(df_filtered):,}건)")
    return df_filtered


# ==============================================================================
# [B] 분기별 통계 요약
# ==============================================================================

def step_B_quarter_summary(df=None):
    """
    분기별 특허 수 분포를 산출하고, 분기 범위(100 vs 120분기)를 확정한다.
    """
    print("\n" + "=" * 80)
    print("[STEP B] 분기 라벨링 & 분기별 통계 요약")
    print("=" * 80)

    if df is None:
        df = pd.read_parquet(CORPUS_PATH)

    q_counts = (
        df.groupby("quarter_label")
        .agg(
            patent_count=("final_text", "count"),
            tier1_count=("text_tier", lambda x: (x == 1).sum()),
            tier2_count=("text_tier", lambda x: (x == 2).sum()),
            tier3_count=("text_tier", lambda x: (x == 3).sum()),
        )
        .reset_index()
        .sort_values("quarter_label")
    )
    q_counts["year"] = q_counts["quarter_label"].str[:4].astype(int)
    q_counts["q"] = q_counts["quarter_label"].str[-1].astype(int)

    print(f"\n전체 분기 수: {len(q_counts):,}")
    print(f"최초 분기: {q_counts['quarter_label'].min()}")
    print(f"최종 분기: {q_counts['quarter_label'].max()}")
    print(f"분기당 평균 특허수: {q_counts['patent_count'].mean():.1f}")
    print(f"분기당 최소/최대: {q_counts['patent_count'].min()} / "
          f"{q_counts['patent_count'].max()}")

    # 연도별 집계
    yearly = q_counts.groupby("year")["patent_count"].sum().reset_index()
    print("\n연도별 특허수 (상위/하위 각 5년):")
    print(yearly.sort_values("patent_count", ascending=False).head(5).to_string(index=False))
    print("...")
    print(yearly.sort_values("patent_count").head(5).to_string(index=False))

    # 100분기 vs 120분기 검토
    q_2000 = q_counts[q_counts["year"] >= 2000]
    q_1995 = q_counts[q_counts["year"] >= 1995]
    print(f"\n[분기 범위 검토]")
    print(f"  2000-2024 (100분기): {len(q_2000):,}분기, "
          f"총 {q_2000['patent_count'].sum():,}건")
    print(f"  1995-2024 (120분기): {len(q_1995):,}분기, "
          f"총 {q_1995['patent_count'].sum():,}건")

    # 엑셀 저장
    with pd.ExcelWriter(QUARTER_SUMMARY_PATH) as writer:
        q_counts.to_excel(writer, sheet_name="분기별통계", index=False)
        yearly.to_excel(writer, sheet_name="연도별통계", index=False)
    print(f"\n분기별 요약 저장: {QUARTER_SUMMARY_PATH}")
    return q_counts


# ==============================================================================
# [C] 임베딩
# ==============================================================================

def _embed_ollama(texts: list, model=OLLAMA_EMBED_MODEL, batch_size=32) -> np.ndarray:
    """Ollama 임베딩 API 호출 (배치)"""
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Ollama 임베딩"):
        batch = texts[i: i + batch_size]
        vecs = []
        for text in batch:
            try:
                resp = requests.post(
                    OLLAMA_URL,
                    json={"model": model, "prompt": text[:2048]},
                    timeout=60,
                )
                resp.raise_for_status()
                vecs.append(resp.json()["embedding"])
            except Exception as e:
                print(f"  임베딩 오류: {e} → 영벡터로 대체")
                vecs.append([0.0] * 768)  # nomic-embed-text 차원
        all_vecs.extend(vecs)
    return np.array(all_vecs, dtype=np.float32)


def _embed_sbert(texts: list, model_name="paraphrase-multilingual-MiniLM-L12-v2") -> np.ndarray:
    """sentence-transformers fallback 임베딩"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)


def step_C_embedding(df=None, embed_model="ollama"):
    """
    final_text를 임베딩하여 EMBED_CACHE_PATH에 저장한다.
    embed_model: 'ollama' | 'sentence-transformers'
    캐싱: 텍스트 MD5 기준으로 이미 임베딩된 것은 재사용.
    """
    print("\n" + "=" * 80)
    print(f"[STEP C] 임베딩 ({embed_model})")
    print("=" * 80)

    if df is None:
        df = pd.read_parquet(CORPUS_PATH)

    # 빈 텍스트 제외
    valid = df[df["final_text"].str.strip() != ""].copy()
    print(f"임베딩 대상: {len(valid):,}건 (Tier3 빈값 제외)")

    # 캐시 로드
    if EMBED_CACHE_PATH.exists():
        with open(EMBED_CACHE_PATH, "r") as f:
            cache = json.load(f)
        print(f"기존 캐시: {len(cache):,}건")
    else:
        cache = {}

    # 미캐싱 텍스트만 추출
    texts_to_embed = []
    keys_to_embed = []
    for _, row in valid.iterrows():
        text = row["final_text"][:2048]
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key not in cache:
            texts_to_embed.append(text)
            keys_to_embed.append(key)

    print(f"신규 임베딩 필요: {len(texts_to_embed):,}건")

    if texts_to_embed:
        if embed_model == "ollama":
            try:
                vecs = _embed_ollama(texts_to_embed)
            except Exception as e:
                print(f"  Ollama 임베딩 실패: {e}")
                print("  sentence-transformers로 fallback...")
                vecs = _embed_sbert(texts_to_embed)
        else:
            vecs = _embed_sbert(texts_to_embed)

        for key, vec in zip(keys_to_embed, vecs):
            cache[key] = vec.tolist()

        with open(EMBED_CACHE_PATH, "w") as f:
            json.dump(cache, f)
        print(f"캐시 저장: {EMBED_CACHE_PATH}  ({len(cache):,}건)")

    # df에 임베딩 키 컬럼 추가
    df["embed_key"] = df["final_text"].apply(
        lambda t: hashlib.md5(str(t)[:2048].encode("utf-8")).hexdigest()
        if pd.notna(t) and str(t).strip() != "" else None
    )
    safe_to_parquet(df, CORPUS_PATH)
    print("corpus에 embed_key 컬럼 추가 저장 완료")
    return df, cache


# ==============================================================================
# [D] 키워드 추출 (KeyBERT)
# ==============================================================================

def step_D_keywords(df=None, cache=None, n_keywords=DEFAULT_N_KEYWORDS):
    """
    전체 코퍼스에서 KeyBERT MMR로 상위 키워드 n_keywords개를 추출하고
    각 키워드의 임베딩 벡터도 저장한다.

    저장 구조 (KEYWORDS_PATH):
    {
      "keywords": ["keyword1", ...],       # 길이 N
      "keyword_embeddings": [[...], ...],  # (N, D) float list
      "keyword_doc_freq": {"kw": count},   # 문서 빈도
    }
    """
    print("\n" + "=" * 80)
    print(f"[STEP D] 키워드 추출 (KeyBERT, Top {n_keywords})")
    print("=" * 80)

    if df is None:
        df = pd.read_parquet(CORPUS_PATH)

    if KEYWORDS_PATH.exists():
        with open(KEYWORDS_PATH) as f:
            existing = json.load(f)
        if len(existing.get("keywords", [])) >= n_keywords:
            print(f"기존 키워드 {len(existing['keywords'])}개 로드 (재추출 불필요)")
            return existing

    # 전체 텍스트 수집
    texts = df[df["final_text"].str.strip() != ""]["final_text"].tolist()
    print(f"키워드 추출 대상: {len(texts):,}건")

    # ── KeyBERT 설정 ─────────────────────────────────────────────────
    try:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("pip install keybert sentence-transformers 설치 필요")
        raise

    # 임베딩 모델 (캐시가 있으면 재활용, 없으면 multilingual 모델 사용)
    sbert_model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    kw_model = KeyBERT(model=sbert_model)

    print("전체 코퍼스 대상 키워드 추출 중 (n-gram 1-3, MMR)...")

    # 샘플링: 53,000건 전체를 KeyBERT에 넣으면 RAM 부담 — 전체 텍스트를 하나로 연결
    # 실용 전략: TF 기반 후보 필터 후 KeyBERT로 MMR 다양성 선택
    # Step 1: TF 기반 유니그램/바이그램 후보 수집 (scikit-learn CountVectorizer)
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    print("  TF-IDF 후보 키워드 추출 중...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=5,            # 최소 5건 이상 등장
        max_df=0.8,          # 80% 이상 문서에 등장하면 제거 (너무 보편적)
        max_features=20000,  # 상위 2만개 후보
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",  # 영문 3글자 이상
    )
    tfidf_matrix = tfidf.fit_transform(texts)
    vocab = tfidf.get_feature_names_out()
    mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_idx = mean_tfidf.argsort()[::-1][: n_keywords * 3]  # 3배 후보
    candidate_keywords = [vocab[i] for i in top_idx]
    print(f"  TF-IDF 후보: {len(candidate_keywords):,}개")

    # Step 2: KeyBERT MMR로 다양성 선택
    # 전체 텍스트를 하나의 대표 문서로 연결 (처음 1만 건만 샘플)
    sample_text = " ".join(texts[:10000])
    print(f"  KeyBERT MMR 적용 중 (샘플 10,000건, Top {n_keywords})...")
    keywords_scored = kw_model.extract_keywords(
        sample_text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_mmr=True,
        diversity=0.5,
        top_n=n_keywords,
        candidates=candidate_keywords,
    )
    keywords = [kw for kw, _ in keywords_scored]
    print(f"  추출 완료: {len(keywords):,}개")

    # Step 3: 문서 빈도 계산
    print("  키워드 문서빈도 계산 중...")
    from sklearn.feature_extraction.text import CountVectorizer as CV
    cv = CV(vocabulary=keywords, ngram_range=(1, 3), token_pattern=r"(?u)\b\w[\w\-]+\b")
    doc_kw_matrix = cv.fit_transform(texts)
    doc_freq = {kw: int((doc_kw_matrix[:, i] > 0).sum())
                for i, kw in enumerate(keywords)}

    # Step 4: 키워드 임베딩 벡터
    print("  키워드 임베딩 중...")
    kw_embeddings = sbert_model.encode(
        keywords, batch_size=128, show_progress_bar=True,
        convert_to_numpy=True
    ).tolist()

    result = {
        "keywords": keywords,
        "keyword_embeddings": kw_embeddings,
        "keyword_doc_freq": doc_freq,
        "n_keywords": len(keywords),
        "n_corpus": len(texts),
    }
    with open(KEYWORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"\n키워드 저장: {KEYWORDS_PATH}")
    print(f"상위 30 키워드: {keywords[:30]}")
    return result


# ==============================================================================
# [E] 분기별 그래프 (PMI 엣지 + 인용 엣지)
# ==============================================================================

def step_E_edges(df=None, kw_data=None):
    """
    분기별로 키워드 공출현 PMI 엣지와 인용 엣지를 계산한다.

    저장 컬럼:
      quarter_label, src_node, tgt_node, edge_type (cooccur | citation),
      weight (PMI 또는 인용횟수), n_patents
    """
    print("\n" + "=" * 80)
    print("[STEP E] 분기별 그래프 엣지 구축 (PMI 공출현 + 인용)")
    print("=" * 80)

    if df is None:
        df = pd.read_parquet(CORPUS_PATH)
    if kw_data is None:
        with open(KEYWORDS_PATH) as f:
            kw_data = json.load(f)

    keywords = kw_data["keywords"]
    kw_set = set(keywords)
    kw_index = {kw: i for i, kw in enumerate(keywords)}
    N = len(keywords)

    print(f"키워드 노드 수: {N:,}")
    print(f"분기 수: {df['quarter_label'].nunique():,}")

    # 텍스트에서 키워드 매칭 (빠른 처리를 위해 정규식 미사용, 단순 lower split)
    # 바이그램/트리그램도 포함해야 하므로 n-gram 매칭 함수
    def extract_kw_in_text(text: str) -> set:
        """텍스트에 등장하는 키워드 집합 반환"""
        if not isinstance(text, str) or not text.strip():
            return set()
        text_lower = text.lower()
        found = set()
        for kw in keywords:
            if kw in text_lower:
                found.add(kw)
        return found

    edges_rows = []

    quarters = sorted(df["quarter_label"].unique())
    quarters = [q for q in quarters if q != "UNKNOWN"]

    for qlab in tqdm(quarters, desc="분기별 엣지"):
        subset = df[df["quarter_label"] == qlab]
        if len(subset) < 2:
            continue

        # 각 특허의 키워드 집합
        doc_kws = subset["final_text"].apply(extract_kw_in_text).tolist()
        n_docs = len(doc_kws)

        # ── PMI 계산 ──────────────────────────────────────────────────
        # 단어 빈도 (문서 기준)
        kw_freq = defaultdict(int)
        pair_freq = defaultdict(int)

        for kw_set_doc in doc_kws:
            for kw in kw_set_doc:
                kw_freq[kw] += 1
            # 공출현 쌍 (조합)
            kw_list = sorted(kw_set_doc)
            for i in range(len(kw_list)):
                for j in range(i + 1, len(kw_list)):
                    pair = (kw_list[i], kw_list[j])
                    pair_freq[pair] += 1

        # PMI = log2(P(x,y) / (P(x)*P(y)))
        for (kw_a, kw_b), co_count in pair_freq.items():
            if co_count < MIN_COOCCUR:
                continue
            p_ab = co_count / n_docs
            p_a = kw_freq[kw_a] / n_docs
            p_b = kw_freq[kw_b] / n_docs
            if p_a == 0 or p_b == 0:
                continue
            pmi = np.log2(p_ab / (p_a * p_b))
            if pmi > PMI_THRESHOLD:
                edges_rows.append({
                    "quarter_label": qlab,
                    "src_node": kw_a,
                    "tgt_node": kw_b,
                    "edge_type": "cooccur",
                    "weight": round(float(pmi), 4),
                    "n_patents": int(co_count),
                })

    print(f"\n총 PMI 엣지 수: {len(edges_rows):,}")

    # ── 인용 엣지 (forward citation) ─────────────────────────────────
    # 인용 컬럼이 있으면 처리 (없으면 생략)
    citation_col = None
    for col in ("Citations", "Forward Citations", "Cited By", "References"):
        if col in df.columns:
            citation_col = col
            break

    if citation_col:
        print(f"인용 컬럼 발견: {citation_col} — 인용 엣지 추가 예정")
        # 인용 엣지는 특허 수준 (특허 ID 간)이므로 노드 레벨 키워드 인용은
        # 해당 특허의 키워드 집합으로 전파 (aggregate)
        # 실제 구현은 인용 데이터 구조에 따라 커스터마이즈 필요
        print("  ※ 인용 엣지 구현은 인용 데이터 포맷 확인 후 커스터마이즈 필요")
        print("  현재는 PMI 공출현 엣지만 저장")

    edges_df = pd.DataFrame(edges_rows)
    if len(edges_df) == 0:
        print("⚠ 엣지 없음 — MIN_COOCCUR 또는 PMI_THRESHOLD 조정 필요")
        edges_df = pd.DataFrame(columns=[
            "quarter_label", "src_node", "tgt_node",
            "edge_type", "weight", "n_patents"
        ])

    safe_to_parquet(edges_df, EDGES_PATH)
    print(f"엣지 저장: {EDGES_PATH}  ({len(edges_df):,}행)")

    # 분기별 엣지 수 요약
    if len(edges_df) > 0:
        eq = edges_df.groupby("quarter_label").size().reset_index(name="edge_count")
        print(f"\n분기당 평균 엣지: {eq['edge_count'].mean():.1f}")
        print(f"분기당 최소/최대 엣지: {eq['edge_count'].min()} / {eq['edge_count'].max()}")

    return edges_df


# ==============================================================================
# [F] 노드 특징 행렬 (f1~f8)
# ==============================================================================

def step_F_node_features(df=None, kw_data=None):
    """
    분기 × 노드 × 8특징 행렬을 long format으로 저장.

    컬럼: quarter_label, node (keyword), f1~f8
    """
    print("\n" + "=" * 80)
    print("[STEP F] 노드 특징 행렬 구축 (f1~f8)")
    print("=" * 80)

    if df is None:
        df = pd.read_parquet(CORPUS_PATH)
    if kw_data is None:
        with open(KEYWORDS_PATH) as f:
            kw_data = json.load(f)

    keywords = kw_data["keywords"]
    kw_doc_freq = kw_data.get("keyword_doc_freq", {})
    N = len(keywords)
    total_docs = len(df)

    print(f"노드(키워드) 수: {N:,}")

    # IDF 사전 계산 (f2)
    idf = {
        kw: np.log((total_docs + 1) / (kw_doc_freq.get(kw, 0) + 1)) + 1
        for kw in keywords
    }

    # CPC 컬럼 탐지 (f6: CPC 서브그룹 다양성)
    cpc_col = None
    for col in ("CPC", "CPC Classifications", "IPC", "Classification"):
        if col in df.columns:
            cpc_col = col
            break

    # 인용 컬럼 탐지 (f3: forward citation)
    fcite_col = None
    for col in ("Forward Citations", "Citations Received", "Cited By Count"):
        if col in df.columns:
            fcite_col = col
            break

    bcite_col = None
    for col in ("Backward Citations", "References Count", "Citations Made"):
        if col in df.columns:
            bcite_col = col
            break

    print(f"CPC 컬럼: {cpc_col}  |  Forward citation: {fcite_col}  "
          f"|  Backward citation: {bcite_col}")

    # 텍스트 → 키워드 집합 (빠른 처리)
    def extract_kws(text):
        if not isinstance(text, str) or not text.strip():
            return set()
        tl = text.lower()
        return {kw for kw in keywords if kw in tl}

    quarters = sorted([q for q in df["quarter_label"].unique() if q != "UNKNOWN"])
    feature_rows = []

    for qlab in tqdm(quarters, desc="분기별 특징"):
        subset = df[df["quarter_label"] == qlab].copy()
        n_q = len(subset)
        if n_q == 0:
            continue

        # 출원 연도 (f5: 노드 나이 계산용)
        try:
            q_year = int(qlab[:4])
        except Exception:
            q_year = 2000

        # 키워드별 통계 초기화
        kw_patent_count = defaultdict(int)       # 등장 특허수 (f1 TF)
        kw_fcite_sum = defaultdict(float)         # forward citation 합 (f3)
        kw_bcite_sum = defaultdict(float)         # backward citation 합 (f4)
        kw_year_sum = defaultdict(int)            # 출원연도 합 (f5 평균 나이 계산)
        kw_cpc_sets = defaultdict(set)            # CPC 서브그룹 집합 (f6)
        kw_group_sets = defaultdict(set)          # 출원인 그룹 집합 (f7)
        kw_tier_sum = defaultdict(int)            # text_tier 합 (f8)

        for _, row in subset.iterrows():
            kws = extract_kws(str(row.get("final_text", "")))
            if not kws:
                continue

            fcite = float(row[fcite_col]) if fcite_col and pd.notna(row.get(fcite_col)) else 0.0
            bcite = float(row[bcite_col]) if bcite_col and pd.notna(row.get(bcite_col)) else 0.0
            year_val = int(row["year"]) if pd.notna(row.get("year")) else q_year
            tier = int(row.get("text_tier", 1))

            # CPC 서브그룹 파싱 (예: "B60W10/04;;B60W20/00" → {"B60W10", "B60W20"})
            cpc_subgroups = set()
            if cpc_col and pd.notna(row.get(cpc_col)):
                for cpc_code in str(row[cpc_col]).split(";;"):
                    sg = cpc_code.strip()[:6]   # 서브그룹 앞 6자
                    if sg:
                        cpc_subgroups.add(sg)

            # 출원인 그룹
            group = str(row.get("applicant_group_primary", "UNKNOWN"))

            for kw in kws:
                kw_patent_count[kw] += 1
                kw_fcite_sum[kw] += fcite
                kw_bcite_sum[kw] += bcite
                kw_year_sum[kw] += year_val
                kw_cpc_sets[kw].update(cpc_subgroups)
                kw_group_sets[kw].add(group)
                kw_tier_sum[kw] += tier

        # 특징값 계산
        for kw in keywords:
            cnt = kw_patent_count.get(kw, 0)
            if cnt == 0:
                # 미등장 키워드는 저장 생략 (희소 저장)
                continue

            f1 = cnt / n_q                                  # TF (정규화)
            f2 = idf.get(kw, 1.0)                          # IDF
            f3 = kw_fcite_sum[kw] / cnt                    # 평균 forward citation
            f4 = kw_bcite_sum[kw] / cnt                    # 평균 backward citation
            avg_year = kw_year_sum[kw] / cnt
            f5 = max(0.0, q_year - avg_year)               # 평균 특허 나이(분기 기준)

            # f6: CPC 서브그룹 Shannon entropy
            cpc_set = kw_cpc_sets[kw]
            if len(cpc_set) <= 1:
                f6 = 0.0
            else:
                # 균등 가정 entropy (실제 빈도 집계 생략 → 고유수 기반)
                n_cpc = len(cpc_set)
                f6 = np.log2(n_cpc)

            f7 = len(kw_group_sets[kw])                    # 출원인 그룹 다양성 (고유수)
            f8 = kw_tier_sum[kw] / cnt                     # 평균 text_tier

            feature_rows.append({
                "quarter_label": qlab,
                "node": kw,
                "f1_tf": round(f1, 6),
                "f2_idf": round(f2, 4),
                "f3_fcite": round(f3, 4),
                "f4_bcite": round(f4, 4),
                "f5_age": round(f5, 2),
                "f6_cpc_entropy": round(f6, 4),
                "f7_group_diversity": int(f7),
                "f8_text_tier": round(f8, 4),
                "patent_count": int(cnt),
            })

    feat_df = pd.DataFrame(feature_rows)
    safe_to_parquet(feat_df, NODE_FEATURES_PATH)
    print(f"\n노드 특징 행렬 저장: {NODE_FEATURES_PATH}")
    print(f"총 행 수 (분기×등장노드): {len(feat_df):,}")
    print(f"평균 분기당 활성 노드: {len(feat_df)/len(quarters):.1f}")

    # 특징 통계
    print("\n특징값 기술통계:")
    feat_cols = [c for c in feat_df.columns if c.startswith("f")]
    print(feat_df[feat_cols].describe().round(3).to_string())

    return feat_df


# ==============================================================================
# [G] 그래프 메타 저장 + 분기 범위 최종 확정
# ==============================================================================

def step_G_meta(df=None, kw_data=None, edges_df=None, feat_df=None,
                start_year=DEFAULT_START_YEAR):
    """
    파이프라인 메타데이터를 저장하고 분기 범위를 최종 확정한다.
    """
    print("\n" + "=" * 80)
    print("[STEP G] 그래프 메타데이터 저장 및 분기 범위 확정")
    print("=" * 80)

    if df is None:
        df = pd.read_parquet(CORPUS_PATH)
    if kw_data is None:
        with open(KEYWORDS_PATH) as f:
            kw_data = json.load(f)

    keywords = kw_data["keywords"]
    quarters = sorted([q for q in df["quarter_label"].unique() if q != "UNKNOWN"])

    # 분기 범위 통계
    q_2000 = [q for q in quarters if int(q[:4]) >= 2000]
    q_1995 = [q for q in quarters if int(q[:4]) >= 1995]

    # 연구계획서 기준: 100분기(2000-2024) vs 120분기(1995-2024)
    recommendation = "120분기 (1995-2024)" if len(q_1995) >= 100 else "100분기 (2000-2024)"

    meta = {
        "version": "1.0",
        "total_patents": len(df),
        "n_keywords": len(keywords),
        "quarters_all": quarters,
        "n_quarters_total": len(quarters),
        "n_quarters_2000_2024": len(q_2000),
        "n_quarters_1995_2024": len(q_1995),
        "quarter_range_recommendation": recommendation,
        "train_valid_test_split": {
            "train": f"{start_year}-2014",
            "valid": "2015-2018",
            "test": "2019-2024",
        },
        "node_features": {
            "f1": "TF (분기 내 출현빈도 / 분기 특허수)",
            "f2": "IDF (전체 코퍼스 역문서빈도)",
            "f3": "평균 Forward Citation (피인용수)",
            "f4": "평균 Backward Citation (인용수)",
            "f5": "평균 특허 나이 (출원연도 vs 분기연도 차이)",
            "f6": "CPC 서브그룹 다양성 (log2 고유 서브그룹 수)",
            "f7": "출원인 그룹 다양성 (고유 그룹 수)",
            "f8": "평균 text_tier (1=Abstract, 2=Title+CPC, 3=Claims)",
        },
        "edge_types": ["cooccur (PMI)", "citation (forward)"],
        "pmi_min_cooccur": MIN_COOCCUR,
        "pmi_threshold": PMI_THRESHOLD,
        "output_files": {
            "corpus": str(CORPUS_PATH),
            "keywords": str(KEYWORDS_PATH),
            "node_features": str(NODE_FEATURES_PATH),
            "edges": str(EDGES_PATH),
            "quarter_summary": str(QUARTER_SUMMARY_PATH),
        },
    }

    with open(GRAPH_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"메타데이터 저장: {GRAPH_META_PATH}")
    print(f"\n[분기 범위 최종 권고]")
    print(f"  전체 유효 분기: {len(quarters)}개")
    print(f"  1995년 이후: {len(q_1995)}분기")
    print(f"  2000년 이후: {len(q_2000)}분기")
    print(f"  → 권고: {recommendation}")
    print(f"\n  Train/Valid/Test 분할:")
    print(f"    Train  : {start_year}-2014")
    print(f"    Valid  : 2015-2018")
    print(f"    Test   : 2019-2024")

    return meta


# ==============================================================================
# 메인
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="STAGE 1 언어그래프 구축 파이프라인")
    parser.add_argument(
        "--step",
        choices=["A", "B", "C", "D", "E", "F", "G", "all"],
        default="all",
        help=(
            "실행 단계: A=코퍼스통합, B=분기요약, C=임베딩, "
            "D=키워드추출, E=엣지구축, F=노드특징, G=메타저장, all=전체"
        ),
    )
    parser.add_argument(
        "--embed-model",
        choices=["ollama", "sentence-transformers"],
        default="ollama",
        help="임베딩 모델 선택 (기본: ollama nomic-embed-text)",
    )
    parser.add_argument(
        "--n-keywords", type=int, default=DEFAULT_N_KEYWORDS,
        help=f"키워드(노드) 수 (기본: {DEFAULT_N_KEYWORDS})",
    )
    parser.add_argument(
        "--start-year", type=int, default=DEFAULT_START_YEAR,
        help=f"분기 시작 연도 (기본: {DEFAULT_START_YEAR})",
    )
    parser.add_argument(
        "--end-year", type=int, default=DEFAULT_END_YEAR,
        help=f"분기 종료 연도 (기본: {DEFAULT_END_YEAR})",
    )
    args = parser.parse_args()

    df = kw_data = edges_df = feat_df = None
    steps = (
        ["A", "B", "C", "D", "E", "F", "G"]
        if args.step == "all"
        else [args.step]
    )

    for step in steps:
        if step == "A":
            df = step_A_corpus(start_year=args.start_year, end_year=args.end_year)

        elif step == "B":
            if df is None and CORPUS_PATH.exists():
                df = pd.read_parquet(CORPUS_PATH)
            step_B_quarter_summary(df)

        elif step == "C":
            if df is None and CORPUS_PATH.exists():
                df = pd.read_parquet(CORPUS_PATH)
            df, _ = step_C_embedding(df, embed_model=args.embed_model)

        elif step == "D":
            if df is None and CORPUS_PATH.exists():
                df = pd.read_parquet(CORPUS_PATH)
            kw_data = step_D_keywords(df, n_keywords=args.n_keywords)

        elif step == "E":
            if df is None and CORPUS_PATH.exists():
                df = pd.read_parquet(CORPUS_PATH)
            if kw_data is None and KEYWORDS_PATH.exists():
                with open(KEYWORDS_PATH) as f:
                    kw_data = json.load(f)
            edges_df = step_E_edges(df, kw_data)

        elif step == "F":
            if df is None and CORPUS_PATH.exists():
                df = pd.read_parquet(CORPUS_PATH)
            if kw_data is None and KEYWORDS_PATH.exists():
                with open(KEYWORDS_PATH) as f:
                    kw_data = json.load(f)
            feat_df = step_F_node_features(df, kw_data)

        elif step == "G":
            if df is None and CORPUS_PATH.exists():
                df = pd.read_parquet(CORPUS_PATH)
            if kw_data is None and KEYWORDS_PATH.exists():
                with open(KEYWORDS_PATH) as f:
                    kw_data = json.load(f)
            step_G_meta(df, kw_data, edges_df, feat_df,
                        start_year=args.start_year)

    print("\n" + "=" * 80)
    print("STAGE 1 파이프라인 완료")
    print("=" * 80)
    print("\n다음 단계: stage2_mamba_lstm.py (STAGE 2 Mamba-LSTM 모델 훈련)")
    print(f"  입력: {NODE_FEATURES_PATH}")
    print(f"        {EDGES_PATH}")
    print(f"  키워드: {KEYWORDS_PATH}")


if __name__ == "__main__":
    main()
