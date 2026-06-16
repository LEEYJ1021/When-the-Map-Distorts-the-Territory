"""
================================================================================
STAGE 4: sLLM Multi-Agent Simulation (Ollama 로컬 구동)  v1.0
================================================================================
연구계획 v5.0 STAGE 4 사양 반영:
  - 모델: llama3.1:8b (Ollama, 4 에이전트)
  - 에이전트 구성: STAGE 0-B 그룹통합 Top10 기반 (확정 후 설계) — 본 스크립트는
    기본값으로 4개 대표 그룹(JP/Toyota Group, KR/Hyundai Motor Group, US/Ford,
    DE/Bosch)을 사용하되, --agents-config 로 STAGE0-B 산출물을 주입 가능
  - RAG 소스: STAGE 0-C final_text (그룹별 클러스터), TF-IDF 검색
  - 3-Round 구조:
      Round 1 (독립포지션)   : 각 에이전트가 자기 그룹 관점에서 초기 입장 표명
      Round 2 (상호반박)     : 다른 에이전트 입장을 보고 반박/수정
                                (조건부 시나리오: KR>JP 역전 확인/미확인 분기)
      Round 3 (수렴-분기)    : 최종 확률추정 + 수렴/분기 판정
  - N회 반복 시뮬레이션 (변동요소: target_quarter, RAG top-K, temperature,
    scenario_branch) → H3a/H3b/H3c 검정용 로그 산출

조건부 시나리오 분기 (H2a/H2a', H3c):
  --scenario-branch reversal_confirmed | reversal_not_confirmed | auto
  auto인 경우 STAGE3 H2a 검정 결과 파일(stage3_work/s3_h2a_verdict.json)이
  있으면 그 결과를 사용하고, 없으면 시뮬레이션마다 50:50으로 무작위 배정
  (→ 두 시나리오 모두에 대한 데이터를 동시에 축적해 사후 비교 가능).

출력:
  stage4_work/s4_simulations.jsonl     (시뮬레이션별 전체 라운드 로그, append)
  stage4_work/s4_simulation_summary.csv (시뮬레이션별 핵심 지표 요약)

실행:
  # Ollama 서버가 로컬에서 실행 중이어야 함 (ollama serve)
  # 필요 모델: ollama pull llama3.1:8b
  python stage4_sllm_multiagent.py --n-sims 10 --dry-run     # 코드 점검(LLM 호출 없이 mock)
  python stage4_sllm_multiagent.py --n-sims 1000             # 실제 실행
  python stage4_sllm_multiagent.py --n-sims 50 --corpus stage0_work/s0_corpus_grouped.parquet
================================================================================
"""

import json
import argparse
import random
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ==============================================================================
# 경로
# ==============================================================================
BASE_DIR    = Path.cwd()
STAGE1_DIR  = BASE_DIR / "stage1_work"
STAGE3_DIR  = BASE_DIR / "stage3_work"
STAGE4_DIR  = BASE_DIR / "stage4_work"
STAGE4_DIR.mkdir(exist_ok=True)

GRAPH_META_PATH = STAGE1_DIR / "s1_graph_meta.json"
H2A_VERDICT_PATH = STAGE3_DIR / "s3_h2a_verdict.json"  # STAGE0-B/STAGE3 검증 후 생성 (옵션)

OUT_JSONL = STAGE4_DIR / "s4_simulations.jsonl"
OUT_SUMMARY = STAGE4_DIR / "s4_simulation_summary.csv"

DEFAULT_MODEL = "llama3.1:8b"

# ==============================================================================
# 기본 에이전트 구성 (STAGE 0-B 산출물 없을 때 fallback)
# ------------------------------------------------------------------------------
# STAGE 0-B 산출물(applicant_group, applicant_nationality, jurisdiction 등)이
# 준비되면 --agents-config <json> 으로 교체 주입 가능. 형식:
#   [{"name": "...", "group_key": "...", "nationality": "..", "persona": "..."}]
# ==============================================================================
DEFAULT_AGENTS = [
    {
        "name": "Toyota Group (JP)",
        "group_key": "Toyota Group",
        "nationality": "JP",
        "persona": (
            "당신은 Toyota Group(토요타, 덴소, 아이신 계열 포함)의 전동화 파워트레인 "
            "기술전략 담당 임원입니다. 일본 기업의 신중하고 점진적인 기술 로드맵 운영 "
            "방식을 반영하여 발언하십시오."
        ),
    },
    {
        "name": "Hyundai Motor Group (KR)",
        "group_key": "Hyundai Motor Group",
        "nationality": "KR",
        "persona": (
            "당신은 Hyundai Motor Group(현대자동차, 기아, 현대모비스 포함)의 전동화 "
            "파워트레인 기술전략 담당 임원입니다. 빠른 의사결정과 적극적인 시장 "
            "추격 전략을 반영하여 발언하십시오."
        ),
    },
    {
        "name": "Ford (US)",
        "group_key": "Ford",
        "nationality": "US",
        "persona": (
            "당신은 Ford Motor Company의 전동화 파워트레인 기술전략 담당 임원입니다. "
            "북미 시장 규제환경과 대형/픽업 차량 중심의 전동화 전략을 반영하여 "
            "발언하십시오."
        ),
    },
    {
        "name": "Bosch (DE)",
        "group_key": "Bosch",
        "nationality": "DE",
        "persona": (
            "당신은 Bosch의 전동화 파워트레인 부품/시스템 기술전략 담당 임원입니다. "
            "OEM이 아닌 Tier-1 공급사 관점에서, 여러 OEM에 공급하는 표준화·모듈화 "
            "전략을 반영하여 발언하십시오."
        ),
    },
]

# 시뮬레이션 주제: B60W60(전동화 파워트레인 제어) 핵심노드 부상 관련 RQ2/H1c
SIM_TOPIC = (
    "B60W60(하이브리드/전기차 파워트레인 제어, 회생제동·에너지관리·배터리관리 등) "
    "관련 기술의 특허 핵심노드(허브) 부상 동향"
)

# H3a/b/c 검정용 최종질문 (Round 3)
FINAL_QUESTION = (
    "2017-2020년 B60W60 활용 가속구간 이후, 이 기술영역의 특허 핵심노드 중심성이 "
    "향후에도 계속 증가할 것인가?"
)


# ==============================================================================
# RAG: 그룹별 final_text TF-IDF 검색
# ==============================================================================

DEMO_CORPUS_TEXTS = {
    "Toyota Group": [
        "Hybrid vehicle energy management system using state of charge estimation and "
        "regenerative braking torque distribution for series-parallel hybrid powertrain.",
        "Battery management system for plug-in hybrid electric vehicle with thermal "
        "management of traction battery pack.",
        "Power split device control method for hybrid transmission using planetary gear "
        "set and electric motor torque blending.",
    ],
    "Hyundai Motor Group": [
        "Electric vehicle powertrain control apparatus for dual-motor all-wheel-drive "
        "with torque distribution based on regenerative braking demand.",
        "Battery management system for electric vehicle including state of charge "
        "balancing across multiple battery modules.",
        "Fuel cell hybrid vehicle energy management strategy combining battery and "
        "fuel cell stack power output.",
    ],
    "Ford": [
        "Mild hybrid powertrain control system for pickup truck with 48V battery "
        "and belt-driven starter generator energy recovery.",
        "Plug-in hybrid electric vehicle charging control method with state of charge "
        "based range extender activation.",
        "Regenerative braking control for electric truck with torque distribution "
        "between front and rear axle motors.",
    ],
    "Bosch": [
        "Modular battery management system architecture for multiple OEM hybrid and "
        "electric vehicle platforms with standardized state of charge interface.",
        "Inverter and DC-DC converter integration for electric vehicle powertrain "
        "with thermal management of power electronics.",
        "Energy management control unit for hybrid powertrain supplied as Tier-1 "
        "component across multiple vehicle platforms.",
    ],
}


class RagIndex:
    """그룹별 final_text 코퍼스에 대한 TF-IDF 검색 인덱스."""

    def __init__(self, group_texts: dict):
        self.group_texts = group_texts
        self.vectorizers = {}
        self.matrices = {}
        if SKLEARN_AVAILABLE:
            for grp, texts in group_texts.items():
                if len(texts) == 0:
                    continue
                vec = TfidfVectorizer(max_features=2000, stop_words="english")
                mat = vec.fit_transform(texts)
                self.vectorizers[grp] = vec
                self.matrices[grp] = mat

    def retrieve(self, group_key: str, query: str, top_k: int = 3):
        texts = self.group_texts.get(group_key, [])
        if len(texts) == 0:
            return []
        if not SKLEARN_AVAILABLE or group_key not in self.vectorizers:
            # sklearn 없으면 무작위 샘플로 대체
            k = min(top_k, len(texts))
            return random.sample(texts, k)
        vec = self.vectorizers[group_key]
        mat = self.matrices[group_key]
        q_vec = vec.transform([query])
        sims = cosine_similarity(q_vec, mat)[0]
        order = np.argsort(-sims)[:top_k]
        return [texts[i] for i in order]


def build_rag_index(corpus_path: Path, agents: list) -> RagIndex:
    """
    corpus_path: STAGE0-C 산출물 (final_text, applicant_group 컬럼 포함) 가정.
    파일이 없으면 DEMO_CORPUS_TEXTS로 fallback (코드 점검/파이프라인 테스트용).
    """
    group_keys = [a["group_key"] for a in agents]

    if corpus_path is not None and corpus_path.exists():
        df = pd.read_parquet(corpus_path)
        if "applicant_group" not in df.columns or "final_text" not in df.columns:
            print(f"  ⚠️  {corpus_path} 에 'applicant_group'/'final_text' 컬럼 없음 → DEMO 코퍼스 사용")
            group_texts = {g: DEMO_CORPUS_TEXTS.get(g, []) for g in group_keys}
        else:
            group_texts = {}
            for g in group_keys:
                texts = df.loc[df["applicant_group"] == g, "final_text"].dropna().astype(str).tolist()
                if len(texts) == 0:
                    print(f"  ⚠️  그룹 '{g}'에 해당하는 final_text 없음 → DEMO 코퍼스로 대체")
                    texts = DEMO_CORPUS_TEXTS.get(g, [])
                group_texts[g] = texts
            print(f"  RAG 코퍼스 로드: {corpus_path}  (그룹별 문서수: "
                  f"{ {g: len(t) for g, t in group_texts.items()} })")
    else:
        print(f"  ⚠️  RAG 코퍼스 파일 없음 ({corpus_path}) → DEMO 코퍼스 사용 (파이프라인 점검용)")
        group_texts = {g: DEMO_CORPUS_TEXTS.get(g, []) for g in group_keys}

    return RagIndex(group_texts)


# ==============================================================================
# Ollama 호출 래퍼
# ==============================================================================

def call_llm(model: str, system: str, user: str, temperature: float,
              dry_run: bool, agent_name: str, round_name: str) -> str:
    """
    Ollama chat 호출. dry_run=True면 LLM 호출 없이 mock 응답을 생성한다
    (파이프라인 구조 점검용 — 실제 모델 출력이 아님).
    """
    if dry_run or not OLLAMA_AVAILABLE:
        return _mock_response(agent_name, round_name, temperature)

    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": temperature},
        )
        return resp["message"]["content"]
    except Exception as e:
        print(f"  ⚠️  Ollama 호출 실패 ({agent_name}, {round_name}): {e} → mock 응답으로 대체")
        return _mock_response(agent_name, round_name, temperature)


def _mock_response(agent_name: str, round_name: str, temperature: float) -> str:
    """dry-run / 호출 실패 시 사용하는 JSON 형식 mock 응답 (구조 검증용)."""
    rng = random.Random(hash((agent_name, round_name, round(temperature, 2))) & 0xFFFFFFFF)
    if round_name == "round3":
        prob = round(rng.uniform(0.3, 0.9), 2)
        return json.dumps({
            "probability": prob,
            "stance": "increase" if prob >= 0.5 else "decrease",
            "rationale": f"[MOCK:{agent_name}] 더미 응답 — 실제 Ollama 미연동 상태.",
        }, ensure_ascii=False)
    else:
        stance = rng.choice(["expand", "maintain", "reduce"])
        return json.dumps({
            "stance": stance,
            "confidence": rng.randint(1, 10),
            "rationale": f"[MOCK:{agent_name}/{round_name}] 더미 응답 — 실제 Ollama 미연동 상태.",
        }, ensure_ascii=False)


def parse_json_response(text: str, fallback: dict) -> dict:
    """LLM 응답에서 JSON 블록을 최대한 안전하게 파싱."""
    text = text.strip()
    # 코드블록 마커 제거
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    # 첫 '{' ~ 마지막 '}' 구간만 시도
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return dict(fallback)


# ==============================================================================
# 프롬프트 빌더
# ==============================================================================

def build_round1_prompt(agent: dict, rag_chunks: list, target_quarter: str) -> tuple:
    system = (
        f"{agent['persona']}\n\n"
        "당신의 답변은 반드시 아래 JSON 형식으로만 작성하십시오 (다른 텍스트 금지):\n"
        '{"stance": "expand|maintain|reduce", "confidence": <1-10 정수>, '
        '"rationale": "<2-3문장 한국어 설명>"}'
    )
    context = "\n".join(f"- {c}" for c in rag_chunks) if rag_chunks else "(참고 특허 정보 없음)"
    user = (
        f"주제: {SIM_TOPIC}\n"
        f"대상 시점: {target_quarter}\n\n"
        f"[당신 그룹의 최근 관련 특허 발췌]\n{context}\n\n"
        "위 정보를 바탕으로, 귀사가 이 기술영역에 대한 투자/연구개발을 "
        "확대(expand)/유지(maintain)/축소(reduce)할 전략적 입장을 표명하고 "
        "그 확신도(confidence, 1-10)와 근거를 제시하십시오."
    )
    return system, user


def build_round2_prompt(agent: dict, round1_all: dict, scenario_branch: str) -> tuple:
    others = "\n".join(
        f"- {name}: stance={r.get('stance','?')}, confidence={r.get('confidence','?')}, "
        f"근거: {r.get('rationale','')}"
        for name, r in round1_all.items() if name != agent["name"]
    )

    if scenario_branch == "reversal_confirmed":
        scenario_note = (
            "[시나리오 정보] 최근 STAGE3 분석에 따르면, 한국(KR) 출원사들의 서브그룹 "
            "다양성(subgroup diversity) 기반 기술영향력 지표가 일본(JP) 출원사를 "
            "추월한 것으로 확인되었습니다. 이 정보를 반영하여 입장을 재검토하십시오."
        )
    else:
        scenario_note = (
            "[시나리오 정보] STAGE3 분석 결과, 한국(KR)과 일본(JP) 출원사 간 "
            "서브그룹 다양성 지표의 역전 여부는 아직 명확하지 않습니다 "
            "(추세상 KR의 증가속도가 더 가파르나, JP를 추월했는지는 불확실). "
            "이 불확실성을 반영하여 입장을 재검토하십시오."
        )

    system = (
        f"{agent['persona']}\n\n"
        "당신의 답변은 반드시 아래 JSON 형식으로만 작성하십시오 (다른 텍스트 금지):\n"
        '{"stance": "expand|maintain|reduce", "confidence": <1-10 정수>, '
        '"rebuttal_target": "<반박/동의 대상 그룹명 또는 none>", '
        '"rationale": "<2-3문장 한국어 설명>"}'
    )
    user = (
        f"주제: {SIM_TOPIC}\n\n"
        f"[1라운드에서의 다른 그룹들의 입장]\n{others}\n\n"
        f"{scenario_note}\n\n"
        "다른 그룹의 입장 중 동의하거나 반박할 부분을 지적하고, 필요하다면 "
        "당신의 1라운드 입장을 수정하여 다시 stance/confidence/근거를 제시하십시오."
    )
    return system, user


def build_round3_prompt(agent: dict, round1_all: dict, round2_all: dict) -> tuple:
    history = []
    for name in round1_all:
        r1 = round1_all.get(name, {})
        r2 = round2_all.get(name, {})
        history.append(
            f"- {name}: R1 stance={r1.get('stance','?')}(conf={r1.get('confidence','?')}) "
            f"→ R2 stance={r2.get('stance','?')}(conf={r2.get('confidence','?')}), "
            f"R2 근거: {r2.get('rationale','')}"
        )
    history_str = "\n".join(history)

    system = (
        f"{agent['persona']}\n\n"
        "당신의 답변은 반드시 아래 JSON 형식으로만 작성하십시오 (다른 텍스트 금지):\n"
        '{"probability": <0과 1 사이 실수>, "stance": "increase|decrease", '
        '"rationale": "<2-3문장 한국어 설명>"}'
    )
    user = (
        f"최종 질문: {FINAL_QUESTION}\n\n"
        f"[지금까지의 라운드별 입장 변화]\n{history_str}\n\n"
        "위 논의를 종합하여, 이 질문에 대한 귀사의 최종 확률추정(probability, "
        "0~1)과 방향(increase/decrease)을 제시하십시오."
    )
    return system, user


# ==============================================================================
# 시뮬레이션 1회 실행
# ==============================================================================

def run_single_simulation(sim_id: int, agents: list, rag_index: RagIndex,
                            model: str, dry_run: bool, rng: random.Random,
                            quarters: list, scenario_branch_mode: str) -> dict:

    target_quarter = rng.choice(quarters)
    rag_top_k = rng.randint(3, 7)
    temperature = round(rng.uniform(0.3, 1.2), 2)

    if scenario_branch_mode == "auto" and H2A_VERDICT_PATH.exists():
        try:
            with open(H2A_VERDICT_PATH, "r", encoding="utf-8") as f:
                h2a = json.load(f)
            scenario_branch = "reversal_confirmed" if h2a.get("reversal_confirmed") else "reversal_not_confirmed"
        except Exception:
            scenario_branch = rng.choice(["reversal_confirmed", "reversal_not_confirmed"])
    elif scenario_branch_mode in ("reversal_confirmed", "reversal_not_confirmed"):
        scenario_branch = scenario_branch_mode
    else:  # "auto" without verdict file, or "random"
        scenario_branch = rng.choice(["reversal_confirmed", "reversal_not_confirmed"])

    record = {
        "sim_id": sim_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_quarter": target_quarter,
        "rag_top_k": rag_top_k,
        "temperature": temperature,
        "scenario_branch": scenario_branch,
        "model": model,
        "rounds": {"round1": {}, "round2": {}, "round3": {}},
    }

    # ---- Round 1: 독립포지션 ----
    round1_all = {}
    for agent in agents:
        rag_chunks = rag_index.retrieve(agent["group_key"], SIM_TOPIC, top_k=rag_top_k)
        system, user = build_round1_prompt(agent, rag_chunks, target_quarter)
        raw = call_llm(model, system, user, temperature, dry_run, agent["name"], "round1")
        parsed = parse_json_response(raw, fallback={"stance": "maintain", "confidence": 5, "rationale": raw[:200]})
        round1_all[agent["name"]] = parsed
    record["rounds"]["round1"] = round1_all

    # ---- Round 2: 상호반박 ----
    round2_all = {}
    for agent in agents:
        system, user = build_round2_prompt(agent, round1_all, scenario_branch)
        raw = call_llm(model, system, user, temperature, dry_run, agent["name"], "round2")
        parsed = parse_json_response(
            raw, fallback={"stance": round1_all[agent["name"]].get("stance", "maintain"),
                           "confidence": round1_all[agent["name"]].get("confidence", 5),
                           "rebuttal_target": "none", "rationale": raw[:200]}
        )
        round2_all[agent["name"]] = parsed
    record["rounds"]["round2"] = round2_all

    # ---- Round 3: 수렴-분기 ----
    round3_all = {}
    for agent in agents:
        system, user = build_round3_prompt(agent, round1_all, round2_all)
        raw = call_llm(model, system, user, temperature, dry_run, agent["name"], "round3")
        parsed = parse_json_response(raw, fallback={"probability": 0.5, "stance": "increase", "rationale": raw[:200]})
        # probability 안전 클리핑
        try:
            p = float(parsed.get("probability", 0.5))
        except (TypeError, ValueError):
            p = 0.5
        parsed["probability"] = min(max(p, 0.0), 1.0)
        round3_all[agent["name"]] = parsed
    record["rounds"]["round3"] = round3_all

    # ---- 메트릭 (H3a/b/c용) ----
    probs = np.array([round3_all[a["name"]]["probability"] for a in agents], dtype=np.float64)
    stances = [round3_all[a["name"]].get("stance", "increase") for a in agents]

    mean_prob = float(probs.mean())
    std_prob  = float(probs.std())

    # 이진화(increase/decrease) 엔트로피 (H3b)
    p_increase = sum(1 for s in stances if s == "increase") / len(stances)
    eps = 1e-9
    entropy = -sum(
        p * np.log2(p + eps) for p in [p_increase, 1 - p_increase] if p > 0
    )
    convergence = bool(p_increase in (0.0, 1.0))  # 전원 일치 여부

    record["metrics"] = {
        "mean_probability": mean_prob,
        "std_probability": std_prob,
        "p_increase": p_increase,
        "stance_entropy_bits": float(entropy),
        "all_agents_converged": convergence,
    }

    return record


# ==============================================================================
# Main
# ==============================================================================

def load_quarters():
    if GRAPH_META_PATH.exists():
        with open(GRAPH_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        all_q = meta["quarters_all"]
        # test 구간(2019Q1~)을 우선 사용, 없으면 전체에서 후반부 사용
        test_q = [q for q in all_q if int(q[:4]) >= 2019]
        return test_q if test_q else all_q
    # fallback: 2019Q1~2024Q4
    return [f"{y}Q{q}" for y in range(2019, 2025) for q in (1, 2, 3, 4)]


def main():
    parser = argparse.ArgumentParser(description="STAGE 4 sLLM Multi-Agent Simulation v1.0")
    parser.add_argument("--n-sims", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--corpus", default=None,
                        help="STAGE0-C 산출물 parquet 경로 (final_text, applicant_group 포함)")
    parser.add_argument("--agents-config", default=None,
                        help="STAGE0-B 기반 에이전트 설정 JSON 경로 (없으면 DEFAULT_AGENTS 사용)")
    parser.add_argument("--scenario-branch", default="auto",
                        choices=["auto", "reversal_confirmed", "reversal_not_confirmed", "random"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Ollama 호출 없이 mock 응답으로 파이프라인 구조만 검증")
    parser.add_argument("--seed", type=int, default=42)
    args, _unknown = parser.parse_known_args()

    print("=" * 70)
    print("STAGE 4: sLLM Multi-Agent Simulation (Ollama)")
    print(f"  n_sims={args.n_sims}  model={args.model}  dry_run={args.dry_run}")
    print(f"  ollama 패키지 사용가능: {OLLAMA_AVAILABLE}  sklearn 사용가능: {SKLEARN_AVAILABLE}")
    print("=" * 70)

    if not args.dry_run and not OLLAMA_AVAILABLE:
        print("\n⚠️  ollama 패키지가 설치되어 있지 않습니다. "
              "`pip install ollama --break-system-packages` 후 `ollama serve` 를 실행하세요.")
        print("   지금은 --dry-run 모드로 자동 전환하여 파이프라인 구조만 점검합니다.\n")
        args.dry_run = True

    # ---- 에이전트 구성 ----
    if args.agents_config and Path(args.agents_config).exists():
        with open(args.agents_config, "r", encoding="utf-8") as f:
            agents = json.load(f)
        print(f"\n에이전트 구성 로드: {args.agents_config} ({len(agents)}개)")
    else:
        agents = DEFAULT_AGENTS
        print(f"\n기본 에이전트 구성 사용 ({len(agents)}개): "
              f"{[a['name'] for a in agents]}")
        print("  ※ STAGE0-B 그룹통합 Top10 산출 후에는 --agents-config 로 교체 권장")

    # ---- RAG 인덱스 ----
    print("\nRAG 인덱스 구축 중 ...")
    corpus_path = Path(args.corpus) if args.corpus else (BASE_DIR / "stage0_work" / "s0_corpus_grouped.parquet")
    rag_index = build_rag_index(corpus_path, agents)

    # ---- 시계열 분기 ----
    quarters = load_quarters()
    print(f"\n시뮬레이션 대상 분기 풀: {quarters[0]} .. {quarters[-1]}  (n={len(quarters)})")

    # ---- 시뮬레이션 루프 ----
    rng = random.Random(args.seed)
    summary_rows = []

    t0 = time.time()
    with open(OUT_JSONL, "a", encoding="utf-8") as f_jsonl:
        for sim_id in range(args.n_sims):
            record = run_single_simulation(
                sim_id=sim_id, agents=agents, rag_index=rag_index,
                model=args.model, dry_run=args.dry_run, rng=rng,
                quarters=quarters, scenario_branch_mode=args.scenario_branch,
            )
            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_jsonl.flush()

            m = record["metrics"]
            summary_rows.append({
                "sim_id": sim_id,
                "target_quarter": record["target_quarter"],
                "rag_top_k": record["rag_top_k"],
                "temperature": record["temperature"],
                "scenario_branch": record["scenario_branch"],
                "mean_probability": m["mean_probability"],
                "std_probability": m["std_probability"],
                "p_increase": m["p_increase"],
                "stance_entropy_bits": m["stance_entropy_bits"],
                "all_agents_converged": m["all_agents_converged"],
            })

            if (sim_id + 1) % max(1, args.n_sims // 10) == 0 or sim_id == 0:
                elapsed = time.time() - t0
                print(f"  [{sim_id + 1}/{args.n_sims}] "
                      f"scenario={record['scenario_branch']:<24} "
                      f"mean_p={m['mean_probability']:.2f}  "
                      f"entropy={m['stance_entropy_bits']:.3f}  "
                      f"converged={m['all_agents_converged']}  "
                      f"(elapsed {elapsed:.1f}s)")

    summary_df = pd.DataFrame(summary_rows)
    # append 모드: 기존 summary가 있으면 합쳐서 다시 저장
    if OUT_SUMMARY.exists():
        prev = pd.read_csv(OUT_SUMMARY)
        summary_df = pd.concat([prev, summary_df], ignore_index=True)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    # ---- H3a/b/c용 간단 집계 ----
    print("\n" + "=" * 70)
    print("시뮬레이션 요약 (H3a/b/c 사전 점검)")
    print("=" * 70)
    by_scenario = summary_df.groupby("scenario_branch").agg(
        n=("sim_id", "count"),
        mean_p=("mean_probability", "mean"),
        mean_entropy=("stance_entropy_bits", "mean"),
        convergence_rate=("all_agents_converged", "mean"),
    )
    print(by_scenario.to_string())

    print(f"\n저장 → {OUT_JSONL}  (전체 라운드 로그, append)")
    print(f"저장 → {OUT_SUMMARY}  (시뮬레이션별 요약, 누적)")

    if args.dry_run:
        print("\n⚠️  이번 실행은 --dry-run(mock 응답) 결과입니다. "
              "실제 LLM 추론 결과가 아니므로 H3a/b/c 본문 보고에 사용하지 마십시오.")

    print("\n" + "=" * 70)
    print("STAGE 4 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
