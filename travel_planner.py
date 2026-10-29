#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
국내 여행 추천 프로그램 (LLM API + 지도/장소 검색 API 조합)

흐름
  [1/3] LLM API (POST)  : 여행 날짜 -> 추천 지역/날씨/행사/이유 (JSON)
  [2/3] 장소 API (GET)  : recommended_city -> 맛집 N곳 (JSON)
  [3/3] LLM API (POST)  : 1차 JSON + 맛집 목록 -> 최종 여행 리포트 (Markdown)

사용 예
  python travel_planner.py --date "2026-03-15"
  python travel_planner.py -date "2026-03-15" --llm gemini --place naver
  python travel_planner.py --date "2026-03-15" --cities 3      # 보너스: 복수 지역
  python travel_planner.py --date "2026-03-15" --refresh       # 캐시 무시하고 새로 호출

API 키는 코드에 쓰지 않습니다. .env 파일 또는 환경변수에서 읽습니다.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

try:
    import requests
except ImportError:  # pragma: no cover - 설치 안내용
    print("[오류] requests 패키지가 없습니다. 먼저 `pip install -r requirements.txt` 를 실행하세요.")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv 가 없어도 환경변수만으로 동작 가능
    load_dotenv = None


VERSION = "1.0.0"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = BASE_DIR / "results"

# 모델명은 서비스 사정으로 바뀔 수 있으므로 .env 의 OPENAI_MODEL / GEMINI_MODEL 로 덮어쓸 수 있다.
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "gemini": "gemini-2.5-flash"}

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
NAVER_LOCAL_URL = "https://openapi.naver.com/v1/search/local.json"

LLM_TIMEOUT = 60      # 초. LLM 은 응답 생성에 시간이 걸린다.
PLACE_TIMEOUT = 10    # 초. 장소 검색은 빠르게 실패하는 편이 낫다.

# 제공자별로 필요한 환경변수 이름
REQUIRED_KEYS = {
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY"],
    "kakao": ["KAKAO_REST_API_KEY"],
    "naver": ["NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET"],
}
SECRET_ENV_NAMES = [name for names in REQUIRED_KEYS.values() for name in names]

# 리포트 섹션 (이 순서/제목으로 고정 출력한다)
SECTION_ORDER = [
    ("region", "추천 지역"),
    ("reason", "추천 이유"),
    ("weather", "날씨 요약"),
    ("events", "행사/축제"),
    ("restaurants", "맛집 추천"),
    ("itinerary", "1일 일정 제안"),
    ("errors", "오류 요약(errors)"),
]


# =============================================================================
# 0. 공통 유틸: 오류 타입, 로그, 키 마스킹
# =============================================================================
class APIError(Exception):
    """외부 API 호출 중 발생한 오류를 '종류(error_type)'로 분류해서 들고 다닌다.

    error_type: AUTH_ERROR / QUOTA_ERROR / NETWORK_ERROR / SERVER_ERROR /
                BAD_REQUEST / NOT_FOUND / HTTP_ERROR / PARSE_ERROR / BLOCKED
    """

    def __init__(self, error_type: str, message: str, status: int | None = None):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.status = status


class SchemaError(ValueError):
    """LLM 이 만든 JSON 이 요구 스키마(필수 키/타입)를 만족하지 않을 때."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def log(msg: str = "") -> None:
    print(msg, flush=True)


def redact(text: Any) -> str:
    """문자열 안에 API 키가 섞여 있으면 *** 로 가린다 (로그/결과 파일 유출 방지)."""
    s = str(text)
    for name in SECRET_ENV_NAMES:
        value = os.environ.get(name, "")
        if value and len(value) >= 6:
            s = s.replace(value, "***")
    # 혹시 모를 키 패턴 (OpenAI sk-..., Google AIza...)
    s = re.sub(r"sk-[A-Za-z0-9_\-\*]{8,}", "sk-***", s)
    s = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "AIza***", s)
    return s


def add_error(errors: list[dict], step: str, error_type: str, message: str) -> dict:
    """내부 오류 목록에 한 건 추가하고 그 항목을 돌려준다."""
    item = {
        "step": step,
        "type": error_type,
        "message": redact(message)[:300],
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    errors.append(item)
    return item


# =============================================================================
# 1. HTTP 호출 공통부 (REST: 요청 -> 상태코드 확인 -> JSON 파싱)
# =============================================================================
def classify_status(status: int, detail: str = "") -> str:
    if status in (401, 403):
        return "AUTH_ERROR"
    if status == 429:
        return "QUOTA_ERROR"
    if status == 400 and re.search(r"api[_ ]?key", detail, re.I):
        # Gemini 는 잘못된 키를 400 (API_KEY_INVALID) 으로 돌려준다.
        return "AUTH_ERROR"
    if status == 400:
        return "BAD_REQUEST"
    if status == 404:
        return "NOT_FOUND"
    if status >= 500:
        return "SERVER_ERROR"
    return "HTTP_ERROR"


def extract_error_detail(resp: requests.Response) -> str:
    """제공자마다 다른 에러 응답 본문에서 사람이 읽을 메시지만 뽑는다."""
    try:
        body = resp.json()
    except ValueError:
        return (resp.text or "")[:200]
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):          # OpenAI / Gemini
            return str(err.get("message") or err.get("status") or err)[:200]
        if isinstance(err, str):
            return err[:200]
        for k in ("message", "errorMessage", "msg"):  # Kakao / Naver
            if body.get(k):
                return str(body[k])[:200]
    return str(body)[:200]


def http_request(method: str, url: str, *, headers: dict | None = None, params: dict | None = None,
                 json_body: dict | None = None, timeout: int = 30) -> Any:
    """requests 호출을 감싸서 '네트워크/HTTP/파싱' 오류를 APIError 하나로 통일한다."""
    try:
        resp = requests.request(method, url, headers=headers, params=params,
                                json=json_body, timeout=timeout)
    except requests.exceptions.Timeout:
        raise APIError("NETWORK_ERROR", f"요청 시간 초과({timeout}초)")
    except requests.exceptions.ConnectionError:
        raise APIError("NETWORK_ERROR", "네트워크 연결 실패(인터넷/방화벽/프록시 확인)")
    except requests.exceptions.RequestException as e:
        raise APIError("NETWORK_ERROR", f"요청 실패: {type(e).__name__}")

    if resp.status_code >= 400:
        detail = redact(extract_error_detail(resp))
        raise APIError(classify_status(resp.status_code, detail),
                       f"HTTP {resp.status_code}: {detail}", status=resp.status_code)
    try:
        return resp.json()
    except ValueError:
        raise APIError("PARSE_ERROR", f"응답 본문이 JSON 이 아닙니다(HTTP {resp.status_code})")


def with_transient_retry(func, *, retries: int = 1, delay: float = 2.0):
    """일시적 오류(네트워크/5xx)일 때만 최대 retries 회 재시도. 인증/쿼터 오류는 재시도하지 않는다."""
    for attempt in range(retries + 1):
        try:
            return func()
        except APIError as e:
            if e.error_type in ("NETWORK_ERROR", "SERVER_ERROR") and attempt < retries:
                log(f"    - 일시적 오류({e.error_type}), {delay:.0f}초 후 1회 재시도합니다...")
                time.sleep(delay)
                continue
            raise


# =============================================================================
# 2. LLM 클라이언트 (OpenAI / Gemini) - 둘 다 HTTP POST
# =============================================================================
class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str, timeout: int = LLM_TIMEOUT):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, system: str, user: str, json_mode: bool) -> str:
        call = self._openai if self.provider == "openai" else self._gemini
        return with_transient_retry(lambda: call(system, user, json_mode))

    # --- OpenAI Chat Completions -------------------------------------------
    def _openai(self, system: str, user: str, json_mode: bool) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if json_mode:
            # JSON 모드: 모델이 '파싱 가능한 JSON 객체'만 출력하도록 강제한다.
            body["response_format"] = {"type": "json_object"}
        data = http_request("POST", OPENAI_URL, headers=headers, json_body=body, timeout=self.timeout)
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise APIError("PARSE_ERROR", "OpenAI 응답에서 choices[0].message.content 를 찾지 못했습니다")
        if not text:
            raise APIError("PARSE_ERROR", f"OpenAI 응답이 비어 있습니다(finish_reason={choice.get('finish_reason')})")
        return text

    # --- Google Gemini generateContent -------------------------------------
    def _gemini(self, system: str, user: str, json_mode: bool) -> str:
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
        }
        if json_mode:
            body["generationConfig"] = {"responseMimeType": "application/json"}
        url = GEMINI_URL.format(model=self.model)
        data = http_request("POST", url, headers=headers, json_body=body, timeout=self.timeout)
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "unknown")
            raise APIError("BLOCKED", f"Gemini 가 응답을 생성하지 않았습니다(blockReason={reason})")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise APIError("PARSE_ERROR",
                           f"Gemini 응답이 비어 있습니다(finishReason={candidates[0].get('finishReason')})")
        return text


# =============================================================================
# 3. LLM 출력 -> JSON 파싱 & 스키마 검증
# =============================================================================
def extract_json(text: str) -> Any:
    """LLM 출력에서 JSON 객체를 꺼낸다. ```json 코드펜스나 앞뒤 설명문이 붙어도 처리."""
    t = (text or "").strip()
    fence = re.search(r"```(?:json|JSON)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        return json.loads(t[start:end + 1])  # 여기서도 실패하면 JSONDecodeError 가 올라간다
    raise json.JSONDecodeError("JSON 객체를 찾을 수 없습니다", t, 0)


def _clean_str(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def _normalize_city(city: str) -> str:
    """검색 정확도를 위해 '제주도'->'제주', '강원도 강릉시'->'강릉' 처럼 핵심 지명만 남긴다."""
    c = re.sub(r"\(.*?\)", "", city).strip()
    parts = c.split()
    if len(parts) >= 2 and re.search(r"(도|특별자치도|광역시|특별시)$", parts[0]):
        c = parts[-1]
    if c in ("제주도", "제주특별자치도"):
        return "제주"
    c = re.sub(r"(특별자치시|특별자치도|광역시|특별시)$", "", c)
    if len(c) >= 3 and c.endswith(("시", "군")):
        c = c[:-1]
    return c or city.strip()


def _validate_city_block(obj: Any, label: str, city_key: str) -> tuple[dict | None, list[str]]:
    problems: list[str] = []
    if not isinstance(obj, dict):
        return None, [f"{label}: 객체(dict)가 아닙니다"]
    city = _clean_str(obj.get(city_key))
    weather = _clean_str(obj.get("weather"))
    reason = _clean_str(obj.get("reason"))
    events_raw = obj.get("events")
    if isinstance(events_raw, str):          # 문자열 하나로 온 경우 관대하게 배열로 감싼다
        events_raw = [events_raw]
    events = [e.strip() for e in events_raw if isinstance(e, str) and e.strip()] \
        if isinstance(events_raw, list) else []

    if not city:
        problems.append(f"{label}: '{city_key}' 가 비어 있거나 string 이 아닙니다")
    if not weather:
        problems.append(f"{label}: 'weather' 가 비어 있거나 string 이 아닙니다")
    if not reason:
        problems.append(f"{label}: 'reason' 이 비어 있거나 string 이 아닙니다")
    if not events:
        problems.append(f"{label}: 'events' 는 string 1~3개 배열이어야 합니다")
    if problems:
        return None, problems
    return {"city": _normalize_city(city), "weather": weather,
            "events": events[:3], "reason": reason}, []


def validate_recommendation(obj: Any, num_cities: int = 1) -> dict:
    """1차 추천 JSON 을 검증하고 표준 형태로 정규화한다. 문제가 있으면 SchemaError."""
    main, problems = _validate_city_block(obj, "최상위", "recommended_city")
    cities: list[dict] = [main] if main else []

    if num_cities > 1 and isinstance(obj, dict):
        extra = obj.get("recommended_cities")
        if not isinstance(extra, list) or not extra:
            problems.append(f"'recommended_cities' 는 {num_cities}개 객체 배열이어야 합니다")
        else:
            for i, item in enumerate(extra):
                block, p = _validate_city_block(item, f"recommended_cities[{i}]", "city")
                problems.extend(p)
                if block and all(block["city"] != c["city"] for c in cities):
                    cities.append(block)
            if not problems and len(cities) < 2:
                problems.append("서로 다른 지역을 2개 이상 추천해야 합니다")
    if problems:
        raise SchemaError(problems)

    cities = cities[:num_cities]
    first = cities[0]
    return {
        # --- 과제 최소 스키마 (필수) ---
        "recommended_city": first["city"],
        "weather": first["weather"],
        "events": first["events"],
        "reason": first["reason"],
        # --- 확장 필드: 복수 지역 (기본 모드에서는 1개) ---
        "recommended_cities": cities,
    }


# =============================================================================
# 4. 프롬프트 설계
# =============================================================================
SEASONS = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
           6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


def date_context(d: date) -> str:
    period = "초" if d.day <= 10 else ("중순" if d.day <= 20 else "말")
    return f"{d.isoformat()} ({WEEKDAYS[d.weekday()]}요일, {d.month}월 {period}, {SEASONS[d.month]})"


RECOMMEND_SYSTEM = (
    "당신은 대한민국 국내 여행 전문가입니다. 사용자가 준 날짜에 여행하기 좋은 국내 지역을 추천합니다. "
    "응답은 반드시 JSON 객체 하나만 출력합니다. 마크다운 코드블록, 설명 문장, 주석을 절대 붙이지 마세요."
)


def build_recommend_prompt(d: date, num_cities: int) -> str:
    base_rules = f"""여행 날짜: {date_context(d)}

아래 규칙을 지켜 JSON 객체 하나만 출력하세요.

[필수 키와 타입]
- "recommended_city": string. 지도 검색에 그대로 쓸 수 있는 시/군 단위 지명 하나 (예: "제주", "강릉", "경주", "여수").
  "강원도", "남해안", "제주 동쪽" 처럼 넓거나 모호한 표현은 금지.
- "weather": string. 그 시기의 '일반적인' 날씨 요약 1~2문장 (평균 기온, 강수/바람 특징, 옷차림 팁).
- "events": string 배열 1~3개. 그 시기 해당 지역에서 열리는(또는 즐길 수 있는) 축제/행사/시즌 체험.
  실제 개최 일정이 확실하지 않으면 항목 끝에 "(일정 변동 가능)" 을 붙이세요.
- "reason": string. 왜 이 날짜에 이 지역이 좋은지 2~4문장 (계절 특징, 볼거리, 혼잡도/비용 등)."""

    if num_cities <= 1:
        example = """
[출력 예시 - 형식만 참고]
{"recommended_city": "제주", "weather": "3월 중순 평균 기온 10~15°C, 바람이 강한 날이 있어 겉옷이 필요합니다.",
 "events": ["제주 들불축제(일정 변동 가능)", "유채꽃 명소 탐방"],
 "reason": "3월 중순 제주는 유채꽃이 피기 시작해 봄 풍경을 즐기기 좋습니다. 여름 성수기보다 항공·숙박 비용 부담이 적습니다."}"""
        return base_rules + example

    return base_rules + f"""
- "recommended_cities": 객체 배열 정확히 {num_cities}개. 각 객체는 "city", "weather", "events", "reason" 키를 가집니다
  (각 값의 규칙은 위와 동일, "city" 는 시/군 단위 지명). 첫 번째 원소는 recommended_city 와 같은 지역이고,
  나머지는 서로 다른 지역이어야 합니다. 최상위 weather/events/reason 은 첫 번째 지역의 정보입니다.

[출력 예시 - 형식만 참고]
{{"recommended_city": "제주", "weather": "...", "events": ["..."], "reason": "...",
 "recommended_cities": [
   {{"city": "제주", "weather": "...", "events": ["..."], "reason": "..."}},
   {{"city": "경주", "weather": "...", "events": ["..."], "reason": "..."}}
 ]}}"""


def build_recommend_retry_prompt(d: date, num_cities: int, problems: list[str], previous: str) -> str:
    keys = '"recommended_city"(string), "weather"(string), "events"(string 배열 1~3개), "reason"(string 2~4문장)'
    if num_cities > 1:
        keys += f', "recommended_cities"(객체 {num_cities}개 배열, 각 객체 키: city/weather/events/reason)'
    return f"""직전 응답을 JSON 으로 파싱/검증하지 못했습니다.
문제: {"; ".join(problems)[:500]}

직전 응답(일부):
{previous[:1500]}

여행 날짜 {date_context(d)} 기준으로, 아래 필수 키만 가진 JSON 객체 하나만 다시 출력하세요.
필수 키: {keys}
설명 문장, 코드블록(```), 주석 없이 순수 JSON 만 출력하세요."""


REPORT_SYSTEM = (
    "당신은 국내 여행 리포트를 쓰는 에디터입니다. 주어진 데이터만 근거로 한국어 Markdown 리포트를 작성합니다. "
    "데이터에 없는 맛집 이름, 주소, 전화번호를 지어내지 마세요."
)


def build_report_prompt(travel_date: str, rec: dict, restaurants_by_city: dict[str, list[dict]]) -> str:
    compact = {
        city: [{"name": r["name"], "category": r.get("category", ""), "address": r.get("address", "")}
               for r in items]
        for city, items in restaurants_by_city.items()
    }
    multi = len(rec["recommended_cities"]) > 1
    multi_rule = (
        "- 추천 지역이 여러 곳이므로 '추천 지역', '추천 이유', '날씨 요약', '행사/축제' 섹션 안에서는 "
        "'### 지역명' 소제목으로 지역별로 나눠 쓰세요. '1일 일정 제안'은 첫 번째(대표) 지역 기준으로 작성하세요.\n"
        if multi else ""
    )
    return f"""아래 데이터를 바탕으로 {travel_date} 국내 여행 추천 리포트를 Markdown 으로 작성하세요.

[1차 추천 데이터(JSON)]
{json.dumps(rec, ensure_ascii=False, indent=2)}

[맛집 검색 결과(JSON, 지역별) - 0건일 수 있음]
{json.dumps(compact, ensure_ascii=False, indent=2)}

[작성 규칙]
- 아래 6개 '##' 섹션을 이 순서와 이 제목 그대로 작성하세요. 문서 맨 위 '#' 제목과 다른 섹션은 쓰지 마세요.
  ## 추천 지역
  ## 추천 이유
  ## 날씨 요약
  ## 행사/축제
  ## 맛집 추천
  ## 1일 일정 제안
- '추천 지역': 지역명과 한 줄 소개.
- '추천 이유': 1차 추천의 reason 을 2~4문장으로 자연스럽게 요약.
- '날씨 요약': weather 를 바탕으로 1~2문장 + 옷차림/준비물 팁 1개.
- '행사/축제': events 를 '- ' 목록으로. 일정이 불확실하면 방문 전 공식 일정 확인을 권하세요.
- '맛집 추천': 한 줄 소개만 쓰세요 (상세 목록은 프로그램이 데이터로 채웁니다).
- '1일 일정 제안': '- **오전**', '- **점심**', '- **오후**', '- **저녁**' 4줄 이상.
  점심/저녁에는 위 맛집 검색 결과에 있는 '이름'만 그대로 사용하세요.
  맛집 결과가 0건이면 특정 상호를 만들지 말고 "현지 맛집(검색 결과 없음)"이라고 쓰세요.
{multi_rule}- 과장 없이 실용적으로, 전체 600~1200자 내외."""


# =============================================================================
# 5. 장소 검색 (Kakao Local / Naver Local) - 둘 다 HTTP GET
# =============================================================================
def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class PlaceSearcher:
    def __init__(self, provider: str, keys: dict[str, str], timeout: int = PLACE_TIMEOUT):
        self.provider = provider
        self.keys = keys
        self.timeout = timeout

    @staticmethod
    def build_query(city: str) -> str:
        return f"{city} 맛집"

    def search_restaurants(self, city: str, size: int = 5) -> tuple[list[dict], str]:
        """(정규화된 맛집 리스트, 사용한 검색어). 0건이면 빈 리스트(예외 아님)."""
        query = self.build_query(city)
        if self.provider == "kakao":
            items = self._kakao(query, size)
        else:
            items = self._naver(query, size)
        # 이름+주소 기준 중복 제거
        seen, unique = set(), []
        for it in items:
            key = (it["name"], it["address"])
            if it["name"] and key not in seen:
                seen.add(key)
                unique.append(it)
        return unique[:size], query

    # --- Kakao Local: 키워드로 장소 검색 ----------------------------------
    def _kakao(self, query: str, size: int) -> list[dict]:
        headers = {"Authorization": f"KakaoAK {self.keys['KAKAO_REST_API_KEY']}"}
        params = {
            "query": query,
            "category_group_code": "FD6",   # FD6 = 음식점
            "size": max(1, min(size, 15)),  # Kakao 허용 범위 1~15
            "sort": "accuracy",
        }
        data = http_request("GET", KAKAO_KEYWORD_URL, headers=headers, params=params, timeout=self.timeout)
        docs = data.get("documents")
        if not isinstance(docs, list):
            raise APIError("PARSE_ERROR", "Kakao 응답에 documents 배열이 없습니다")
        results = []
        for d in docs:
            x, y = _to_float(d.get("x")), _to_float(d.get("y"))
            category = (d.get("category_name") or "").replace("음식점 > ", "", 1)
            results.append({
                "name": d.get("place_name", ""),
                "address": d.get("road_address_name") or d.get("address_name", ""),
                "category": category,
                "url": d.get("place_url", ""),
                "phone": d.get("phone", ""),
                "x": x, "y": y,          # x=경도, y=위도 (WGS84)
                "lat": y, "lng": x,
                "source": "kakao",
            })
        return results

    # --- Naver Local Search ------------------------------------------------
    def _naver(self, query: str, size: int) -> list[dict]:
        headers = {
            "X-Naver-Client-Id": self.keys["NAVER_CLIENT_ID"],
            "X-Naver-Client-Secret": self.keys["NAVER_CLIENT_SECRET"],
        }
        params = {"query": query, "display": max(1, min(size, 5)), "start": 1, "sort": "random"}
        data = http_request("GET", NAVER_LOCAL_URL, headers=headers, params=params, timeout=self.timeout)
        items = data.get("items")
        if not isinstance(items, list):
            raise APIError("PARSE_ERROR", "Naver 응답에 items 배열이 없습니다")
        results = []
        for it in items:
            name = html.unescape(re.sub(r"<[^>]+>", "", it.get("title", ""))).strip()
            mapx, mapy = _to_float(it.get("mapx")), _to_float(it.get("mapy"))
            # Naver mapx/mapy 는 WGS84 좌표 x 10,000,000 정수 문자열
            lng = round(mapx / 1e7, 7) if mapx and mapx > 1000 else mapx
            lat = round(mapy / 1e7, 7) if mapy and mapy > 1000 else mapy
            url = it.get("link") or f"https://map.naver.com/p/search/{quote(name)}"
            results.append({
                "name": name,
                "address": it.get("roadAddress") or it.get("address", ""),
                "category": it.get("category", ""),
                "url": url,
                "phone": it.get("telephone", ""),
                "x": lng, "y": lat,
                "lat": lat, "lng": lng,
                "source": "naver",
            })
        return results


# =============================================================================
# 6. Markdown 리포트 조립
# =============================================================================
def md_escape(s: Any) -> str:
    return str(s or "").replace("|", "\\|").replace("\n", " ").strip()


def split_sections(md: str) -> dict[str, str]:
    """LLM 이 쓴 Markdown 을 '## 제목' 기준으로 잘라 섹션 키(region/reason/...)에 매핑한다."""
    matchers = {
        "region": ("추천지역",), "reason": ("추천이유", "이유"), "weather": ("날씨",),
        "events": ("행사", "축제"), "restaurants": ("맛집",), "itinerary": ("일정",),
    }
    sections: dict[str, str] = {}
    current, buf = None, []

    def flush():
        if current and "\n".join(buf).strip() and current not in sections:
            sections[current] = "\n".join(buf).strip()

    text = re.sub(r"^```(?:markdown|md)?\s*|```\s*$", "", md.strip())
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+?)\s*#*\s*$", line)
        if m and not line.startswith("###"):
            flush()
            title = re.sub(r"[\s()（）\d.\-:*]", "", m.group(1))
            current = next((k for k, keys in matchers.items() if any(t in title for t in keys)), None)
            buf = []
        elif line.startswith("# ") and not line.startswith("##"):
            continue  # LLM 이 붙인 문서 제목은 버리고 프로그램 제목을 쓴다
        elif current:
            buf.append(line)
    flush()
    return sections


def render_restaurants(restaurants_by_city: dict[str, list[dict]], place_status: dict[str, str],
                       multi: bool) -> str:
    out: list[str] = []
    for city, items in restaurants_by_city.items():
        if multi:
            out.append(f"### {city}")
        if not items:
            out.append(f"- 데이터 없음 ({place_status.get(city, '장소 검색 결과 0건')})")
        else:
            out.append("| # | 이름 | 카테고리 | 주소 | 좌표(위도, 경도) |")
            out.append("|---|---|---|---|---|")
            for i, r in enumerate(items, 1):
                name = md_escape(r["name"])
                name_cell = f"[{name}]({r['url']})" if r.get("url") else name
                coord = f"{r['lat']:.5f}, {r['lng']:.5f}" if r.get("lat") is not None and r.get("lng") is not None else "-"
                out.append(f"| {i} | {name_cell} | {md_escape(r.get('category')) or '-'} | "
                           f"{md_escape(r.get('address')) or '-'} | {coord} |")
        out.append("")
    return "\n".join(out).strip()


def render_errors(errors: list[dict]) -> str:
    if not errors:
        return "- 없음 (모든 단계가 정상 완료되었습니다)"
    lines = ["| 단계(step) | 유형(type) | 메시지 |", "|---|---|---|"]
    for e in errors:
        lines.append(f"| {e['step']} | {e['type']} | {md_escape(e['message'])} |")
    return "\n".join(lines)


def fallback_sections(rec: dict, restaurants_by_city: dict[str, list[dict]]) -> dict[str, str]:
    """LLM 리포트 생성이 실패/누락됐을 때 데이터만으로 섹션을 채운다."""
    cities = rec["recommended_cities"]
    multi = len(cities) > 1

    def per_city(fn):
        if not multi:
            return fn(cities[0])
        return "\n\n".join(f"### {c['city']}\n{fn(c)}" for c in cities)

    main = cities[0]
    names = [r["name"] for r in restaurants_by_city.get(main["city"], [])]
    lunch = names[0] if names else "현지 맛집(검색 결과 없음)"
    dinner = names[1] if len(names) > 1 else (names[0] if names else "현지 맛집(검색 결과 없음)")
    event = main["events"][0] if main["events"] else f"{main['city']} 대표 명소"
    return {
        "region": per_city(lambda c: f"**{c['city']}**"),
        "reason": per_city(lambda c: c["reason"]),
        "weather": per_city(lambda c: c["weather"]),
        "events": per_city(lambda c: "\n".join(f"- {e}" for e in c["events"])),
        "restaurants": "지도/장소 검색 API 로 찾은 음식점입니다.",
        "itinerary": "\n".join([
            f"- **오전**: {main['city']} 시내/대표 명소 산책",
            f"- **점심**: {lunch}",
            f"- **오후**: {event}",
            f"- **저녁**: {dinner}",
            "",
            "> LLM 리포트 생성에 실패해 데이터 기반 기본 일정으로 대체했습니다.",
        ]),
    }


def assemble_report(travel_date: str, rec: dict, restaurants_by_city: dict[str, list[dict]],
                    place_status: dict[str, str], llm_md: str | None, errors: list[dict],
                    meta: dict) -> tuple[str, list[str]]:
    """최종 Markdown 을 고정된 섹션 순서로 조립한다. (리포트, LLM 이 빠뜨린 섹션 목록)"""
    llm_sections = split_sections(llm_md) if llm_md else {}
    fb = fallback_sections(rec, restaurants_by_city)
    multi = len(rec["recommended_cities"]) > 1
    missing: list[str] = []

    lines = [f"# {travel_date} 국내 여행 추천 리포트", ""]
    cities_txt = ", ".join(c["city"] for c in rec["recommended_cities"])
    lines += [f"> 생성: {meta['generated_at']} · LLM: {meta['llm_provider']}({meta['llm_model']}) · "
              f"장소 검색: {meta['place_provider']} · 추천 지역: {cities_txt}", ""]

    for key, title in SECTION_ORDER:
        lines.append(f"## {title}")
        if key == "errors":
            body = render_errors(errors)
        elif key == "restaurants":
            # LLM 이 표/목록을 넣었더라도 상호 정확도를 위해 '소개 문장'만 쓰고, 목록은 API 데이터로 채운다
            intro_lines = []
            for ln in llm_sections.get("restaurants", "").splitlines():
                if re.match(r"^\s*(\||[-*+]\s|\d+\.\s|#)", ln):
                    break
                intro_lines.append(ln)
            intro = "\n".join(intro_lines).strip()
            table = render_restaurants(restaurants_by_city, place_status, multi)
            body = f"{intro}\n\n{table}".strip() if intro else table
        else:
            body = llm_sections.get(key)
            if not body:
                if llm_md:
                    missing.append(title)
                body = fb[key]
            if key == "itinerary" and llm_sections.get(key):
                main_city = rec["recommended_cities"][0]["city"]
                if not restaurants_by_city.get(main_city):
                    body += "\n\n> 맛집 검색 데이터가 없어 식사 장소는 현지에서 직접 확인하세요."
        lines += [body, ""]

    lines += ["---", "*날씨·행사 정보는 LLM 이 생성한 일반 정보로, 실제 일정·기상과 다를 수 있습니다. "
              "방문 전 공식 정보를 확인하세요.*", ""]
    return "\n".join(lines), missing


# =============================================================================
# 7. 결과 저장 / 캐시
# =============================================================================
def write_text(path: Path, text: str) -> None:
    """임시 파일에 쓴 뒤 교체(중간에 실패해도 기존 파일이 깨지지 않게)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def display_path(path: Path) -> str:
    """가능하면 현재 폴더 기준 상대경로(results/...)로 보여준다."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def save_json(path: Path, data: dict) -> None:
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def load_cache(path: Path, num_cities: int) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log("    - 캐시 파일을 읽지 못해 새로 호출합니다.")
        return None
    rec = data.get("recommendation")
    if not isinstance(rec, dict) or not rec.get("recommended_city"):
        return None  # 1단계가 실패했던 기록이면 캐시로 쓰지 않는다
    if data.get("meta", {}).get("num_cities", 1) != num_cities:
        log("    - 캐시의 --cities 값이 달라 새로 호출합니다.")
        return None
    return data


# =============================================================================
# 8. CLI / 설정
# =============================================================================
def valid_date(value: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value or ""):
        raise argparse.ArgumentTypeError(f"날짜 형식이 올바르지 않습니다: '{value}' (예: 2026-03-15)")
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"존재하지 않는 날짜입니다: '{value}'")


def int_range(lo: int, hi: int):
    def _check(v: str) -> int:
        try:
            n = int(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"정수를 입력하세요: '{v}'")
        if not lo <= n <= hi:
            raise argparse.ArgumentTypeError(f"{lo}~{hi} 사이 값이어야 합니다: {n}")
        return n
    return _check


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="travel_planner.py",
        description="여행 날짜를 입력하면 LLM + 지도 API 로 국내 여행 리포트를 만들어 results/ 에 저장합니다.",
        epilog='예: python travel_planner.py --date "2026-03-15"',
    )
    p.add_argument("-date", "--date", dest="date", type=valid_date, required=True,
                   metavar="YYYY-MM-DD", help="여행 날짜 (필수)")
    p.add_argument("--llm", choices=["openai", "gemini"], default=None,
                   help="LLM 제공자 (기본: .env 의 LLM_PROVIDER, 없으면 openai)")
    p.add_argument("--place", choices=["kakao", "naver"], default=None,
                   help="장소 검색 제공자 (기본: .env 의 PLACE_PROVIDER, 없으면 kakao)")
    p.add_argument("--num-places", type=int_range(1, 15), default=5, help="지역별 맛집 개수 (기본 5)")
    p.add_argument("--cities", type=int_range(1, 3), default=1, help="[보너스] 추천 지역 수 1~3 (기본 1)")
    p.add_argument("--refresh", action="store_true", help="[보너스] 캐시를 무시하고 API 를 새로 호출")
    p.add_argument("--regen-report", action="store_true",
                   help="[보너스] 캐시된 1·2단계 데이터로 리포트만 LLM 으로 다시 생성")
    p.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR), help="결과 저장 폴더 (기본 ./results)")
    p.add_argument("-v", "--verbose", action="store_true", help="LLM 원문 응답 일부를 출력")
    return p


def key_setup_help(missing: list[str]) -> str:
    first = missing[0]
    return f"""
[오류] API 키가 설정되지 않았습니다: {", ".join(missing)}

설정 방법 (둘 중 하나):
  1) 프로젝트 폴더의 .env 파일에 추가 (권장)
       cp .env.example .env      # Windows: copy .env.example .env
       # .env 를 열어 아래처럼 값을 채우세요 (따옴표/공백 없이)
       {first}=발급받은_키
  2) 현재 터미널에서 환경변수로 설정
       macOS/Linux : export {first}="발급받은_키"
       PowerShell  : $env:{first}="발급받은_키"

키 발급/설정 절차는 README.md 의 'API 키 발급' 절을 참고하세요.
※ 키를 코드·README·결과 파일에 직접 적지 마세요.
"""


def read_keys(names: list[str]) -> tuple[dict[str, str], list[str]]:
    keys, missing = {}, []
    for n in names:
        v = os.environ.get(n, "").strip().strip('"').strip("'")
        if not v or v.upper().startswith(("YOUR_", "여기에")):
            missing.append(n)
        else:
            keys[n] = v
    return keys, missing


# =============================================================================
# 9. 단계별 실행
# =============================================================================
def step_recommend(llm: LLMClient, d: date, num_cities: int, errors: list[dict],
                   verbose: bool) -> tuple[dict | None, list[str]]:
    """1차 추천. JSON 파싱/스키마 실패 시 '필수 키만 다시' 프롬프트로 정확히 1회 재시도."""
    raw_outputs: list[str] = []
    prompt = build_recommend_prompt(d, num_cities)
    for attempt in (1, 2):
        try:
            text = llm.generate(RECOMMEND_SYSTEM, prompt, json_mode=True)
        except APIError as e:
            add_error(errors, "recommend", e.error_type, e.message)
            log(f"    - 오류: {e.error_type} - {redact(e.message)}")
            if e.error_type == "AUTH_ERROR":
                log("      → LLM API 키 값/권한(결제·크레딧 포함)을 확인하세요.")
            elif e.error_type == "NOT_FOUND":
                log("      → 모델명을 확인하세요(.env 의 OPENAI_MODEL / GEMINI_MODEL).")
            return None, raw_outputs
        raw_outputs.append(text)
        if verbose:
            log(f"    [LLM 원문 {attempt}차] {redact(text)[:400]}")
        try:
            return validate_recommendation(extract_json(text), num_cities), raw_outputs
        except (json.JSONDecodeError, SchemaError) as e:
            problems = e.problems if isinstance(e, SchemaError) else [f"JSON 파싱 오류: {e.msg}"]
            add_error(errors, "recommend", "PARSE_ERROR", f"{attempt}차 응답 검증 실패: {'; '.join(problems)}")
            if attempt == 1:
                log("    - JSON 파싱/검증 실패 → 필수 키만 다시 요청합니다(재시도 1회)")
                prompt = build_recommend_retry_prompt(d, num_cities, problems, text)
            else:
                log("    - 재시도 후에도 JSON 파싱 실패. 더 이상 재시도하지 않습니다.")
    return None, raw_outputs


def step_places(searcher: PlaceSearcher, cities: list[str], size: int, errors: list[dict]
                ) -> tuple[dict[str, list[dict]], dict[str, str], dict[str, str]]:
    restaurants_by_city: dict[str, list[dict]] = {}
    status: dict[str, str] = {}
    queries: dict[str, str] = {}
    for city in cities:
        query = searcher.build_query(city)
        queries[city] = query
        prefix = f"    - [{city}] " if len(cities) > 1 else "    - "
        try:
            items, query = searcher.search_restaurants(city, size)
        except APIError as e:
            restaurants_by_city[city] = []
            add_error(errors, "place_search", e.error_type, f"{e.message} (query={query})")
            if e.error_type == "AUTH_ERROR":
                hint = ("REST API 키 값과 [카카오맵] 사용 설정(ON)" if searcher.provider == "kakao"
                        else "Client ID/Secret 값과 '검색' API 사용 설정")
                log(f"{prefix}오류: 인증 실패({e.status}). 키 설정을 확인하세요 → {hint}")
                status[city] = f"장소 검색 API 인증 실패(HTTP {e.status})"
            elif e.error_type == "QUOTA_ERROR":
                log(f"{prefix}오류: 쿼터 초과(429). 사용량/요금 설정을 확인하세요.")
                status[city] = "장소 검색 API 쿼터 초과(HTTP 429)"
            else:
                log(f"{prefix}오류: {e.error_type} - {redact(e.message)}")
                status[city] = f"장소 검색 API 오류({e.error_type})"
            log(f"{prefix}맛집 섹션은 '데이터 없음'으로 처리하고 계속 진행합니다.")
            continue
        restaurants_by_city[city] = items
        if items:
            status[city] = f"{len(items)}건"
            log(f"{prefix}맛집 {len(items)}곳 검색 완료")
        else:
            add_error(errors, "place_search", "EMPTY_RESULT", f"0 results for query={query}")
            status[city] = "장소 검색 결과 0건"
            log(f"{prefix}검색 결과 0건(재시도하지 않고 '데이터 없음'으로 다음 단계 진행)")
        if len(cities) > 1:
            time.sleep(0.2)  # 연속 호출 간 짧은 간격 (쿼터/레이트리밋 배려)
    return restaurants_by_city, status, queries


def step_report(llm: LLMClient | None, travel_date: str, rec: dict, restaurants_by_city: dict,
                place_status: dict, errors: list[dict], meta: dict, verbose: bool) -> str:
    llm_md = None
    if llm is not None:
        try:
            llm_md = llm.generate(REPORT_SYSTEM, build_report_prompt(travel_date, rec, restaurants_by_city),
                                  json_mode=False)
            if verbose:
                log(f"    [LLM 원문] {redact(llm_md)[:400]}")
        except APIError as e:
            add_error(errors, "report", e.error_type, e.message)
            log(f"    - 오류: {e.error_type} - {redact(e.message)}")
            log("    - 데이터 기반 기본 리포트로 대체합니다.")
    report, missing = assemble_report(travel_date, rec, restaurants_by_city, place_status, llm_md, errors, meta)
    if missing:
        add_error(errors, "report", "MISSING_SECTION",
                  f"LLM 응답에 섹션 누락 → 데이터로 대체: {', '.join(missing)}")
        report, _ = assemble_report(travel_date, rec, restaurants_by_city, place_status, llm_md, errors, meta)
    return report


# =============================================================================
# 10. main
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    try:  # Windows 콘솔에서 한글 깨짐 방지
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    if load_dotenv:
        load_dotenv(BASE_DIR / ".env")  # 스크립트 옆 .env
        load_dotenv()                   # 현재 폴더 .env (이미 설정된 값은 덮어쓰지 않음)

    args = build_parser().parse_args(argv)
    travel_date: date = args.date
    ds = travel_date.isoformat()
    llm_provider = (args.llm or os.environ.get("LLM_PROVIDER", "openai")).strip().lower()
    place_provider = (args.place or os.environ.get("PLACE_PROVIDER", "kakao")).strip().lower()
    if llm_provider not in ("openai", "gemini") or place_provider not in ("kakao", "naver"):
        log("[오류] LLM_PROVIDER 는 openai|gemini, PLACE_PROVIDER 는 kakao|naver 중 하나여야 합니다.")
        return 2
    model = os.environ.get(f"{llm_provider.upper()}_MODEL", "").strip() or DEFAULT_MODELS[llm_provider]

    out_dir = Path(args.out_dir)
    raw_path = out_dir / f"{ds}_raw.json"
    md_path = out_dir / f"{ds}_travel_plan.md"

    log(f"여행 날짜: {date_context(travel_date)}")
    if travel_date < date.today():
        log("  (참고) 지난 날짜입니다. 해당 시기의 일반적인 정보로 추천합니다.")

    # ---- 캐시 확인 (보너스) -------------------------------------------------
    cache = None if args.refresh else load_cache(raw_path, args.cities)
    if cache and not args.regen_report and cache.get("report_markdown"):
        log(f"[캐시] {raw_path.name} 이(가) 있어 API 호출 없이 저장된 결과를 사용합니다. (새로 받으려면 --refresh)")
        write_text(md_path, cache["report_markdown"])
        log(f"\n완료! {display_path(md_path)} 를 확인하세요.\n  - 원본 데이터: {display_path(raw_path)}")
        return 0

    # ---- API 키 확인: 없으면 즉시 종료 --------------------------------------
    needed = list(REQUIRED_KEYS[llm_provider])
    if not cache:
        needed += REQUIRED_KEYS[place_provider]
    keys, missing = read_keys(needed)
    if missing:
        log(key_setup_help(missing))
        return 1

    llm = LLMClient(llm_provider, keys[REQUIRED_KEYS[llm_provider][0]], model)
    errors: list[dict] = []
    meta = {
        "program_version": VERSION,
        "travel_date": ds,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "llm_provider": llm_provider,
        "llm_model": model,
        "place_provider": place_provider,
        "num_cities": args.cities,
        "num_places": args.num_places,
        "from_cache": bool(cache),
    }

    if cache:
        log(f"[캐시] {raw_path.name} 의 1·2단계 데이터를 재사용하고 리포트만 다시 생성합니다.")
        rec = cache["recommendation"]
        restaurants_by_city = cache.get("restaurants_by_city") or {rec["recommended_city"]: cache.get("restaurants", [])}
        place_status = cache.get("place_search_status", {})
        queries = cache.get("place_search_queries", {})
        raw_outputs = cache.get("recommendation_raw_outputs", [])
        errors = [e for e in cache.get("errors", []) if e.get("step") != "report"]
        meta["place_provider"] = cache.get("meta", {}).get("place_provider", place_provider)
    else:
        # ---- [1/3] 1차 추천 ---------------------------------------------------
        log("[1/3] 1차 추천 생성 중(LLM)...")
        rec, raw_outputs = step_recommend(llm, travel_date, args.cities, errors, args.verbose)
        if rec is None:
            save_json(raw_path, {"meta": meta, "recommendation": None, "recommendation_raw_outputs":
                                 [redact(t) for t in raw_outputs], "restaurants": [], "errors": errors})
            log("\n[중단] 1차 추천을 만들지 못해 다음 단계로 진행할 수 없습니다.")
            log(f"  - 오류 기록: {display_path(raw_path)}")
            return 1
        for c in rec["recommended_cities"]:
            log(f'    - recommended_city: "{c["city"]}"  / events: {len(c["events"])}개')

        # ---- [2/3] 맛집 검색 --------------------------------------------------
        log("[2/3] 맛집 검색 중(지도/장소 API)...")
        searcher = PlaceSearcher(place_provider, keys)
        restaurants_by_city, place_status, queries = step_places(
            searcher, [c["city"] for c in rec["recommended_cities"]], args.num_places, errors)

    # ---- [3/3] 최종 리포트 --------------------------------------------------
    log("[3/3] 최종 리포트 생성 중(LLM)...")
    report = step_report(llm, ds, rec, restaurants_by_city, place_status, errors, meta, args.verbose)
    log("    - 리포트 생성 완료" + (" (일부 오류는 '오류 요약' 섹션 참고)" if errors else ""))

    raw = {
        "meta": meta,
        "recommendation": rec,                                        # 1차 추천 JSON (파싱 결과)
        "recommendation_raw_outputs": [redact(t) for t in raw_outputs],
        "place_search_queries": queries,
        "place_search_status": place_status,
        "restaurants": restaurants_by_city.get(rec["recommended_city"], []),  # 대표 지역 맛집 (0건 가능)
        "restaurants_by_city": restaurants_by_city,
        "errors": errors,                                             # 오류 요약 (빈 리스트 가능)
        "report_markdown": report,
        "files": {"raw_json": str(raw_path), "report_md": str(md_path)},
    }
    save_json(raw_path, raw)
    write_text(md_path, report)

    log(f"\n완료! {display_path(md_path)} 를 확인하세요.")
    log(f"  - 원본 데이터: {display_path(raw_path)}")
    if errors:
        log(f"  - 기록된 오류/경고: {len(errors)}건 ({', '.join(sorted({e['type'] for e in errors}))})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n사용자가 중단했습니다.")
        sys.exit(130)
