# 국내 여행 추천 프로그램 (LLM API + 지도 API)

여행 날짜를 입력하면 **LLM API** 가 그 시기에 가기 좋은 국내 지역·날씨·행사를 JSON 으로 추천하고,
**지도/장소 검색 API** 로 그 지역 맛집을 찾은 뒤, 다시 **LLM API** 로 최종 여행 리포트(Markdown)를 만드는
터미널(CLI) 프로그램입니다.

```
 --date "2026-03-15"
        │
        ▼
 [1/3] LLM API  (HTTP POST)  ──►  1차 추천 JSON {recommended_city, weather, events[], reason}
        │                              │ recommended_city 를 다음 단계 입력으로 사용
        ▼                              ▼
 [2/3] 장소 API (HTTP GET)   ──►  맛집 N곳 [{name, address, category, url, x, y, lat, lng}]  (0건 가능)
        │
        ▼
 [3/3] LLM API  (HTTP POST)  ──►  최종 리포트 Markdown
        │
        ▼
 results/2026-03-15_raw.json  +  results/2026-03-15_travel_plan.md
```

| 구분 | 지원 제공자 | 선택 방법 |
|---|---|---|
| LLM | OpenAI (Chat Completions) / Google Gemini (generateContent) | `.env` 의 `LLM_PROVIDER` 또는 `--llm` |
| 장소 검색 | Kakao Local (키워드 검색, 음식점 FD6) / Naver Local Search | `.env` 의 `PLACE_PROVIDER` 또는 `--place` |

과제 조건은 "각각 택1" 이므로 **OpenAI + Kakao** (기본값) 조합 하나만 준비해도 됩니다.

---

## 1. 폴더 구조

```
travel_planner/
├── travel_planner.py          # 메인 프로그램 (CLI)
├── requirements.txt           # 의존 패키지 (requests, python-dotenv)
├── .env.example               # 키 설정 템플릿 (값은 비어 있음)
├── .gitignore                 # .env 가 Git 에 올라가지 않도록 차단
├── README.md                  # 이 문서
├── docs/
│   └── CONCEPTS.md            # 과제 목표 4가지 개념 정리 (REST, JSON 연결, 오류, 키 관리)
├── tests/
│   └── test_travel_planner.py # 키 없이 돌아가는 모의(mock) 테스트 24개
└── results/                   # 실행하면 자동 생성
    ├── 2026-03-15_raw.json
    └── 2026-03-15_travel_plan.md
```

---

## 2. 실행 환경 준비

### 2-1. Python 3.10 이상 확인

```bash
python --version        # Windows
python3 --version       # macOS/Linux  → Python 3.10.x 이상이면 OK
```

### 2-2. 가상환경 만들고 패키지 설치

```bash
cd travel_planner

# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
source .venv/bin/activate          # macOS/Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell
# (PowerShell 에서 실행 정책 오류 시: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)

# 패키지 설치
pip install -r requirements.txt
```

---

## 3. API 키 발급

> 콘솔 화면의 메뉴 이름은 서비스 개편에 따라 조금씩 바뀔 수 있습니다.

### 3-1. LLM (둘 중 하나)

**OpenAI**
1. https://platform.openai.com 로그인 → **API keys** → **Create new secret key**
2. 표시된 키(`sk-...`)를 복사해 둡니다 (창을 닫으면 다시 볼 수 없음).
3. **Billing 에 결제수단/크레딧이 있어야** 호출됩니다. 없으면 `429 insufficient_quota` 가 납니다.

**Google Gemini**
1. https://aistudio.google.com 로그인 → **Get API key** → **Create API key**
2. 표시된 키(`AIza...`)를 복사합니다. 무료 등급으로도 실습할 수 있습니다.

### 3-2. 지도/장소 검색 (둘 중 하나)

**Kakao Local (기본값)**
1. https://developers.kakao.com 로그인 → **내 애플리케이션** → **애플리케이션 추가하기**
2. 앱의 **앱 키** 에서 **REST API 키** 를 복사합니다. (JavaScript 키·네이티브 앱 키가 아닙니다!)
3. 앱 관리 페이지의 **[카카오맵] → [사용 설정] 상태를 ON** 으로 바꿉니다.
   - 2024-12-01 이후 새로 카카오맵/로컬 API 를 쓰는 앱은 이 설정이 없으면 401/403 이 납니다.
   - 2026-07-21 정책 변경으로, 무료 쿼터는 **개발자 계정에서 처음 카카오맵을 활성화한 앱** 에만 제공됩니다.
     두 번째 앱부터는 비즈월렛·유료 설정이 필요할 수 있으니 가능하면 첫 앱으로 실습하세요.

**Naver Local Search**
1. https://developers.naver.com → **Application → 애플리케이션 등록**
2. 사용 API 에서 **검색** 을 선택, 비로그인 오픈 API 서비스 환경에 **WEB 설정 → `http://localhost`** 입력
3. 발급된 **Client ID / Client Secret** 을 복사합니다. (Naver 는 한 번에 최대 5건까지 조회됩니다)

---

## 4. API 키 설정 (코드에 직접 쓰지 않기)

### 방법 A — `.env` 파일 (권장)

```bash
cp .env.example .env        # macOS/Linux
copy .env.example .env      # Windows
```

`.env` 를 열어 사용하는 항목만 채웁니다. **따옴표·공백 없이** 적습니다.

```dotenv
LLM_PROVIDER=openai
PLACE_PROVIDER=kakao
OPENAI_API_KEY=여기에_발급받은_키
KAKAO_REST_API_KEY=여기에_발급받은_키
```

프로그램은 시작할 때 `python-dotenv` 로 이 파일을 읽어 환경변수로 올립니다.
`.env` 는 `.gitignore` 에 들어 있어 Git 에 커밋되지 않습니다.

### 방법 B — 터미널 환경변수 (현재 세션에만 적용)

```bash
# macOS/Linux
export OPENAI_API_KEY="YOUR_KEY"
export KAKAO_REST_API_KEY="YOUR_KEY"
```

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:KAKAO_REST_API_KEY="YOUR_KEY"
```

| 환경변수 | 필요한 경우 |
|---|---|
| `OPENAI_API_KEY` | `LLM_PROVIDER=openai` |
| `GEMINI_API_KEY` | `LLM_PROVIDER=gemini` |
| `KAKAO_REST_API_KEY` | `PLACE_PROVIDER=kakao` |
| `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` | `PLACE_PROVIDER=naver` |
| `OPENAI_MODEL` / `GEMINI_MODEL` (선택) | 기본 모델(`gpt-4o-mini` / `gemini-2.5-flash`)을 바꾸고 싶을 때. 모델이 단종돼 404 가 나면 여기서 교체 |

키가 비어 있으면 프로그램은 **API 를 호출하지 않고 즉시 종료**하며 설정 방법을 출력합니다.

---

## 5. 실행 방법

```bash
python travel_planner.py --date "2026-03-15"
# 과제 표기 그대로 -date 도 됩니다
python travel_planner.py -date "2026-03-15"
```

정상 실행 예:

```
여행 날짜: 2026-03-15 (일요일, 3월 중순, 봄)
[1/3] 1차 추천 생성 중(LLM)...
    - recommended_city: "제주"  / events: 2개
[2/3] 맛집 검색 중(지도/장소 API)...
    - 맛집 5곳 검색 완료
[3/3] 최종 리포트 생성 중(LLM)...
    - 리포트 생성 완료

완료! results/2026-03-15_travel_plan.md 를 확인하세요.
  - 원본 데이터: results/2026-03-15_raw.json
```

### 옵션

| 옵션 | 설명 |
|---|---|
| `-date`, `--date YYYY-MM-DD` | **필수.** 여행 날짜. 형식이 틀리거나 없는 날짜(2026-02-30)면 사용법을 출력하고 종료 |
| `--llm openai\|gemini` | LLM 제공자 (`.env` 의 `LLM_PROVIDER` 보다 우선) |
| `--place kakao\|naver` | 장소 검색 제공자 (`.env` 의 `PLACE_PROVIDER` 보다 우선) |
| `--num-places N` | 지역별 맛집 수 1~15 (기본 5, Naver 는 최대 5) |
| `--cities N` | **[보너스]** 추천 지역 수 1~3. 2 이상이면 지역마다 맛집을 검색하고 리포트를 지역별로 정리 |
| `--refresh` | **[보너스]** 캐시를 무시하고 모든 API 를 새로 호출 |
| `--regen-report` | **[보너스]** 저장된 1·2단계 데이터는 재사용하고 리포트만 LLM 으로 다시 생성 |
| `--out-dir 경로` | 결과 저장 폴더 (기본 `results/`) |
| `-v`, `--verbose` | LLM 원문 응답 일부 출력 (프롬프트 디버깅용) |

예시:

```bash
python travel_planner.py --date "2026-10-03" --cities 3            # 복수 지역
python travel_planner.py --date "2026-07-20" --llm gemini --place naver
python travel_planner.py --date "2026-03-15"                       # 두 번째 실행 → 캐시 사용 (API 호출 0회)
python travel_planner.py --date "2026-03-15" --refresh             # 새로 호출
python travel_planner.py --date "2026/03/15"                       # 형식 오류 → 사용법 출력 후 종료
```

---

## 6. 결과물 확인 방법

실행 후 `results/` 폴더에 **입력한 여행 날짜** 기준 이름으로 두 파일이 생깁니다.
(실제 실행 시각은 JSON 의 `meta.generated_at` 과 리포트 상단에 기록됩니다.)

### 6-1. 원본 데이터 `results/YYYY-MM-DD_raw.json`

```jsonc
{
  "meta": { "travel_date": "...", "generated_at": "...", "llm_provider": "openai", "llm_model": "...",
            "place_provider": "kakao", "num_cities": 1, "from_cache": false },
  "recommendation": {                       // ← 1차 추천 JSON (파싱·검증 결과)
    "recommended_city": "제주", "weather": "...", "events": ["...", "..."], "reason": "...",
    "recommended_cities": [ { "city": "제주", ... } ]   // 보너스용 확장 필드 (기본 1개)
  },
  "recommendation_raw_outputs": ["LLM 원문 응답"],
  "place_search_queries": { "제주": "제주 맛집" },
  "restaurants": [                          // ← 맛집 검색 결과 (0건이면 [])
    { "name": "...", "address": "...", "category": "...", "url": "...",
      "phone": "...", "x": 126.5, "y": 33.4, "lat": 33.4, "lng": 126.5, "source": "kakao" }
  ],
  "restaurants_by_city": { "제주": [ ... ] },
  "errors": [                               // ← 오류 요약 (없으면 [])
    { "step": "place_search", "type": "AUTH_ERROR", "message": "HTTP 401: ...", "at": "..." }
  ],
  "report_markdown": "# ... (최종 리포트 전문, 캐시용)"
}
```

### 6-2. 최종 리포트 `results/YYYY-MM-DD_travel_plan.md`

VS Code 에서 열고 `Ctrl+Shift+V`(macOS `Cmd+Shift+V`)로 미리보기하면 표와 링크가 보기 좋게 렌더링됩니다.
섹션은 항상 아래 순서로 고정됩니다.

```
# 2026-03-15 국내 여행 추천 리포트
## 추천 지역
## 추천 이유
## 날씨 요약
## 행사/축제
## 맛집 추천         ← API 데이터로 만든 표(이름 링크·카테고리·주소·좌표). 0건/실패 시 "데이터 없음 (사유)"
## 1일 일정 제안      ← 오전/점심/오후/저녁
## 오류 요약(errors)  ← 없으면 "없음"
```

**리포트 품질을 위해 한 장치**
- 맛집 표는 LLM 이 쓰지 않고 **API 응답 데이터로 직접** 만듭니다 → 존재하지 않는 가게가 섞이지 않습니다.
- LLM 에게는 "맛집 검색 결과에 있는 이름만 일정에 쓰라" 고 지시하고, 0건이면 상호를 만들지 말라고 지시합니다.
- LLM 이 섹션을 빠뜨리면 1차 추천 데이터로 그 섹션을 채우고 `MISSING_SECTION` 으로 기록합니다.
- 리포트 생성 LLM 호출이 실패해도 데이터 기반 기본 리포트를 만들어 저장합니다.
- 추천 지역은 검색 정확도를 위해 "강원도 강릉시" → "강릉" 처럼 정규화합니다.

---

## 7. 에러 처리 정책

| 상황 | 프로그램 동작 | errors 기록 (`type`) |
|---|---|---|
| API 키 미설정 | 호출 없이 **즉시 종료** + 설정 방법 안내 (종료코드 1) | — |
| 날짜 형식 오류 | 사용법(usage) 출력 후 종료 (종료코드 2) | — |
| 1차 추천 JSON 파싱/스키마 실패 | "필수 키만 다시 JSON 으로" 프롬프트로 **정확히 1회** 재시도. 또 실패하면 중단 | `PARSE_ERROR` |
| 1차 추천 LLM 인증/쿼터 실패 | 다음 단계 입력이 없으므로 중단, raw JSON 에 오류 저장 | `AUTH_ERROR` / `QUOTA_ERROR` |
| 장소 API 401/403 | 맛집 = "데이터 없음", **리포트 생성 계속** | `AUTH_ERROR` |
| 장소 API 429 | 맛집 = "데이터 없음", 계속 | `QUOTA_ERROR` |
| 장소 API 네트워크 오류/타임아웃(10초) | 맛집 = "데이터 없음", 계속 | `NETWORK_ERROR` |
| 장소 검색 0건 | 재시도 없이 "데이터 없음"으로 계속 | `EMPTY_RESULT` |
| 리포트 LLM 실패 | 데이터 기반 기본 리포트로 대체 | `SERVER_ERROR` 등 (step=`report`) |
| LLM 네트워크/5xx | 2초 후 1회만 재시도 (인증·쿼터 오류는 재시도 안 함) | `NETWORK_ERROR` / `SERVER_ERROR` |

### 자주 만나는 오류와 점검 포인트

| 증상 | 점검 |
|---|---|
| Kakao `401`/`403` | REST API 키를 썼는지(JavaScript 키 X), `Authorization: KakaoAK {키}` 형식, **[카카오맵] 사용 설정 ON** |
| Naver `401` | Client ID/Secret 오타, 헤더명 `X-Naver-Client-Id` / `X-Naver-Client-Secret`, 앱에 **검색** API 추가 여부 |
| OpenAI `401` | 키 복사 누락/공백, 폐기된 키 |
| OpenAI `429 insufficient_quota` | 결제수단/크레딧 없음 |
| Gemini `400 API key not valid` | 키 오타 (프로그램은 이를 `AUTH_ERROR` 로 분류) |
| LLM `404` | 모델명이 바뀌었거나 단종 → `.env` 의 `OPENAI_MODEL` / `GEMINI_MODEL` 수정 |
| 한글 깨짐(Windows) | 프로그램이 UTF-8 로 출력/저장합니다. 그래도 깨지면 `chcp 65001` 후 실행 |

---

## 8. 보안 주의사항 (API 키 유출 방지)

- **키를 코드·README·로그·결과 파일에 직접 쓰지 않습니다.** 이 프로젝트의 어떤 파일에도 실제 키는 없습니다.
- `.env` 는 `.gitignore` 로 커밋이 막혀 있습니다. 제출 전 `git status` 에 `.env` 가 보이지 않는지 확인하세요.
- 공유용은 값이 빈 `.env.example` 만 제출합니다.
- 프로그램은 에러 메시지·LLM 원문을 저장하기 전에 **키 값을 `***` 로 마스킹** 합니다
  (OpenAI 401 응답에는 키 일부가 포함되기 때문).
- 스크린샷·화면 공유 시 `.env` 를 열어 둔 창이 보이지 않게 주의하세요.
- 키가 노출됐다면 즉시 콘솔에서 **폐기(Revoke) 후 재발급** 하세요. 코드는 수정할 필요 없이 `.env` 만 바꾸면 됩니다.
- 제출 전 확인 명령:
  ```bash
  grep -rnE "sk-[A-Za-z0-9]|AIza[0-9A-Za-z]" --exclude-dir=.venv --exclude=.env .   # 아무것도 안 나와야 정상
  ```

---

## 9. 키 없이 동작 검증하기 (테스트)

실제 API 대신 가짜 응답을 넣어 성공/401/0건/파싱 재시도/키 미설정/캐시/보너스 기능 등 24개 시나리오를 검증합니다.

```bash
python -m unittest discover -s tests -v
```

---

## 10. 과제 요구사항 대응표

| 요구사항 | 구현 위치 (`travel_planner.py`) |
|---|---|
| argparse CLI, `-date` 필수, 형식 검증 | `build_parser()`, `valid_date()` |
| LLM 택1 (OpenAI/Gemini) | `LLMClient._openai()`, `LLMClient._gemini()` |
| 1차 JSON 스키마 & 파싱 | `build_recommend_prompt()`, `extract_json()`, `validate_recommendation()` |
| 파싱 실패 재시도 1회 | `step_recommend()`, `build_recommend_retry_prompt()` |
| 장소 검색 택1 (Kakao/Naver), 최소 필드 | `PlaceSearcher._kakao()`, `PlaceSearcher._naver()` |
| 0건/실패 시 "데이터 없음"으로 진행 | `step_places()` |
| 최종 리포트 Markdown | `build_report_prompt()`, `assemble_report()` |
| errors 목록 관리 | `add_error()`, `render_errors()` |
| 키 미설정 즉시 종료 | `read_keys()`, `key_setup_help()` |
| 키 관리(.env) & 마스킹 | `load_dotenv`, `redact()` |
| 결과 저장 | `save_json()`, `write_text()` |
| 보너스: 복수 지역 | `--cities`, `recommended_cities` |
| 보너스: 캐싱 | `load_cache()`, `--refresh`, `--regen-report` |

