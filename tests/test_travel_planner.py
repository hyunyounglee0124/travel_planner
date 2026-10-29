# -*- coding: utf-8 -*-
"""
실제 API 키 없이 전체 흐름을 검증하는 테스트 (requests 호출을 가짜 응답으로 대체).

실행:  python -m unittest discover -s tests -v
"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import travel_planner as tp  # noqa: E402

REC_JSON = {
    "recommended_city": "제주",
    "weather": "3월 중순 평균 10~15°C, 바람이 강한 날이 있습니다.",
    "events": ["제주 들불축제(일정 변동 가능)", "유채꽃 명소 탐방"],
    "reason": "봄꽃이 피기 시작하는 시기입니다. 성수기 대비 비용 부담이 적습니다.",
}
REC_MULTI = dict(REC_JSON, recommended_cities=[
    {"city": "제주", "weather": "온화", "events": ["유채꽃"], "reason": "봄꽃. 저렴."},
    {"city": "경주시", "weather": "맑음", "events": ["벚꽃 마라톤(일정 변동 가능)"], "reason": "벚꽃 시작. 역사 유적."},
])
REPORT_MD = """# 제목은 버려짐
## 추천 지역
**제주** - 봄꽃이 피는 섬
## 추천 이유
유채꽃이 피기 시작합니다. 비용 부담도 적습니다.
## 날씨 요약
평균 10~15°C, 바람막이를 챙기세요.
## 행사/축제
- 제주 들불축제(일정 변동 가능)
## 맛집 추천
현지인이 찾는 음식점을 모았습니다.
| 지어낸 | 표 |
## 1일 일정 제안
- **오전**: 성산일출봉
- **점심**: 돔베고기집
- **오후**: 유채꽃밭
- **저녁**: 해녀의부엌
"""
KAKAO_DOCS = [
    {"place_name": f"맛집{i}", "road_address_name": f"제주특별자치도 제주시 도로 {i}",
     "address_name": f"제주 지번 {i}", "category_name": "음식점 > 한식 > 해물,생선",
     "place_url": f"http://place.map.kakao.com/{i}", "phone": "064-000-0000",
     "x": "126.53", "y": "33.49"} for i in range(1, 6)
]


class FakeResp:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body, ensure_ascii=False)

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def openai_resp(content):
    return FakeResp(200, {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})


class Router:
    """URL 별로 미리 준비한 응답을 순서대로 돌려주는 가짜 requests.request"""

    def __init__(self, llm_responses, place_responses=None):
        self.llm = list(llm_responses)
        self.place = list(place_responses or [])
        self.calls = []

    def __call__(self, method, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "params": params, "json": json})
        if "openai.com" in url or "generativelanguage" in url:
            r = self.llm.pop(0)
        else:
            r = self.place.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


ENV = {"OPENAI_API_KEY": "sk-test-1234567890abcdef", "KAKAO_REST_API_KEY": "kakao-test-key-123",
       "LLM_PROVIDER": "openai", "PLACE_PROVIDER": "kakao"}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, ENV, clear=True)
        self.env.start()
        self.nodotenv = mock.patch.object(tp, "load_dotenv", None)
        self.nodotenv.start()
        self.nosleep = mock.patch.object(tp.time, "sleep", lambda s: None)
        self.nosleep.start()

    def tearDown(self):
        self.nosleep.stop()
        self.nodotenv.stop()
        self.env.stop()
        self.tmp.cleanup()

    def run_main(self, router, extra=None, date="2026-03-15"):
        buf = io.StringIO()
        with mock.patch.object(tp.requests, "request", router), redirect_stdout(buf), redirect_stderr(buf):
            code = tp.main(["--date", date, "--out-dir", str(self.out)] + (extra or []))
        return code, buf.getvalue()

    def outputs(self, date="2026-03-15"):
        raw = json.loads((self.out / f"{date}_raw.json").read_text(encoding="utf-8"))
        md = (self.out / f"{date}_travel_plan.md").read_text(encoding="utf-8")
        return raw, md


class TestCLI(Base):
    def test_invalid_dates_print_usage_and_exit(self):
        for bad in ["2026/03/15", "2026-3-5", "2026-02-30", "abc", ""]:
            buf = io.StringIO()
            with redirect_stderr(buf), self.assertRaises(SystemExit) as cm:
                tp.main(["--date", bad])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("usage:", buf.getvalue())

    def test_single_dash_date_option(self):
        args = tp.build_parser().parse_args(["-date", "2026-03-15"])
        self.assertEqual(args.date.isoformat(), "2026-03-15")

    def test_missing_date_is_error(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tp.main([])

    def test_missing_api_key_exits_immediately(self):
        os.environ.pop("KAKAO_REST_API_KEY")
        router = Router([])
        code, out = self.run_main(router)
        self.assertEqual(code, 1)
        self.assertIn("KAKAO_REST_API_KEY", out)
        self.assertIn("export", out)
        self.assertEqual(router.calls, [])  # API 를 한 번도 호출하지 않음


class TestParsing(unittest.TestCase):
    def test_extract_json_variants(self):
        self.assertEqual(tp.extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(tp.extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(tp.extract_json('추천합니다:\n{"a": 1}\n이상입니다.'), {"a": 1})
        with self.assertRaises(json.JSONDecodeError):
            tp.extract_json("JSON 없음")

    def test_validate_schema(self):
        rec = tp.validate_recommendation(dict(REC_JSON, recommended_city="제주특별자치도",
                                              events="한 개짜리 문자열"))
        self.assertEqual(rec["recommended_city"], "제주")
        self.assertEqual(rec["events"], ["한 개짜리 문자열"])
        with self.assertRaises(tp.SchemaError) as cm:
            tp.validate_recommendation({"recommended_city": "제주", "events": []})
        self.assertGreaterEqual(len(cm.exception.problems), 3)
        too_many = tp.validate_recommendation(dict(REC_JSON, events=list("abcde")))
        self.assertEqual(len(too_many["events"]), 3)

    def test_city_normalization(self):
        cases = {"강원도 강릉시": "강릉", "부산광역시": "부산", "여수시": "여수", "양양군": "양양", "제주": "제주"}
        for src, want in cases.items():
            self.assertEqual(tp._normalize_city(src), want)

    def test_redact(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-secret-abcdefgh12345"}):
            self.assertNotIn("secret", tp.redact("Incorrect API key provided: sk-secret-abcdefgh12345"))
        self.assertNotIn("AIzaSy", tp.redact("key=AIzaSyA1234567890abcdefghijklmn"))


class TestFlows(Base):
    def test_success_flow(self):
        router = Router([openai_resp(json.dumps(REC_JSON, ensure_ascii=False)), openai_resp(REPORT_MD)],
                        [FakeResp(200, {"documents": KAKAO_DOCS, "meta": {"total_count": 5}})])
        code, out = self.run_main(router)
        self.assertEqual(code, 0, out)
        for s in ["[1/3]", "[2/3]", "[3/3]", 'recommended_city: "제주"', "맛집 5곳 검색 완료", "완료!"]:
            self.assertIn(s, out)
        raw, md = self.outputs()
        self.assertEqual(raw["recommendation"]["recommended_city"], "제주")
        self.assertEqual(len(raw["restaurants"]), 5)
        self.assertEqual(raw["errors"], [])
        r0 = raw["restaurants"][0]
        for k in ("name", "address", "category", "url", "x", "y", "lat", "lng"):
            self.assertIn(k, r0)
        self.assertIsInstance(r0["lat"], float)
        # 리포트 섹션 순서 확인
        titles = [t for _, t in tp.SECTION_ORDER]
        positions = [md.index(f"## {t}") for t in titles]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(md.startswith("# 2026-03-15 국내 여행 추천 리포트"))
        self.assertIn("[맛집1](http://place.map.kakao.com/1)", md)
        self.assertNotIn("지어낸", md)               # LLM 이 만든 가짜 표는 제거
        self.assertIn("- **오전**", md)
        self.assertIn("없음 (모든 단계", md)
        # 요청 구조: LLM=POST + Bearer, Kakao=GET + KakaoAK + FD6
        self.assertEqual(router.calls[0]["method"], "POST")
        self.assertTrue(router.calls[0]["headers"]["Authorization"].startswith("Bearer "))
        self.assertEqual(router.calls[0]["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(router.calls[1]["method"], "GET")
        self.assertEqual(router.calls[1]["headers"]["Authorization"], "KakaoAK kakao-test-key-123")
        self.assertEqual(router.calls[1]["params"]["category_group_code"], "FD6")
        self.assertEqual(router.calls[1]["params"]["query"], "제주 맛집")
        # 키가 결과물에 남지 않았는지
        blob = json.dumps(raw, ensure_ascii=False) + md + out
        self.assertNotIn(ENV["OPENAI_API_KEY"], blob)
        self.assertNotIn(ENV["KAKAO_REST_API_KEY"], blob)

    def test_json_parse_retry_once_then_success(self):
        router = Router([openai_resp("죄송하지만 JSON 이 아닙니다"),
                         openai_resp(json.dumps(REC_JSON, ensure_ascii=False)), openai_resp(REPORT_MD)],
                        [FakeResp(200, {"documents": KAKAO_DOCS})])
        code, out = self.run_main(router)
        self.assertEqual(code, 0, out)
        self.assertIn("재시도 1회", out)
        retry_prompt = router.calls[1]["json"]["messages"][1]["content"]
        self.assertIn("필수 키만", retry_prompt)
        raw, _ = self.outputs()
        self.assertEqual([e["type"] for e in raw["errors"]], ["PARSE_ERROR"])

    def test_json_parse_fails_twice_stops_without_infinite_retry(self):
        router = Router([openai_resp("nope"), openai_resp('{"recommended_city": ""}')])
        code, out = self.run_main(router)
        self.assertEqual(code, 1)
        llm_calls = [c for c in router.calls if "openai" in c["url"]]
        self.assertEqual(len(llm_calls), 2)  # 최초 1회 + 재시도 1회 = 2회
        raw = json.loads((self.out / "2026-03-15_raw.json").read_text(encoding="utf-8"))
        self.assertIsNone(raw["recommendation"])
        self.assertEqual(len(raw["errors"]), 2)

    def test_place_auth_error_continues(self):
        router = Router([openai_resp(json.dumps(REC_JSON, ensure_ascii=False)), openai_resp(REPORT_MD)],
                        [FakeResp(401, {"errorType": "AccessDeniedError", "message": "wrong appKey"})])
        code, out = self.run_main(router)
        self.assertEqual(code, 0, out)
        self.assertIn("인증 실패(401)", out)
        self.assertIn("데이터 없음", out)
        raw, md = self.outputs()
        self.assertEqual(raw["restaurants"], [])
        self.assertEqual(raw["errors"][0]["step"], "place_search")
        self.assertEqual(raw["errors"][0]["type"], "AUTH_ERROR")
        self.assertIn("HTTP 401", raw["errors"][0]["message"])
        self.assertIn("- 데이터 없음 (장소 검색 API 인증 실패(HTTP 401))", md)
        self.assertIn("| place_search | AUTH_ERROR |", md)

    def test_place_empty_result(self):
        router = Router([openai_resp(json.dumps(REC_JSON, ensure_ascii=False)), openai_resp(REPORT_MD)],
                        [FakeResp(200, {"documents": [], "meta": {"total_count": 0}})])
        code, out = self.run_main(router)
        self.assertEqual(code, 0)
        self.assertIn("0건", out)
        raw, md = self.outputs()
        self.assertEqual(raw["errors"][0]["type"], "EMPTY_RESULT")
        self.assertIn("0 results for query=제주 맛집", raw["errors"][0]["message"])
        self.assertIn("- 데이터 없음 (장소 검색 결과 0건)", md)

    def test_place_network_error_continues(self):
        router = Router([openai_resp(json.dumps(REC_JSON, ensure_ascii=False)), openai_resp(REPORT_MD)],
                        [tp.requests.exceptions.ConnectionError("down")])
        code, _ = self.run_main(router)
        self.assertEqual(code, 0)
        raw, _ = self.outputs()
        self.assertEqual(raw["errors"][0]["type"], "NETWORK_ERROR")

    def test_place_quota_error(self):
        router = Router([openai_resp(json.dumps(REC_JSON, ensure_ascii=False)), openai_resp(REPORT_MD)],
                        [FakeResp(429, {"message": "quota exceeded"})])
        code, out = self.run_main(router)
        self.assertEqual(code, 0)
        self.assertIn("쿼터 초과", out)

    def test_llm_auth_error_on_step1(self):
        router = Router([FakeResp(401, {"error": {"message": "Incorrect API key provided: sk-test-1234567890abcdef"}})])
        code, out = self.run_main(router)
        self.assertEqual(code, 1)
        self.assertNotIn(ENV["OPENAI_API_KEY"], out)
        raw = json.loads((self.out / "2026-03-15_raw.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["errors"][0]["type"], "AUTH_ERROR")
        self.assertNotIn(ENV["OPENAI_API_KEY"], json.dumps(raw))

    def test_report_llm_failure_uses_fallback(self):
        router = Router([openai_resp(json.dumps(REC_JSON, ensure_ascii=False)),
                         FakeResp(500, {"error": {"message": "server"}}), FakeResp(503, {"error": {"message": "x"}})],
                        [FakeResp(200, {"documents": KAKAO_DOCS})])
        code, out = self.run_main(router)
        self.assertEqual(code, 0, out)
        raw, md = self.outputs()
        self.assertEqual(raw["errors"][0]["step"], "report")
        self.assertEqual(raw["errors"][0]["type"], "SERVER_ERROR")
        self.assertIn("- **점심**: 맛집1", md)
        self.assertIn("기본 일정으로 대체", md)

    def test_report_missing_section_is_filled(self):
        partial = "## 추천 지역\n제주\n## 1일 일정 제안\n- **오전**: 산책"
        router = Router([openai_resp(json.dumps(REC_JSON, ensure_ascii=False)), openai_resp(partial)],
                        [FakeResp(200, {"documents": KAKAO_DOCS})])
        code, _ = self.run_main(router)
        self.assertEqual(code, 0)
        raw, md = self.outputs()
        self.assertEqual(raw["errors"][0]["type"], "MISSING_SECTION")
        self.assertIn(REC_JSON["reason"], md)
        self.assertIn("MISSING_SECTION", md)

    def test_cache_skips_api(self):
        router = Router([openai_resp(json.dumps(REC_JSON, ensure_ascii=False)), openai_resp(REPORT_MD)],
                        [FakeResp(200, {"documents": KAKAO_DOCS})])
        self.run_main(router)
        router2 = Router([])
        code, out = self.run_main(router2)
        self.assertEqual(code, 0)
        self.assertEqual(router2.calls, [])
        self.assertIn("[캐시]", out)
        # 캐시 모드에서는 키가 없어도 동작
        os.environ.clear()
        code, _ = self.run_main(Router([]))
        self.assertEqual(code, 0)

    def test_regen_report_uses_only_llm(self):
        router = Router([openai_resp(json.dumps(REC_JSON, ensure_ascii=False)), openai_resp(REPORT_MD)],
                        [FakeResp(200, {"documents": KAKAO_DOCS})])
        self.run_main(router)
        router2 = Router([openai_resp(REPORT_MD)])
        code, _ = self.run_main(router2, ["--regen-report"])
        self.assertEqual(code, 0)
        self.assertEqual(len(router2.calls), 1)
        self.assertIn("openai", router2.calls[0]["url"])

    def test_multi_city_bonus(self):
        router = Router([openai_resp(json.dumps(REC_MULTI, ensure_ascii=False)), openai_resp(REPORT_MD)],
                        [FakeResp(200, {"documents": KAKAO_DOCS}), FakeResp(200, {"documents": []})])
        code, out = self.run_main(router, ["--cities", "2"])
        self.assertEqual(code, 0, out)
        raw, md = self.outputs()
        self.assertEqual(list(raw["restaurants_by_city"].keys()), ["제주", "경주"])
        self.assertEqual(router.calls[2]["params"]["query"], "경주 맛집")
        self.assertIn("### 제주", md)
        self.assertIn("### 경주", md)
        self.assertIn("recommended_cities", router.calls[0]["json"]["messages"][1]["content"])


class TestProviders(Base):
    def test_gemini_flow(self):
        os.environ.update({"GEMINI_API_KEY": "AIzaTestKey_abcdefghijklmnop"})

        def gem(text):
            return FakeResp(200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]})
        router = Router([gem("```json\n" + json.dumps(REC_JSON, ensure_ascii=False) + "\n```"), gem(REPORT_MD)],
                        [FakeResp(200, {"documents": KAKAO_DOCS})])
        code, out = self.run_main(router, ["--llm", "gemini"])
        self.assertEqual(code, 0, out)
        c = router.calls[0]
        self.assertIn(":generateContent", c["url"])
        self.assertEqual(c["headers"]["x-goog-api-key"], "AIzaTestKey_abcdefghijklmnop")
        self.assertEqual(c["json"]["generationConfig"]["responseMimeType"], "application/json")

    def test_gemini_invalid_key_400_is_auth_error(self):
        os.environ.update({"GEMINI_API_KEY": "AIzaTestKey_abcdefghijklmnop"})
        router = Router([FakeResp(400, {"error": {"message": "API key not valid. Please pass a valid API key."}})])
        code, _ = self.run_main(router, ["--llm", "gemini"])
        self.assertEqual(code, 1)
        raw = json.loads((self.out / "2026-03-15_raw.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["errors"][0]["type"], "AUTH_ERROR")

    def test_naver_flow(self):
        os.environ.update({"NAVER_CLIENT_ID": "naver-id-123", "NAVER_CLIENT_SECRET": "naver-secret-456"})
        items = [{"title": "<b>제주</b> 고기국수&amp;집", "link": "", "category": "한식>국수",
                  "address": "제주시 지번", "roadAddress": "제주시 도로 1", "mapx": "1265312345", "mapy": "334987654"}]
        router = Router([openai_resp(json.dumps(REC_JSON, ensure_ascii=False)), openai_resp(REPORT_MD)],
                        [FakeResp(200, {"items": items})])
        code, out = self.run_main(router, ["--place", "naver"])
        self.assertEqual(code, 0, out)
        raw, _ = self.outputs()
        r = raw["restaurants"][0]
        self.assertEqual(r["name"], "제주 고기국수&집")
        self.assertAlmostEqual(r["lng"], 126.5312345)
        self.assertAlmostEqual(r["lat"], 33.4987654)
        self.assertTrue(r["url"].startswith("https://map.naver.com/"))
        h = router.calls[1]["headers"]
        self.assertEqual(h["X-Naver-Client-Id"], "naver-id-123")
        self.assertEqual(router.calls[1]["params"]["display"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
