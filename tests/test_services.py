import datetime as dt

import httpx

from prognosis.services import advice, trends


def test_rules_only_mention_values_that_were_entered():
    t = advice.rules_summary("diabetes", "high", 0.8, {"HbA1c_level": 7.2})
    assert "6.5" in t and "BMI" not in t and "80%" in t and "over 99%" in advice.rules_summary("heart", "high", 0.9996, {})


def test_rules_findings_cutoffs():
    assert any("prediabetes" in s for s in advice.findings("diabetes", {"HbA1c_level": 6.0}))
    assert advice.findings("diabetes", {"HbA1c_level": 5.2}) == []
    assert any("creatinine" in s for s in advice.findings("ckd", {"sc": 4.0}))


def test_llm_summary_sends_no_identity_and_parses_reply():
    seen = {}

    def handler(req: httpx.Request):
        seen["body"] = req.content.decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": " Short summary. "}}]})

    out = advice.llm_summary("k", "m", "heart", "high", 0.7, {"age": 60.0}, transport=httpx.MockTransport(handler))
    assert out == "Short summary."
    # the function never receives a name or MRN, so the only patient data in the request is the value line
    assert "Values: age=60.0" in seen["body"] and "mrn" not in seen["body"].lower()


def test_llm_failure_returns_none():
    t = httpx.MockTransport(lambda r: httpx.Response(500))
    assert advice.llm_summary("k", "m", "heart", "high", 0.7, {}, transport=t) is None
    assert advice.llm_summary("", "m", "heart", "high", 0.7, {}) is None


def test_trend_summary():
    d0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    pts = [{"at": d0 + dt.timedelta(days=30 * i), "probability": p, "band": b}
           for i, (p, b) in enumerate([(0.2, "low"), (0.35, "moderate"), (0.5, "moderate")])]
    s = trends.summarise(pts)
    assert s["direction"] == "rising" and s["band_change"] == "low to moderate" and s["slope_per_month"] > 0.1
    assert trends.summarise(pts[:1])["delta"] is None
    flat = [dict(pts[0]), dict(pts[0], at=d0 + dt.timedelta(days=5), probability=0.22)]
    assert trends.summarise(flat)["direction"] == "stable"
