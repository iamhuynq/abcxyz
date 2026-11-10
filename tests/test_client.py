import gzip
from datetime import UTC, datetime

import httpx
import pytest

from fireant_crawler.client.fireant_client import (
    ApiError,
    AuthError,
    FireAntClient,
    ForbiddenRequestError,
)
from fireant_crawler.client.rate_limiter import RateLimiter
from fireant_crawler.store.raw_store import RawStore

TOKEN = "fake-token-1234567890"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(handler, sleeps=None, max_retries=3):
    clock = FakeClock()
    return FireAntClient(
        token=TOKEN,
        rate_limiter=RateLimiter(1000, clock=clock.clock, sleep=clock.sleep),
        max_retries=max_retries,
        transport=httpx.MockTransport(handler),
        sleep=(sleeps.append if sleeps is not None else lambda _s: None),
    )


def test_rate_limiter_spaces_requests():
    clock = FakeClock()
    limiter = RateLimiter(2.5, clock=clock.clock, sleep=clock.sleep)
    for _ in range(3):
        limiter.acquire()
    assert clock.sleeps == pytest.approx([0.4, 0.4])


def test_sends_bearer_token_and_returns_json():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=[{"date": "2026-09-04T00:00:00"}], headers={"X-RateLimit-Remaining": "99"})

    with make_client(handler) as client:
        res = client.get("/symbols/VOS/historical-quotes", {"startDate": "2026-09-01", "limit": 20, "offset": None})
    assert seen["auth"] == f"Bearer {TOKEN}"
    assert seen["params"] == {"startDate": "2026-09-01", "limit": "20"}
    assert res.data == [{"date": "2026-09-04T00:00:00"}]
    assert res.rate_limit_headers == {"x-ratelimit-remaining": "99"}


def test_blocks_non_whitelisted_path_without_network_call():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be called")

    with make_client(handler) as client, pytest.raises(ForbiddenRequestError):
        client.get("/broker-users/1/accounts/2/orderHistory")


def test_auth_error_stops_immediately_without_retry():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(401, json={"message": "Authorization has been denied for this request."})

    with make_client(handler) as client, pytest.raises(AuthError) as exc:
        client.get("/symbols/VOS")
    assert len(calls) == 1
    assert TOKEN not in str(exc.value)


def test_retries_429_honouring_retry_after_then_succeeds():
    responses = iter([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(503), httpx.Response(200, json={})])
    sleeps: list[float] = []

    with make_client(lambda _r: next(responses), sleeps=sleeps) as client:
        res = client.get("/symbols/VOS")
    assert res.attempts == 3
    assert sleeps[0] == 7.0
    assert 1.0 <= sleeps[1] <= 3.0


def test_gives_up_after_max_retries():
    with make_client(lambda _r: httpx.Response(500), max_retries=3) as client, pytest.raises(ApiError) as exc:
        client.get("/symbols/VOS")
    assert "3 attempts" in str(exc.value)


def test_error_body_is_redacted():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=f"bad request for Bearer {TOKEN}")

    with make_client(handler) as client, pytest.raises(ApiError) as exc:
        client.get("/symbols/VOS")
    assert TOKEN not in str(exc.value)


def test_raw_store_writes_envelope_without_token(tmp_path):
    with make_client(lambda _r: httpx.Response(200, json=[{"a": 1}])) as client:
        res = client.get("/symbols/VOS/historical-quotes", {"limit": 20})
    store = RawStore(tmp_path)
    path = store.write("quotes", "VOS", "2026", res, datetime(2026, 10, 3, tzinfo=UTC))

    assert path == tmp_path / "raw/quotes/VOS/2026-10-03/2026__000000.json.gz"
    assert store.iter_files("quotes") == [("VOS", path)]
    assert store.read(path)["data"] == [{"a": 1}]
    assert TOKEN not in gzip.decompress(path.read_bytes()).decode()


def test_raw_store_never_overwrites_within_the_same_second(tmp_path):
    """Two fetches in the same second keep both payloads; names sort in write order; no temp file is left."""
    store, at = RawStore(tmp_path), datetime(2026, 10, 3, 9, 30, 15, tzinfo=UTC)
    paths = []
    for value in (1, 2, 3):
        with make_client(lambda _r, v=value: httpx.Response(200, json=[{"a": v}])) as client:
            paths.append(store.write("quotes", "VOS", "2026", client.get("/symbols/VOS/historical-quotes"), at))
    assert [p.name for p in paths] == ["2026__093015.json.gz", "2026__093015_001.json.gz", "2026__093015_002.json.gz"]
    assert [store.read(p)["data"][0]["a"] for p in paths] == [1, 2, 3]
    later = at.replace(second=16)
    with make_client(lambda _r: httpx.Response(200, json=[{"a": 4}])) as client:
        paths.append(store.write("quotes", "VOS", "2026", client.get("/symbols/VOS/historical-quotes"), later))
    assert [p for _, p in store.iter_files("quotes")] == paths            # chronological replay order
    assert not list((tmp_path / "raw").rglob("*.tmp"))


def test_raw_store_rejects_unsafe_names(tmp_path):
    with make_client(lambda _r: httpx.Response(200, json=[])) as client:
        res = client.get("/symbols/VOS")
    with pytest.raises(ValueError):
        RawStore(tmp_path).write("quotes", "../etc", "x", res, datetime(2026, 10, 3, tzinfo=UTC))
