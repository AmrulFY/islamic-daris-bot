from __future__ import annotations

from datetime import date

import httpx
import pytest

from muslim_bot import hijri
from muslim_bot.hijri import SOURCE_COMPUTED, SOURCE_FALLBACK, SOURCE_TABLE
from tests.test_hijri import TABLE_PAYLOAD, make_calendar, run
from tests.test_hijri_events import EVENTS_PAYLOAD_2027, by_key, routed

DAY = date(2026, 10, 17)


class Flaky:
    """Fails the first `failures` requests with `error`, then answers with `payload`."""

    def __init__(self, error: Exception, failures: int = 1, payload: dict | None = None) -> None:
        self.error = error
        self.failures = failures
        self.payload = payload or TABLE_PAYLOAD
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return httpx.Response(200, json=self.payload)


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(hijri.time, "monotonic", lambda: now[0])
    return now


# --- a connection that stalls is tried once more -------------------------------------------------


@pytest.mark.parametrize("error", [httpx.ConnectTimeout("stalled"), httpx.ConnectError("refused")])
def test_a_connection_that_never_formed_is_retried_once(error):
    flaky = Flaky(error, failures=1)
    calendar, requests = make_calendar(flaky)

    result = run(calendar.convert(DAY))

    assert result.source == SOURCE_TABLE  # the second attempt got the real answer, not the estimate
    assert len(requests) == 2


def test_it_gives_up_after_the_second_failed_connection():
    flaky = Flaky(httpx.ConnectTimeout("stalled"), failures=5)
    calendar, requests = make_calendar(flaky)

    assert run(calendar.convert(DAY)).source == SOURCE_FALLBACK
    assert len(requests) == 2  # not an endless loop


@pytest.mark.parametrize(
    "failure",
    [httpx.ReadTimeout("slow"), httpx.RemoteProtocolError("dropped")],
)
def test_a_request_that_was_sent_is_not_repeated(failure):
    flaky = Flaky(failure, failures=5)
    calendar, requests = make_calendar(flaky)

    assert run(calendar.convert(DAY)).source == SOURCE_FALLBACK
    assert len(requests) == 1


def test_an_http_error_is_not_retried():
    calendar, requests = make_calendar(lambda request: httpx.Response(503))
    assert run(calendar.convert(DAY)).source == SOURCE_FALLBACK
    assert len(requests) == 1


def test_events_ride_out_a_stalled_connection_too():
    flaky = Flaky(httpx.ConnectTimeout("stalled"), failures=1, payload=EVENTS_PAYLOAD_2027)
    calendar, requests = make_calendar(lambda request: flaky(request) if request.url.path.endswith("/events") else routed()(request))

    events = run(calendar.events(2027))

    assert by_key(events)["idul_adha"].source == SOURCE_COMPUTED  # from the API, not the local estimate
    assert flaky.calls == 2


# --- older real data beats a local estimate --------------------------------------------------------


def test_a_stale_answer_is_used_when_the_api_is_unreachable(clock):
    state = {"up": True}

    def handler(request):
        return httpx.Response(200, json=TABLE_PAYLOAD) if state["up"] else httpx.Response(503)

    calendar, requests = make_calendar(handler)
    fresh = run(calendar.convert(DAY))
    assert fresh.source == SOURCE_TABLE

    clock[0] += hijri.CACHE_TTL_SECONDS + 60  # expired, so it is fetched again, but the API is down
    state["up"] = False
    stale = run(calendar.convert(DAY))

    assert stale == fresh and stale.source == SOURCE_TABLE  # Kemenag's date from hours ago, not Umm al-Qura
    again = run(calendar.convert(DAY))  # now in the cooldown: no new request either
    assert again == fresh
    assert len(requests) == 2


def test_the_api_recovering_replaces_the_stale_answer(clock):
    state = {"payload": TABLE_PAYLOAD, "up": True}

    def handler(request):
        return httpx.Response(200, json=state["payload"]) if state["up"] else httpx.Response(503)

    calendar, _ = make_calendar(handler)
    run(calendar.convert(DAY))
    clock[0] += hijri.CACHE_TTL_SECONDS + 60
    state["up"] = False
    run(calendar.convert(DAY))  # served stale
    state["up"] = True
    state["payload"] = {**TABLE_PAYLOAD, "output": {**TABLE_PAYLOAD["output"], "day": 7, "weekday": "Ahad"}}
    clock[0] += hijri.FAILURE_COOLDOWN_SECONDS + 1

    assert run(calendar.convert(DAY)).day == 7  # fresh data wins as soon as the API answers


def test_a_stale_answer_is_not_trusted_forever(clock):
    state = {"up": True}

    def handler(request):
        return httpx.Response(200, json=TABLE_PAYLOAD) if state["up"] else httpx.Response(503)

    calendar, _ = make_calendar(handler)
    run(calendar.convert(DAY))
    state["up"] = False

    clock[0] += hijri.CACHE_TTL_SECONDS + hijri.STALE_MAX_SECONDS - 60
    assert run(calendar.convert(DAY)).source == SOURCE_TABLE  # still within the stale limit
    clock[0] += 120
    calendar._down_until = 0.0
    assert run(calendar.convert(DAY)).source == SOURCE_FALLBACK  # too old: back to the local estimate


def test_stale_holidays_are_used_when_the_api_is_unreachable(clock):
    state = {"up": True}
    healthy = routed()

    def handler(request):
        return healthy(request) if state["up"] else httpx.Response(503)

    calendar, _ = make_calendar(handler)
    fresh = run(calendar.events(2027))
    assert by_key(fresh)["idul_adha"].source == SOURCE_COMPUTED

    clock[0] += hijri.CACHE_TTL_SECONDS + 60
    state["up"] = False
    stale = run(calendar.events(2027))

    assert stale == fresh  # including Isra Miraj and Nisfu Syaban with the API's own dates
    assert all(event.source != SOURCE_FALLBACK for event in stale)


def test_nothing_stale_to_fall_back_on_means_the_local_estimate():
    calendar, _ = make_calendar(lambda request: httpx.Response(503))
    assert run(calendar.convert(DAY)).source == SOURCE_FALLBACK
    assert all(event.source == SOURCE_FALLBACK for event in run(calendar.events(2027)))
