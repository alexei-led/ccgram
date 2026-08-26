import base64
import hashlib
import hmac
import json
from urllib.parse import urlencode

import pytest

from ccgram.miniapp.auth import (
    DEFAULT_TOKEN_TTL,
    InvalidTokenError,
    TokenPayload,
    _b64url_encode,
    _signing_key,
    authorize_api_request,
    init_data_user_id,
    sign_token,
    validate_init_data,
    verify_token,
)

from ._helpers import make_init_data

BOT = "1234:abcdef"
WID = "ccgram:@7"
UID = 42
NOW = 1_700_000_000.0


def test_sign_then_verify_roundtrip():
    tok = sign_token(bot_token=BOT, window_id=WID, user_id=UID, now=NOW)
    payload = verify_token(tok, bot_token=BOT, now=NOW + 10)
    assert isinstance(payload, TokenPayload)
    assert payload.window_id == WID
    assert payload.user_id == UID
    assert payload.exp == int(NOW) + DEFAULT_TOKEN_TTL


def test_verify_rejects_expired_token():
    tok = sign_token(bot_token=BOT, window_id=WID, user_id=UID, ttl=60, now=NOW)
    with pytest.raises(InvalidTokenError, match="expired"):
        verify_token(tok, bot_token=BOT, now=NOW + 120)


def test_verify_rejects_wrong_bot_token():
    tok = sign_token(bot_token=BOT, window_id=WID, user_id=UID, now=NOW)
    with pytest.raises(InvalidTokenError, match="signature"):
        verify_token(tok, bot_token="9999:other", now=NOW + 10)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "not.a.token.at.all",
        "no_dot_here",
        "...",
        "!!!.!!!",
    ],
)
def test_verify_rejects_malformed(bad):
    with pytest.raises(InvalidTokenError):
        verify_token(bad, bot_token=BOT, now=NOW)


def test_verify_rejects_tampered_payload():
    tok = sign_token(bot_token=BOT, window_id=WID, user_id=UID, now=NOW)
    body, sig = tok.split(".")
    # Flip a byte in body — sig won't match.
    tampered = body[:-1] + ("A" if body[-1] != "A" else "B") + "." + sig
    with pytest.raises(InvalidTokenError, match="signature"):
        verify_token(tampered, bot_token=BOT, now=NOW + 10)


def test_signing_key_rejects_empty_bot_token():
    with pytest.raises(InvalidTokenError, match="bot_token"):
        sign_token(bot_token="", window_id=WID, user_id=UID, now=NOW)


def _make_init_data(bot_token: str, params: dict[str, str]) -> str:
    """Build a signed initData string per the WebApp legacy HMAC spec."""
    pairs = sorted(params.items())
    data_check = "\n".join(f"{k}={v}" for k, v in pairs)
    secret = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    h = hmac.new(secret, data_check.encode("utf-8"), hashlib.sha256).hexdigest()
    return urlencode({**params, "hash": h})


def test_validate_init_data_happy_path():
    params = {
        "auth_date": str(int(NOW)),
        "user": '{"id":42,"first_name":"Alice"}',
        "query_id": "AAH",
    }
    init = _make_init_data(BOT, params)
    out = validate_init_data(init, bot_token=BOT, now=NOW + 10)
    assert out["query_id"] == "AAH"
    assert out["hash"]


def test_validate_init_data_rejects_bad_signature():
    params = {"auth_date": str(int(NOW)), "user": '{"id":1}'}
    init = _make_init_data(BOT, params)
    with pytest.raises(InvalidTokenError, match="signature"):
        validate_init_data(init, bot_token="other:token", now=NOW + 10)


def test_validate_init_data_rejects_missing_hash():
    init = urlencode({"auth_date": str(int(NOW)), "user": '{"id":1}'})
    with pytest.raises(InvalidTokenError, match="missing hash"):
        validate_init_data(init, bot_token=BOT, now=NOW + 10)


def test_validate_init_data_rejects_stale_auth_date():
    params = {"auth_date": str(int(NOW)), "user": '{"id":1}'}
    init = _make_init_data(BOT, params)
    with pytest.raises(InvalidTokenError, match="stale"):
        validate_init_data(init, bot_token=BOT, max_age=60, now=NOW + 3600)


def test_validate_init_data_rejects_non_numeric_auth_date():
    params = {"auth_date": "not-a-number", "user": '{"id":1}'}
    init = _make_init_data(BOT, params)
    with pytest.raises(InvalidTokenError, match="auth_date"):
        validate_init_data(init, bot_token=BOT, now=NOW + 10)


def test_validate_init_data_rejects_empty():
    with pytest.raises(InvalidTokenError, match="empty"):
        validate_init_data("", bot_token=BOT)


def _sign_body(body: bytes, bot_token: str = BOT) -> str:
    """Mint a correctly-signed token around an arbitrary payload body."""
    sig = hmac.new(_signing_key(bot_token), body, hashlib.sha256).digest()
    return f"{_b64url_encode(body)}.{_b64url_encode(sig)}"


class TestVerifyTokenPayloadValidation:
    """A valid signature is not enough — the body must still decode to a payload."""

    def test_rejects_non_json_body(self):
        with pytest.raises(InvalidTokenError, match="not JSON"):
            verify_token(_sign_body(b"not json at all"), bot_token=BOT, now=NOW)

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"u": UID, "exp": int(NOW) + 60}, id="missing_window_id"),
            pytest.param({"w": WID, "exp": int(NOW) + 60}, id="missing_user_id"),
            pytest.param({"w": WID, "u": UID}, id="missing_exp"),
            pytest.param(
                {"w": WID, "u": "not-an-int", "exp": int(NOW) + 60},
                id="user_id_not_an_int",
            ),
            pytest.param(
                {"w": WID, "u": UID, "exp": "not-an-int"}, id="exp_not_an_int"
            ),
        ],
    )
    def test_rejects_incomplete_payload(self, payload):
        body = json.dumps(payload).encode()
        with pytest.raises(InvalidTokenError, match="missing fields"):
            verify_token(_sign_body(body), bot_token=BOT, now=NOW)

    def test_rejects_body_that_is_not_base64(self):
        sig = base64.urlsafe_b64encode(b"x" * 32).rstrip(b"=").decode()
        with pytest.raises(InvalidTokenError):
            verify_token(f"!!!not-base64!!!.{sig}", bot_token=BOT, now=NOW)


class TestInitDataUserId:
    def test_extracts_id(self):
        assert init_data_user_id({"user": '{"id":42,"first_name":"A"}'}) == 42

    @pytest.mark.parametrize(
        ("params", "match"),
        [
            pytest.param({}, "missing user", id="absent"),
            pytest.param({"user": ""}, "missing user", id="empty"),
            pytest.param({"user": "{not json"}, "malformed", id="not_json"),
            pytest.param({"user": '{"name":"A"}'}, "malformed", id="no_id_field"),
            pytest.param({"user": '{"id":"abc"}'}, "malformed", id="id_not_numeric"),
            pytest.param({"user": '{"id":null}'}, "malformed", id="id_null"),
        ],
    )
    def test_rejects_malformed_user(self, params, match):
        with pytest.raises(InvalidTokenError, match=match):
            init_data_user_id(params)


class TestAuthorizeApiRequest:
    """The bearer token travels in the URL, so initData must independently
    bind the request to the same Telegram user — that match is the URL-leak
    defense and must never be skippable."""

    def test_accepts_matching_token_and_init_data(self):
        tok = sign_token(bot_token=BOT, window_id=WID, user_id=UID, now=NOW)
        init = make_init_data(bot_token=BOT, user_id=UID, auth_date=int(NOW))
        payload = authorize_api_request(
            bot_token=BOT, token=tok, init_data=init, now=NOW + 10
        )
        assert payload.window_id == WID
        assert payload.user_id == UID

    def test_rejects_init_data_for_a_different_user(self):
        tok = sign_token(bot_token=BOT, window_id=WID, user_id=UID, now=NOW)
        init = make_init_data(bot_token=BOT, user_id=UID + 1, auth_date=int(NOW))
        with pytest.raises(InvalidTokenError, match="user mismatch"):
            authorize_api_request(
                bot_token=BOT, token=tok, init_data=init, now=NOW + 10
            )

    @pytest.mark.parametrize("init_data", [None, ""])
    def test_rejects_absent_init_data(self, init_data):
        tok = sign_token(bot_token=BOT, window_id=WID, user_id=UID, now=NOW)
        with pytest.raises(InvalidTokenError, match="missing initData"):
            authorize_api_request(
                bot_token=BOT, token=tok, init_data=init_data, now=NOW + 10
            )

    def test_rejects_bad_token_before_looking_at_init_data(self):
        init = make_init_data(bot_token=BOT, user_id=UID, auth_date=int(NOW))
        with pytest.raises(InvalidTokenError, match="signature"):
            authorize_api_request(
                bot_token=BOT,
                token=sign_token(
                    bot_token="9999:other", window_id=WID, user_id=UID, now=NOW
                ),
                init_data=init,
                now=NOW + 10,
            )

    def test_rejects_init_data_signed_by_another_bot(self):
        tok = sign_token(bot_token=BOT, window_id=WID, user_id=UID, now=NOW)
        init = make_init_data(bot_token="9999:other", user_id=UID, auth_date=int(NOW))
        with pytest.raises(InvalidTokenError, match="signature"):
            authorize_api_request(
                bot_token=BOT, token=tok, init_data=init, now=NOW + 10
            )

    def test_rejects_expired_token_even_with_valid_init_data(self):
        tok = sign_token(bot_token=BOT, window_id=WID, user_id=UID, ttl=60, now=NOW)
        init = make_init_data(bot_token=BOT, user_id=UID, auth_date=int(NOW) + 120)
        with pytest.raises(InvalidTokenError, match="expired"):
            authorize_api_request(
                bot_token=BOT, token=tok, init_data=init, now=NOW + 120
            )
