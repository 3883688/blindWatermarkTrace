import base64

import pytest

from password_security import hash_password, verify_password


def test_hash_is_salted_versioned_and_not_plaintext() -> None:
    first = hash_password("same-password")
    second = hash_password("same-password")

    assert first.startswith("scrypt$v1$")
    assert first != second
    assert "same-password" not in first


def test_verify_accepts_only_the_original_password() -> None:
    encoded = hash_password("correct-password")

    assert verify_password("correct-password", encoded) is True
    assert verify_password("wrong-password", encoded) is False
    assert verify_password("correct-password", "invalid") is False


def test_hash_rejects_empty_password() -> None:
    try:
        hash_password("")
    except ValueError as exc:
        assert str(exc) == "password must not be empty"
    else:
        raise AssertionError("empty password must be rejected")


def test_verify_rejects_untrusted_scrypt_parameters_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = "$".join(
        (
            "scrypt",
            "v1",
            str(2**20),
            "8",
            "1",
            base64.b64encode(b"s" * 16).decode("ascii"),
            base64.b64encode(b"d" * 32).decode("ascii"),
        )
    )
    monkeypatch.setattr(
        "password_security.hashlib.scrypt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("untrusted parameters reached scrypt")
        ),
    )

    assert verify_password("password", encoded) is False
