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
