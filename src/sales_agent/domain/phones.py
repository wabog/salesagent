from __future__ import annotations


def normalize_phone_number(phone_number: str | None, *, default_country_code: str = "57") -> str:
    raw = str(phone_number or "").strip()
    if not raw:
        return ""

    digits = "".join(char for char in raw if char.isdigit())
    if not digits:
        return raw

    if raw.startswith("+"):
        return f"+{digits}"
    if digits.startswith("00") and len(digits) > 2:
        return f"+{digits[2:]}"
    if digits.startswith(default_country_code) and len(digits) > len(default_country_code):
        return f"+{digits}"
    if len(digits) == 10:
        return f"+{default_country_code}{digits}"
    return f"+{digits}" if len(digits) >= 11 else digits


def phone_to_provider_digits(phone_number: str | None, *, default_country_code: str = "57") -> str:
    canonical = normalize_phone_number(phone_number, default_country_code=default_country_code)
    return "".join(char for char in canonical if char.isdigit())


def build_legacy_phone_candidates(phone_number: str | None, *, default_country_code: str = "57") -> list[str]:
    canonical = normalize_phone_number(phone_number, default_country_code=default_country_code)
    digits = "".join(char for char in canonical if char.isdigit())
    candidates: list[str] = []

    def add(candidate: str) -> None:
        value = candidate.strip()
        if value and value not in candidates:
            candidates.append(value)

    add(canonical)
    add(digits)
    if digits.startswith(default_country_code) and len(digits) > len(default_country_code):
        add(digits[len(default_country_code) :])
    return candidates
