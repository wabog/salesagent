from sales_agent.domain.phones import build_legacy_phone_candidates, normalize_phone_number, phone_to_provider_digits


def test_normalize_phone_number_to_e164_with_default_country():
    assert normalize_phone_number("315 6832405") == "+573156832405"
    assert normalize_phone_number("573156832405") == "+573156832405"
    assert normalize_phone_number("+57 3156832405") == "+573156832405"


def test_phone_to_provider_digits_strips_plus_and_formatting():
    assert phone_to_provider_digits("+57 315 6832405") == "573156832405"


def test_build_legacy_phone_candidates_covers_common_notion_formats():
    assert build_legacy_phone_candidates("+57 315 6832405") == [
        "+573156832405",
        "573156832405",
        "3156832405",
    ]
