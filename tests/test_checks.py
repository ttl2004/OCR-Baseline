from ocr_baseline.extraction.checks import numbers_in, unverified_numbers


def test_numbers_normalize_decimal_comma():
    assert numbers_in("Đường 6,6 g, natri 15mg") == {"6.6", "15"}


def test_flags_number_not_in_ocr():
    fields = {"lieu": "Uống 2 viên x 3 lần", "ham_luong": "500 mg"}
    assert unverified_numbers(fields, "Uống 2 viên x 3 lần. Paracetamol 250 mg") == {"ham_luong": ["500"]}


def test_code_list_written_together_is_not_flagged():
    # OCR "(434,903)" là 2 mã INS viết liền; LLM tách "434, 903" -> không cảnh báo nhầm
    fields = {"additive": ["Chất làm bóng (434, 903)"]}
    assert unverified_numbers(fields, "chất làm bóng (434,903)") == {}
    # ngược lại: OCR tách, LLM viết liền
    assert unverified_numbers({"additive": "(307c,320,321,330)"}, "(307c, 320, 321,\n330)") == {}


def test_decimal_change_is_still_flagged():
    # "6.6" không bị tách thành "6" và "6" -> đổi số thập phân vẫn bị phát hiện
    assert unverified_numbers({"x": "6,6 g"}, "Đường 6 g, 6 viên") == {"x": ["6.6"]}


def test_list_and_nutrition_values():
    fields = {"nutrition": [{"name": "Natri", "value": "126 mg"}], "warning": []}
    assert unverified_numbers(fields, "Natri 126 mg") == {}
