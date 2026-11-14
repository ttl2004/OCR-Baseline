import pytest

from ocr_baseline.profiles import PROFILES, get_profile


def test_food_normalize_fixes_types():
    food = get_profile("food")
    raw = {
        "product_name": "  Sữa tươi ",
        "ingredient": "Sữa bò\nĐường",                       # chuỗi thay vì list
        "additive": ["Chất ổn định (471)", "", None],
        "nutrition": [{"name": "Năng lượng", "value": "60 kcal"}, "rác", {"name": "", "value": ""}],
        "origin": None,
        "khong_co_trong_profile": "bỏ qua",
    }
    result = food.normalize(raw)
    assert result["product_name"] == "Sữa tươi"
    assert result["ingredient"] == ["Sữa bò", "Đường"]
    assert result["additive"] == ["Chất ổn định (471)"]
    assert result["nutrition"] == [{"name": "Năng lượng", "value": "60 kcal"}]
    assert result["origin"] == "" and result["warning"] == []
    assert set(result) == set(food.field_keys())


def test_drug_fields_are_text():
    drug = get_profile("drug")
    result = drug.normalize({"ten_thuoc": "Methocarbamol", "chi_dinh": ["Đau cơ", "Co thắt"]})
    assert result["ten_thuoc"] == "Methocarbamol"
    assert result["chi_dinh"] == "Đau cơ\nCo thắt"
    assert len(result) == 16


def test_every_profile_title_field_exists():
    for profile in PROFILES.values():
        assert profile.title_field in profile.field_keys()


def test_unknown_profile():
    with pytest.raises(ValueError):
        get_profile("khong_ton_tai")
