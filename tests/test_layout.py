"""Thứ tự đọc với layout nhiều cột (toạ độ box thật từ ảnh vỏ hộp IMG_1036)."""
import numpy as np

from ocr_baseline.ocr.local.layout import sort_reading_order
from ocr_baseline.ocr.types import OcrLine, OcrResult


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


# (text, box) đúng như PaddleOCR detection + VietOCR trả về
BOXES = [
    ("Bảo quản / Storage:", rect(118, 1221, 448, 1267)),
    ("Thành phần:", rect(882, 1225, 1102, 1270)),
    ("INDICATION, CONTRAINDICATIONS,", rect(1603, 1248, 2152, 1280)),
    ("Nơi khô, nhiệt độ dưới 30°C.", rect(118, 1264, 579, 1306)),
    ("Mỗi viên nén bao phim chứa:", rect(882, 1277, 1376, 1329)),
    ("Store in a dry place, below 30°C.", rect(122, 1300, 655, 1349)),
    ("USAGE, DOSAGE", rect(1603, 1313, 1879, 1345)),
    ("Để xa tầm tay trẻ em", rect(125, 1339, 474, 1388)),
    ("Acetylspiramycin......100000 UI", rect(889, 1339, 1415, 1381)),
    ("Keep out of reach of children", rect(128, 1378, 599, 1427)),
    ("125mg", rect(1288, 1383, 1420, 1444)),
    ("AND OTHER INFORMATION:", rect(1596, 1371, 2037, 1414)),
    ("Đọc kỹ hướng dẫn sử dụng trước khi dùng", rect(135, 1410, 820, 1476)),
    ("Metronidazol", rect(892, 1394, 1109, 1430)),
    ("Tá dược vừa đủ", rect(892, 1443, 1158, 1489)),
    (".1 viên", rect(1297, 1449, 1415, 1485)),
    ("See the package insert inside", rect(1596, 1440, 2060, 1482)),
]


def _ocr_text(boxes):
    text_of = {id(b): t for t, b in boxes}
    lines = sort_reading_order([b for _, b in boxes])
    return OcrResult([OcrLine(text_of[id(b)], 1.0, b.tolist(), line_id=i)
                      for i, line in enumerate(lines) for b in line]).text


def test_columns_are_not_interleaved_and_dotted_lines_rejoin():
    text = _ocr_text(BOXES)
    assert text.split("\n") == [
        "Bảo quản / Storage:",
        "Nơi khô, nhiệt độ dưới 30°C.",
        "Store in a dry place, below 30°C.",
        "Để xa tầm tay trẻ em",
        "Keep out of reach of children",
        "Đọc kỹ hướng dẫn sử dụng trước khi dùng",
        "Thành phần:",
        "Mỗi viên nén bao phim chứa:",
        "Acetylspiramycin......100000 UI",
        "Metronidazol 125mg",
        "Tá dược vừa đủ .1 viên",
        "INDICATION, CONTRAINDICATIONS,",
        "USAGE, DOSAGE",
        "AND OTHER INFORMATION:",
        "See the package insert inside",
    ]


def test_full_width_title_above_columns_comes_first():
    boxes = [
        ("TIÊU ĐỀ RỘNG", rect(100, 0, 1500, 50)),
        ("cột trái 1", rect(100, 100, 600, 140)),
        ("cột phải 1", rect(900, 120, 1500, 160)),
        ("cột trái 2", rect(100, 160, 600, 200)),
        ("cột phải 2", rect(900, 180, 1500, 220)),
        ("cột trái 3", rect(100, 220, 600, 260)),
    ]
    assert _ocr_text(boxes).split("\n") == [
        "TIÊU ĐỀ RỘNG", "cột trái 1", "cột trái 2", "cột trái 3", "cột phải 1", "cột phải 2",
    ]


def test_table_without_dots_reads_row_by_row():
    boxes = [
        ("Thành phần:", rect(100, 0, 400, 40)),
        ("Paracetamol", rect(100, 60, 400, 100)),
        ("500 mg", rect(700, 62, 850, 102)),
        ("Cafein", rect(100, 120, 300, 160)),
        ("65 mg", rect(700, 118, 830, 158)),
    ]
    assert _ocr_text(boxes).split("\n") == ["Thành phần:", "Paracetamol 500 mg", "Cafein 65 mg"]


def test_single_column_keeps_top_down_order():
    boxes = [(f"dòng {i}", rect(100, 50 * i, 900, 50 * i + 40)) for i in range(5)]
    assert _ocr_text(boxes).split("\n") == [f"dòng {i}" for i in range(5)]


def test_big_font_title_next_to_address_column_is_not_a_table():
    # toạ độ thật từ ảnh vỏ hộp IMG_1035: tên thuốc chữ to bên trái, địa chỉ chữ nhỏ bên phải
    boxes = [
        ("Sansvigyl", rect(73, 1400, 417, 1512)),
        ("Acetylspiramycin-Metronidazole", rect(76, 1505, 887, 1575)),
        ("THUỐC SX TẠI: CÔNG TY C.P DƯỢC PHẨM HÀ TÂY", rect(1180, 1407, 1919, 1454)),
        ("(Đạt tiêu chuẩn GMP-WHO)", rect(1209, 1457, 1891, 1499)),
        ("Tổ dân phố số 4 - phường La Khê", rect(1089, 1496, 2014, 1537)),
        ("Phân phối bởi: CÔNG TY CP Y DƯỢC PHÁP ÂU", rect(1073, 1534, 2023, 1582)),
        ("Số 202A Thuỵ Lĩnh", rect(1190, 1578, 1906, 1620)),
        ("No. 202A Thuy Linh", rect(1193, 1617, 1903, 1658)),
        ("EU MEDIPHARM FRANCE", rect(1354, 1655, 1723, 1696)),
    ]
    assert _ocr_text(boxes).split("\n") == [
        "Sansvigyl",
        "Acetylspiramycin-Metronidazole",
        "THUỐC SX TẠI: CÔNG TY C.P DƯỢC PHẨM HÀ TÂY",
        "(Đạt tiêu chuẩn GMP-WHO)",
        "Tổ dân phố số 4 - phường La Khê",
        "Phân phối bởi: CÔNG TY CP Y DƯỢC PHÁP ÂU",
        "Số 202A Thuỵ Lĩnh",
        "No. 202A Thuy Linh",
        "EU MEDIPHARM FRANCE",
    ]
