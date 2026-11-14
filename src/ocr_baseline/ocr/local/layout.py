"""Sắp xếp box chữ theo thứ tự đọc (recursive XY-cut), hỗ trợ nhiều cột và bảng."""
import numpy as np

from .geometry import Box, bbox


def sort_reading_order(polys: list[Box]) -> list[list[Box]]:
    """Trả về danh sách dòng hiển thị; mỗi dòng là các box xếp trái -> phải."""
    if not polys:
        return []
    med_h = float(np.median([bbox(p)[3] - bbox(p)[1] for p in polys]))
    return _xy_cut(list(polys), max(med_h, 1.0))


def _gaps(intervals: list[tuple[float, float]], min_width: float) -> list[tuple[float, float]]:
    """Các khoảng trống giữa những đoạn [a, b] (sau khi gộp đoạn chồng nhau)."""
    gaps, end = [], None
    for a, b in sorted(intervals):
        if end is not None and a - end >= min_width:
            gaps.append((end, a))
        end = b if end is None else max(end, b)
    return gaps


def _is_table_split(left: list[Box], right: list[Box], min_overlap: float = 0.8) -> bool:
    """Mọi box ở bên ít box hơn đều có box thẳng hàng ở bên kia -> là bảng
    (vd: "Paracetamol | 500 mg"), nên đọc theo hàng thay vì tách cột.

    Cột văn bản thật hiếm khi có tất cả các dòng thẳng hàng >= 80% với cột bên cạnh.
    Ô cùng hàng của bảng có cỡ chữ gần bằng nhau, và bên nhiều box hơn không thừa
    nhiều dòng lẻ (cho phép thừa một dòng tiêu đề như "Thành phần:").
    """
    small, big = (left, right) if len(left) <= len(right) else (right, left)
    if len(small) < 2:
        return False

    def aligned(a: Box, b: Box) -> bool:
        _, ay0, _, ay1 = bbox(a)
        _, by0, _, by1 = bbox(b)
        ha, hb = ay1 - ay0, by1 - by0
        if max(ha, hb) > 2.0 * min(ha, hb):  # tiêu đề chữ to cạnh chữ nhỏ -> không phải bảng
            return False
        return min(ay1, by1) - max(ay0, by0) >= min_overlap * min(ha, hb)

    matched: set[int] = set()
    for a in small:
        partners = [i for i, b in enumerate(big) if aligned(a, b)]
        if not partners:
            return False
        matched.update(partners)
    return len(big) - len(matched) <= 1


def _xy_cut(polys: list[Box], med_h: float) -> list[list[Box]]:
    if len(polys) == 1:
        return [polys]
    boxes = [bbox(p) for p in polys]

    def by_x(ps: list[Box]) -> list[Box]:
        return sorted(ps, key=lambda p: bbox(p)[0])

    # co box theo chiều dọc 20% mỗi phía để các dòng sát nhau (chồng 1-2px) vẫn tách được
    y_iv = [(y0 + 0.2 * (y1 - y0), y1 - 0.2 * (y1 - y0)) for _, y0, _, y1 in boxes]
    x_iv = [(x0, x1) for x0, _, x1, _ in boxes]
    y_gaps = _gaps(y_iv, 0.0)
    x_gaps = _gaps(x_iv, 0.3 * med_h)

    height = max(b[3] for b in boxes) - min(b[1] for b in boxes)
    max_h = max(b[3] - b[1] for b in boxes)
    if not y_gaps and height <= 1.6 * max_h:
        return [by_x(polys)]  # cùng một dòng hiển thị

    x_split = None
    if x_gaps:
        gx = max(x_gaps, key=lambda g: g[1] - g[0])
        left = [p for p, b in zip(polys, boxes) if b[2] <= gx[0]]
        right = [p for p, b in zip(polys, boxes) if b[0] >= gx[1]]
        x_split = (gx[1] - gx[0], left, right)

    gy = max(y_gaps, key=lambda g: g[1] - g[0]) if y_gaps else None
    use_x = x_split is not None and (
        gy is None or (x_split[0] >= gy[1] - gy[0] and not _is_table_split(x_split[1], x_split[2]))
    )

    if use_x:
        parts = [x_split[1], x_split[2]]
    elif gy is not None:
        parts = [[p for p, iv in zip(polys, y_iv) if iv[1] <= gy[0]],
                 [p for p, iv in zip(polys, y_iv) if iv[0] >= gy[1]]]
    else:  # các box chồng chéo, không cắt được: mỗi box một dòng theo y
        return [[p] for p in sorted(polys, key=lambda p: (bbox(p)[1], bbox(p)[0]))]

    return [line for part in parts for line in _xy_cut(part, med_h)]
