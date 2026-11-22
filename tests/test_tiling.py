import numpy as np
import pytest

from smallobj.tiling import Tile, iou_matrix, make_tiles, merge_detections, nms, shift_boxes


def test_tiles_cover_image_and_stay_inside():
    w, h, tile, overlap = 1360, 765, 640, 0.2
    tiles = make_tiles(w, h, tile, overlap)
    covered = np.zeros((h, w), dtype=bool)
    for t in tiles:
        assert 0 <= t.x0 < t.x1 <= w and 0 <= t.y0 < t.y1 <= h
        assert t.width == tile and t.height == tile
        covered[t.y0 : t.y1, t.x0 : t.x1] = True
    assert covered.all()
    # stride = 512 -> x starts 0, 512, 720(flush); y starts 0, 125(flush)
    assert len(tiles) == 3 * 2


def test_small_image_gives_single_tile():
    assert make_tiles(300, 200, 640, 0.2) == [Tile(0, 0, 300, 200)]


def test_zero_overlap_tiles_are_disjoint_except_flush_last():
    tiles = make_tiles(1280, 640, 640, 0.0)
    assert len(tiles) == 2
    assert tiles[0].x1 == tiles[1].x0


def test_invalid_overlap():
    with pytest.raises(ValueError):
        make_tiles(100, 100, 50, 1.0)


def test_shift_boxes():
    t = Tile(100, 50, 740, 690)
    out = shift_boxes(np.array([[10, 20, 30, 40]]), t)
    assert out.tolist() == [[110, 70, 130, 90]]


def test_iou_matrix_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
    ious = iou_matrix(a, b)[0]
    assert ious[0] == pytest.approx(1.0)
    assert ious[1] == pytest.approx(25 / 175)
    assert ious[2] == 0.0


def _brute_force_nms(boxes, scores, thr):
    order = np.argsort(-scores, kind="stable")
    keep = []
    for i in order:
        if all(iou_matrix(boxes[i : i + 1], boxes[j : j + 1])[0, 0] <= thr for j in keep):
            keep.append(i)
    return np.array(keep)


def test_nms_matches_brute_force():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 200, size=(60, 2))
    wh = rng.uniform(5, 60, size=(60, 2))
    boxes = np.concatenate([xy, xy + wh], axis=1)
    scores = rng.uniform(0, 1, size=60)
    np.testing.assert_array_equal(nms(boxes, scores, 0.5), _brute_force_nms(boxes, scores, 0.5))


def test_nms_empty():
    assert nms(np.zeros((0, 4)), np.zeros(0)).size == 0


def test_merge_is_class_aware_by_default():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    classes = np.array([0, 1, 0])
    b, s, c = merge_detections(boxes, scores, classes, iou_thr=0.5)
    # the two class-0 duplicates collapse to one, the class-1 box survives
    assert len(b) == 2 and set(c.tolist()) == {0, 1}
    assert s.tolist() == sorted(s.tolist(), reverse=True)
    b2, _, _ = merge_detections(boxes, scores, classes, iou_thr=0.5, class_agnostic=True)
    assert len(b2) == 1
