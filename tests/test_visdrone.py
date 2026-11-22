import json

import numpy as np
import pytest
from PIL import Image

from smallobj.visdrone import (
    CLASSES,
    convert_split,
    export_coco_gt,
    load_ignored_regions,
    parse_annotation_file,
    write_data_yaml,
)


def _make_split(tmp_path, name="0000001_00001_d_0000001"):
    split_dir = tmp_path / "VisDrone2019-DET-val"
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "annotations").mkdir()
    Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)).save(split_dir / "images" / f"{name}.jpg")
    (split_dir / "annotations" / f"{name}.txt").write_text(
        "\n".join(
            [
                "10,10,20,10,1,4,0,0",  # car
                "0,0,30,20,0,0,0,0",  # ignored region
                "5,5,4,4,1,11,0,1",  # 'others' -> dropped
                "60,40,20,20,1,1,1,2",  # pedestrian, partly outside -> clipped to 4x8
                "1,1,0,5,1,2,0,0",  # degenerate width -> dropped
            ]
        )
        + "\n"
    )
    return split_dir


def test_parse_annotation_file(tmp_path):
    split_dir = _make_split(tmp_path)
    boxes = parse_annotation_file(next((split_dir / "annotations").glob("*.txt")))
    assert len(boxes) == 5
    assert boxes[1].is_ignored_region and not boxes[0].is_ignored_region
    assert boxes[0].area == 200


def test_convert_and_export(tmp_path):
    split_dir = _make_split(tmp_path)
    out = tmp_path / "converted"
    stats = convert_split(split_dir, out, "val", link_images=False)
    assert stats.images == 1
    assert stats.kept_boxes == 2
    assert stats.ignored_regions == 1
    assert stats.dropped_others == 1
    assert stats.dropped_degenerate == 1
    assert stats.per_class == {"car": 1, "pedestrian": 1}

    label = (out / "labels" / "val" / "0000001_00001_d_0000001.txt").read_text().splitlines()
    assert len(label) == 2
    cls, cx, cy, w, h = label[0].split()
    assert cls == "3"  # car -> index 3
    assert (float(cx), float(cy)) == pytest.approx((20 / 64, 15 / 48), abs=1e-6)
    assert (float(w), float(h)) == pytest.approx((20 / 64, 10 / 48), abs=1e-6)
    # clipped pedestrian: x 60..64, y 40..48
    cls2, cx2, cy2, w2, h2 = label[1].split()
    assert cls2 == "0"
    assert (float(w2), float(h2)) == pytest.approx((4 / 64, 8 / 48), abs=1e-6)

    regions = load_ignored_regions(out, "val")
    assert regions["0000001_00001_d_0000001"] == [(0.0, 0.0, 30.0, 20.0)]

    coco = export_coco_gt(out, "val", out / "val_coco.json")
    assert len(coco["images"]) == 1 and len(coco["annotations"]) == 2
    assert [c["name"] for c in coco["categories"]] == CLASSES
    car = coco["annotations"][0]
    assert car["category_id"] == 4 and car["bbox"] == [10.0, 10.0, 20.0, 10.0] and car["area"] == 200.0
    assert json.loads((out / "val_coco.json").read_text())["images"][0]["width"] == 64

    write_data_yaml(out, out / "d.yaml", ("val",))
    text = (out / "d.yaml").read_text()
    assert "val: images/val" in text and "9: motor" in text
