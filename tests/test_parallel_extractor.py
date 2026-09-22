from pathlib import Path

import pytest

from backend.parallel_extractor import (
    ParallelExtractorConfigError,
    _COCO_TO_MP,
    validate_parallel_config,
)


def test_parallel_config_requires_all_paths():
    with pytest.raises(ParallelExtractorConfigError, match="YOLOv8 model"):
        validate_parallel_config(None, None, None)


def test_parallel_config_requires_existing_matching_config_files(tmp_path: Path):
    yolo = tmp_path / "yolo.pt"
    config = tmp_path / "vitpose.py"
    checkpoint = tmp_path / "vitpose.pth"
    for path in (yolo, config, checkpoint):
        path.write_text("placeholder", encoding="utf-8")
    assert validate_parallel_config(yolo, config, checkpoint) == (yolo, config, checkpoint)


def test_parallel_config_reports_missing_checkpoint(tmp_path: Path):
    yolo = tmp_path / "yolo.pt"
    config = tmp_path / "vitpose.py"
    yolo.write_text("", encoding="utf-8")
    config.write_text("", encoding="utf-8")
    with pytest.raises(ParallelExtractorConfigError, match="ViTPose checkpoint"):
        validate_parallel_config(yolo, config, tmp_path / "missing.pth")


def test_parallel_mapping_preserves_coco_left_right_order():
    assert _COCO_TO_MP[5] == 11
    assert _COCO_TO_MP[6] == 12
    assert _COCO_TO_MP[9] == 15
    assert _COCO_TO_MP[10] == 16
