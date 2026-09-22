from pathlib import Path

import pytest

from backend.parallel_extractor import (
    ParallelExtractorConfigError,
    _COCO_TO_MP,
    validate_parallel_config,
)
from backend.process_video import _find_parallel_config


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


def test_parallel_config_search_finds_easy_vitpose_layout(tmp_path: Path, monkeypatch):
    model_root = tmp_path / "model"
    config = (
        tmp_path / "easy_ViTPose" / "easy_ViTPose" / "configs"
        / "ViTPose_coco_25.py"
    )
    config.parent.mkdir(parents=True)
    config.write_text("channel_cfg = dict(num_output_channels=25)", encoding="utf-8")
    monkeypatch.setenv("MIMO_MODEL_DIR", str(model_root))
    assert _find_parallel_config() == config
