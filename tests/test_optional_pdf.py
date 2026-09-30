"""Raster segmentation must work without the optional PDF dependency."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _run_without_pdf(code: str, path: Path) -> None:
    # A fresh process catches eager imports even when another test loaded DECIMER.
    bootstrap = """
import sys
sys.modules["pymupdf"] = None
sys.modules["fitz"] = None
from decimer_segmentation import decimer_segmentation as segmentation
"""
    subprocess.run(
        [sys.executable, "-c", bootstrap + textwrap.dedent(code), str(path)],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "3"},
    )


def test_raster_segmentation_without_pdf_extra(tmp_path: Path) -> None:
    """Exercise real image loading and cropping with only model prediction stubbed."""
    _run_without_pdf(
        """
        import cv2
        import numpy as np

        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        image[20:40, 10:30] = [10, 20, 30]
        assert cv2.imwrite(sys.argv[1], image)
        masks = np.zeros((64, 64, 1), dtype=bool)
        masks[20:40, 10:30, 0] = True
        segmentation.get_mrcnn_results = lambda image: (
            masks, np.array([[20, 10, 40, 30]]), np.array([0.95])
        )

        segments = segmentation.segment_chemical_structures_from_file(
            sys.argv[1], expand=False
        )
        assert len(segments) == 1
        assert segments[0].shape == (20, 20, 4)
        assert segments[0].dtype == np.uint8
        assert np.all(segments[0] == [10, 20, 30, 255])
        """,
        tmp_path / "image.png",
    )


def test_pdf_segmentation_without_extra_requires_pymupdf(tmp_path: Path) -> None:
    """Missing PDF support fails on a PDF request, not a raster request."""
    _run_without_pdf(
        """
        try:
            segmentation.segment_chemical_structures_from_file(sys.argv[1])
        except ModuleNotFoundError as error:
            assert error.name == "pymupdf"
        else:
            raise AssertionError("PDF segmentation requires the pdf extra")
        """,
        tmp_path / "document.pdf",
    )
