"""Tests for stage-aware swap progress (stage_cb plumbing)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _image_opts(tmp_path):
    from processing.orchestrator import ProcessOptions

    return ProcessOptions(
        source_image=np.zeros((8, 8, 3), dtype=np.uint8),
        dest_path=str(tmp_path / "dest.png"),
        media_type="image",
        output_dir=str(tmp_path / "outputs"),
        gpu_selection="GPU 0: test",
    )


def _patched_orchestrator():
    return (
        patch("processing.orchestrator.MediaProcessor"),
        patch("processing.orchestrator.compress_media_file"),
        patch("processing.orchestrator.save_output_metadata"),
    )


class TestStageCallback:
    def test_process_media_forwards_stage_cb_to_image(self, tmp_path):
        """stage_cb given to process_media must reach process_image."""
        from processing.orchestrator import process_media

        seen = {}

        def fake_process_image(*args, **kwargs):
            seen.update(kwargs)
            cb = kwargs.get("stage_cb")
            assert callable(cb)
            cb(0.45, "Face swap complete — restoring…")
            return ("out.png", None)

        mock_proc, mock_compress, mock_meta = _patched_orchestrator()
        with (
            mock_proc,
            mock_compress,
            mock_meta,
            patch(
                "processing.orchestrator.process_image",
                side_effect=fake_process_image,
            ),
        ):
            stages = []
            process_media(
                _image_opts(tmp_path), stage_cb=lambda f, d: stages.append((f, d))
            )

        assert seen.get("stage_cb") is not None
        assert stages == [(0.45, "Face swap complete — restoring…")]

    def test_process_media_without_stage_cb_passes_none(self, tmp_path):
        """Omitting stage_cb must pass None down, not crash dispatch."""
        from processing.orchestrator import process_media

        seen = {}

        def fake_process_image(*args, **kwargs):
            seen.update(kwargs)
            return ("out.png", None)

        mock_proc, mock_compress, mock_meta = _patched_orchestrator()
        with (
            mock_proc,
            mock_compress,
            mock_meta,
            patch(
                "processing.orchestrator.process_image",
                side_effect=fake_process_image,
            ),
        ):
            process_media(_image_opts(tmp_path))

        assert "stage_cb" in seen
        assert seen["stage_cb"] is None

    def test_process_input_bridges_stages_to_gradio_progress(
        self, tmp_path, monkeypatch
    ):
        """The real _process_input must forward stage descs to progress()."""
        from PIL import Image as PILImage

        import ui.handlers.processing_handlers as handlers

        monkeypatch.chdir(tmp_path)
        target = tmp_path / "target.png"
        PILImage.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(target)
        source = PILImage.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))

        calls = []

        def fake_progress(frac, desc=None):
            calls.append((frac, desc))

        def fake_process_media(opts, stage_cb=None):
            assert callable(stage_cb)
            stage_cb(0.45, "Face swap complete — restoring…")
            return ("done.png", None)

        with (
            patch.object(handlers, "process_media", side_effect=fake_process_media),
            patch.object(handlers, "invalidate_gallery_for_new_file"),
        ):
            result = handlers._process_input(
                source,
                target_image_path=str(target),
                progress=fake_progress,
            )

        assert result == "done.png"
        assert calls[0] == (0.02, "Validating inputs…")
        assert (0.45, "Face swap complete — restoring…") in calls

    def test_process_input_survives_failing_progress_sink(self, tmp_path, monkeypatch):
        """A raising progress() must not break the swap itself."""
        from PIL import Image as PILImage

        import ui.handlers.processing_handlers as handlers

        monkeypatch.chdir(tmp_path)
        target = tmp_path / "target.png"
        PILImage.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(target)
        source = PILImage.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))

        def boom(frac, desc=None):
            raise RuntimeError("progress sink exploded")

        def fake_process_media(opts, stage_cb=None):
            stage_cb(0.9, "Encoding video…")
            return ("done.png", None)

        with (
            patch.object(handlers, "process_media", side_effect=fake_process_media),
            patch.object(handlers, "invalidate_gallery_for_new_file"),
        ):
            result = handlers._process_input(
                source,
                target_image_path=str(target),
                progress=boom,
            )

        assert result == "done.png"
