"""Backward-compatible re-export — implementation lives in utils/."""

from utils.model_optimizer import optimize_all_models, optimize_onnx_model

__all__ = ["optimize_onnx_model", "optimize_all_models"]