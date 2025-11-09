"""
ONNX model optimization for faster inference.

This module optimizes ONNX models by:
- Constant folding
- Layer fusion  
- Dead node elimination
- Graph simplification
"""
import logging
import onnx
from pathlib import Path
from typing import Optional

logger = logging.getLogger("FaceOff")


def optimize_onnx_model(
    model_path: str,
    optimized_path: Optional[str] = None,
    optimization_level: str = "all"
) -> str:
    """
    Optimize an ONNX model for faster inference.
    
    Args:
        model_path: Path to original ONNX model
        optimized_path: Path to save optimized model (if None, uses _optimized suffix)
        optimization_level: Level of optimization ('basic', 'extended', 'all')
        
    Returns:
        Path to optimized model
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.warning("Model file not found: %s", model_path)
        return str(model_path)
    
    # Generate optimized path if not provided
    if optimized_path is None:
        optimized_path = model_path.parent / f"{model_path.stem}_optimized{model_path.suffix}"
    else:
        optimized_path = Path(optimized_path)
    
    # Skip if optimized version already exists
    if optimized_path.exists():
        logger.info("Using cached optimized model: %s", optimized_path)
        return str(optimized_path)
    
    try:
        logger.info("Optimizing ONNX model: %s", model_path.name)
        
        # Load model
        model = onnx.load(str(model_path))
        
        # Check model validity
        onnx.checker.check_model(model)
        
        # Apply optimization passes
        import onnxoptimizer
        
        if optimization_level == "basic":
            passes = [
                'eliminate_identity',
                'eliminate_nop_transpose',
                'eliminate_nop_pad',
                'eliminate_unused_initializer'
            ]
        elif optimization_level == "extended":
            passes = [
                'eliminate_identity',
                'eliminate_nop_transpose',
                'eliminate_nop_pad',
                'eliminate_unused_initializer',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_transpose_into_gemm'
            ]
        else:  # 'all'
            passes = [
                'eliminate_deadend',
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_monotone_argmax',
                'eliminate_nop_pad',
                'eliminate_nop_transpose',
                'eliminate_unused_initializer',
                'extract_constant_to_initializer',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
                'fuse_consecutive_concats',
                'fuse_consecutive_log_softmax',
                'fuse_consecutive_reduce_unsqueeze',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm',
            ]
        
        # Apply optimizations
        optimized_model = onnxoptimizer.optimize(model, passes)
        
        # Validate the optimized model before saving
        try:
            onnx.checker.check_model(optimized_model)
        except Exception as validation_error:
            logger.warning(
                "Optimized model failed validation: %s. Using original model.",
                validation_error
            )
            return str(model_path)
        
        # Save optimized model
        onnx.save(optimized_model, str(optimized_path))
        
        # Get file size reduction
        original_size = model_path.stat().st_size / (1024 * 1024)  # MB
        optimized_size = optimized_path.stat().st_size / (1024 * 1024)  # MB
        reduction = ((original_size - optimized_size) / original_size) * 100
        
        logger.info(
            "Model optimized: %.2f MB -> %.2f MB (%.1f%% reduction)",
            original_size, optimized_size, reduction
        )
        
        return str(optimized_path)
        
    except Exception as e:
        logger.warning("Model optimization failed: %s. Using original model.", e)
        # Clean up any partially created optimized file
        if optimized_path and Path(optimized_path).exists():
            try:
                Path(optimized_path).unlink()
            except:
                pass
        return str(model_path)


def optimize_all_models(models_dir: Path) -> dict:
    """
    Optimize all ONNX models in a directory.
    
    Args:
        models_dir: Directory containing ONNX models
        
    Returns:
        Dictionary mapping original paths to optimized paths
    """
    optimized_models = {}
    
    for model_file in models_dir.rglob("*.onnx"):
        # Skip already optimized models
        if "_optimized" in model_file.stem:
            continue
            
        optimized_path = optimize_onnx_model(str(model_file))
        optimized_models[str(model_file)] = optimized_path
    
    return optimized_models
