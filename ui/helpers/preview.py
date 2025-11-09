"""
Preview utilities for UI.
Helper functions for displaying file previews.
"""
import logging
import gradio as gr
from pathlib import Path

logger = logging.getLogger("FaceOff")


def show_gif_preview(file):
    """
    Show GIF preview when uploaded.
    
    Args:
        file: Uploaded file object from Gradio
        
    Returns:
        Gradio update dict for image component
    """
    if file is None:
        return gr.update(visible=False, value=None)
    
    file_path = file.name if hasattr(file, 'name') else file
    return gr.update(visible=True, value=file_path)


def show_video_preview(file):
    """
    Show video preview when uploaded.
    
    Args:
        file: Uploaded file object from Gradio
        
    Returns:
        Gradio update dict for video component
    """
    if file is None:
        return gr.update(visible=False, value=None)
    
    file_path = file.name if hasattr(file, 'name') else file
    return gr.update(visible=True, value=file_path)
