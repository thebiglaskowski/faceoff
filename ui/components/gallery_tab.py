"""Gallery tab component for viewing processed media."""

import gradio as gr
import os
from pathlib import Path
from ui.helpers.gallery_utils import (
    get_image_files, get_gif_files, get_video_files, 
    count_media_files, clear_gallery_cache
)


def create_gallery_tab():
    """Create the gallery tab for viewing processed results."""
    
    with gr.Tab("📁 Gallery"):
        gr.Markdown("## Results Gallery")
        gr.Markdown("View all your processed images, GIFs, and videos.")
        
        with gr.Row():
            # Media type selector
            media_type_radio = gr.Radio(
                choices=["Images", "GIFs", "Videos"],
                value="Images",
                label="Media Type",
                info="Select which type of media to display"
            )
            
            # Limit selector
            limit_selector = gr.Dropdown(
                choices=["12", "24", "50", "100"],
                value="24",
                label="Files to Show",
                info="Fewer files = faster loading"
            )
            
            # Refresh button
            refresh_btn = gr.Button("🔄 Refresh Gallery", variant="secondary", size="sm")
        
        # File count display
        file_count_text = gr.Markdown(value=_get_count_text())
        
        # Gallery display - optimized for performance
        gallery = gr.Gallery(
            label="Processed Files",
            show_label=False,
            elem_id="gallery",
            columns=4,
            rows=3,
            height="auto",
            object_fit="cover",  # Faster rendering than "contain"
            allow_preview=True,
            selected_index=None
        )
        
        # Delete section (appears when viewing a file)
        with gr.Row():
            with gr.Column():
                selected_file_display = gr.Textbox(
                    label="Selected File",
                    value="",
                    interactive=False,
                    visible=False
                )
            with gr.Column(scale=0):
                delete_btn = gr.Button("🗑️ Delete Selected File", variant="stop", size="sm", visible=False)
        
        delete_status = gr.Markdown(value="", visible=False)

        processing_settings_display = gr.Markdown(
            value="",
            label="Processing Settings",
            visible=False,
        )
        
        # Info text
        gr.Markdown("""
        **Tips:**
        - Click any item to preview it and see the settings used to create it
        - Captions under thumbnails show a short settings summary
        - Adjust "Files to Show" for faster/slower loading
        - Files are sorted by newest first
        - Select a file to view full settings or delete it
        """)
    
    return {
        "media_type_radio": media_type_radio,
        "limit_selector": limit_selector,
        "refresh_btn": refresh_btn,
        "gallery": gallery,
        "file_count_text": file_count_text,
        "selected_file_display": selected_file_display,
        "delete_btn": delete_btn,
        "delete_status": delete_status,
        "processing_settings_display": processing_settings_display,
    }


def _get_count_text():
    """Generate file count text."""
    counts = count_media_files()
    return f"**📊 Files:** {counts['image']} Images | {counts['gif']} GIFs | {counts['video']} Videos"


def update_gallery(media_type: str, limit: str = "24", force_refresh: bool = False):
    """
    Update gallery based on selected media type and limit.

    Args:
        media_type: "Images", "GIFs", or "Videos"
        limit: Number of files to show as string
        force_refresh: If True, clear cache before loading (for media type changes)

    Returns:
        Tuple of (gallery_update, count_text_update)
    """
    import logging
    logger = logging.getLogger("FaceOff")

    # Map display names to internal types
    type_map = {
        "Images": "image",
        "GIFs": "gif",
        "Videos": "video"
    }

    internal_type = type_map.get(media_type, "image")
    max_files = int(limit)

    # Force cache refresh when switching media types for responsive UX
    # This ensures users always see fresh data when toggling
    clear_gallery_cache(internal_type)

    # Get files with specified limit
    if internal_type == "image":
        files = get_image_files(max_files=max_files)
    elif internal_type == "gif":
        files = get_gif_files(max_files=max_files)
    elif internal_type == "video":
        files = get_video_files(max_files=max_files)
    else:
        files = []

    logger.info(f"update_gallery: Returning {len(files)} files for {media_type} ({internal_type})")
    if files:
        logger.debug(f"First file: {files[0][0]}")

    # Update count text with info about limiting
    count_text = _get_count_text()
    if len(files) >= max_files:
        count_text += f"\n\n_Showing {max_files} most recent {media_type.lower()}_"

    return gr.update(value=files, selected_index=None), gr.update(value=count_text)


def refresh_gallery(media_type: str, limit: str = "24"):
    """
    Refresh gallery by clearing cache and reloading.
    
    Args:
        media_type: "Images", "GIFs", or "Videos"
        limit: Number of files to show as string
        
    Returns:
        Tuple of (gallery_update, count_text_update)
    """
    # Map display names to internal types
    type_map = {
        "Images": "image",
        "GIFs": "gif",
        "Videos": "video"
    }
    
    internal_type = type_map.get(media_type, "image")
    
    # Clear cache for this media type to force reload
    clear_gallery_cache(internal_type)
    
    # Now get fresh data
    return update_gallery(media_type, limit)

