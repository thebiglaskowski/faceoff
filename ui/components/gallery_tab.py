"""Gallery tab component for viewing processed media."""

import logging
import os
from pathlib import Path

import gradio as gr

from ui.helpers.gallery_utils import (
    clear_gallery_cache,
    count_media_files,
    delete_file,
    get_gif_files,
    get_image_files,
    get_video_files,
)

logger = logging.getLogger("FaceOff")


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
                info="Select which type of media to display",
            )

            # Limit selector
            limit_selector = gr.Dropdown(
                choices=["12", "24", "50", "100"],
                value="24",
                label="Files to Show",
                info="Fewer files = faster loading",
            )

            # Refresh button
            refresh_btn = gr.Button(
                "🔄 Refresh Gallery", variant="secondary", size="sm"
            )

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
            selected_index=None,
        )

        # Delete section (appears when viewing a file)
        with gr.Row():
            with gr.Column():
                selected_file_display = gr.Textbox(
                    label="Selected File", value="", interactive=False, visible=False
                )
            with gr.Column(scale=0):
                delete_btn = gr.Button(
                    "🗑️ Delete Selected File", variant="stop", size="sm", visible=False
                )

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
    type_map = {"Images": "image", "GIFs": "gif", "Videos": "video"}

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

    logger.info(
        f"update_gallery: Returning {len(files)} files for {media_type} ({internal_type})"
    )
    if files:
        logger.debug(f"First file: {files[0][0]}")

    # Update count text with info about limiting
    count_text = _get_count_text()
    if len(files) >= max_files:
        count_text += f"\n\n_Showing {max_files} most recent {media_type.lower()}_"

    return gr.update(value=files, selected_index=None), gr.update(value=count_text)


def show_delete_controls(evt: gr.SelectData, media_type_display: str):
    """Show delete controls and processing settings when a gallery item is selected."""
    from utils.output_metadata import format_settings_detail, load_output_metadata

    logger.info(
        "Gallery selection event - index: %s, value type: %s",
        evt.index,
        type(evt.value),
    )

    if evt.index is None:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            gr.update(value="", visible=False),
        )

    selected_filename = None
    file_path = None

    if isinstance(evt.value, dict):
        caption = evt.value.get("caption", "")
        if caption:
            selected_filename = caption.split("\n")[0].strip()
        file_path = (
            evt.value.get("image", {}).get("path")
            if isinstance(evt.value.get("image"), dict)
            else evt.value.get("image")
        )
    elif isinstance(evt.value, str):
        file_path = evt.value
        selected_filename = Path(evt.value).name

    if not selected_filename and file_path:
        selected_filename = Path(str(file_path)).name

    if selected_filename:
        media_type_map = {"Images": "image", "GIFs": "gif", "Videos": "video"}
        internal_type = media_type_map.get(media_type_display, "image")
        if not file_path:
            file_path = str(Path("outputs") / internal_type / selected_filename)

        metadata = load_output_metadata(file_path)
        settings_md = format_settings_detail(metadata, file_path)

        return (
            gr.update(value=selected_filename, visible=True),
            gr.update(visible=True),
            "",
            gr.update(value=settings_md, visible=True),
        )

    logger.warning("Could not extract filename from gallery selection")
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        gr.update(value="", visible=False),
    )


def delete_and_refresh(file_path: str, media_type_display: str, limit: str = "24"):
    """Delete selected file and refresh the gallery."""
    logger.info(
        f"Delete triggered - file_path: '{file_path}', media_type: '{media_type_display}'"
    )

    if not file_path:
        logger.warning("No file path provided for deletion")
        return (
            "❌ No file selected",  # delete_status
            gr.update(),  # gallery (no change)
            gr.update(),  # file_count_text (no change)
            gr.update(visible=False),  # delete_btn (hide)
            gr.update(visible=False),  # selected_file_display (hide)
            gr.update(value="", visible=False),  # processing_settings_display (hide)
        )

    # Map display name to internal type
    media_type_map = {"Images": "image", "GIFs": "gif", "Videos": "video"}
    media_type = media_type_map.get(media_type_display, "image")
    logger.info(f"Mapped media type: {media_type}")

    # Reconstruct the full path from filename
    output_base = Path("outputs")
    full_file_path = str(output_base / media_type / file_path)
    logger.info(f"Reconstructed full path: {full_file_path}")

    # Delete the file
    _, message = delete_file(full_file_path, media_type)
    logger.info(f"Delete result: {message}")

    # Refresh the gallery (returns tuple of gr.update() objects)
    gallery_update, count_text_update = update_gallery(media_type_display, limit)

    # Return updates
    return (
        message,  # delete_status (plain string for Markdown)
        gallery_update,  # gallery
        count_text_update,  # file_count_text
        gr.update(visible=False),  # delete_btn (hide after delete)
        gr.update(visible=False),  # selected_file_display (hide after delete)
        gr.update(
            value="", visible=False
        ),  # processing_settings_display (hide after delete)
    )


def wire_gallery_tab(gallery_components):
    """Bind all Gallery tab events (moved out of ui/app.py)."""
    # Update gallery when media type changes
    gallery_components["media_type_radio"].change(
        fn=update_gallery,
        inputs=[
            gallery_components["media_type_radio"],
            gallery_components["limit_selector"],
        ],
        outputs=[
            gallery_components["gallery"],
            gallery_components["file_count_text"],
        ],
    )

    # Limit selector change
    gallery_components["limit_selector"].change(
        fn=update_gallery,
        inputs=[
            gallery_components["media_type_radio"],
            gallery_components["limit_selector"],
        ],
        outputs=[
            gallery_components["gallery"],
            gallery_components["file_count_text"],
        ],
    )

    # Refresh gallery button (clears cache and reloads)
    gallery_components["refresh_btn"].click(
        fn=refresh_gallery,
        inputs=[
            gallery_components["media_type_radio"],
            gallery_components["limit_selector"],
        ],
        outputs=[
            gallery_components["gallery"],
            gallery_components["file_count_text"],
        ],
    )

    # Gallery selection handler (show delete button when file selected)
    gallery_components["gallery"].select(
        fn=show_delete_controls,
        inputs=[gallery_components["media_type_radio"]],
        outputs=[
            gallery_components["selected_file_display"],
            gallery_components["delete_btn"],
            gallery_components["delete_status"],
            gallery_components["processing_settings_display"],
        ],
    )

    # Delete button handler
    gallery_components["delete_btn"].click(
        fn=delete_and_refresh,
        inputs=[
            gallery_components["selected_file_display"],
            gallery_components["media_type_radio"],
            gallery_components["limit_selector"],
        ],
        outputs=[
            gallery_components["delete_status"],
            gallery_components["gallery"],
            gallery_components["file_count_text"],
            gallery_components["delete_btn"],
            gallery_components["selected_file_display"],
            gallery_components["processing_settings_display"],
        ],
    )


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
    type_map = {"Images": "image", "GIFs": "gif", "Videos": "video"}

    internal_type = type_map.get(media_type, "image")

    # Clear cache for this media type to force reload
    clear_gallery_cache(internal_type)

    # Now get fresh data
    return update_gallery(media_type, limit)
