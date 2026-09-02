"""Models tab: installed weights, upstream downloads, cache tools."""

import gradio as gr

from processing.model_manager import (
    AVAILABLE_DOWNLOADS,
    clear_trt_cache,
    get_model_status,
    get_trt_cache_text,
    run_model_download,
    verify_models,
)


def _download_and_refresh(selection: str, force: bool):
    """Run the download, then refresh status table + cache text."""
    log = run_model_download(selection, force)
    return log, get_model_status(), get_trt_cache_text()


def create_models_tab() -> dict:
    """Create the Models tab with self-contained event wiring."""
    with gr.Tab("🧩 Models"):
        gr.Markdown(
            "## Model Management\n"
            "Weights are **not** stored in the git repo — they download "
            "from upstream (Hugging Face / InsightFace releases) on demand."
        )

        gr.Markdown("### Installed Models")
        model_table = gr.Dataframe(
            headers=["Model", "Size", "Status"],
            value=get_model_status(),
            interactive=False,
        )
        with gr.Row():
            refresh_models_btn = gr.Button("🔄 Refresh", size="sm", variant="secondary")
            verify_btn = gr.Button(
                "🔍 Verify integrity", size="sm", variant="secondary"
            )
        verify_report = gr.Markdown(value="")

        gr.Markdown("### Download Models")
        with gr.Row():
            download_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_DOWNLOADS.keys()),
                value="All models (recommended)",
                label="Package",
            )
            force_checkbox = gr.Checkbox(
                label="Force re-download", value=False, info="Even if files exist"
            )
        download_btn = gr.Button("⬇️ Download", variant="primary")
        download_log = gr.Textbox(
            label="Download log",
            value="",
            lines=12,
            max_lines=24,
            interactive=False,
            buttons=["copy"],
        )

        gr.Markdown("### TensorRT Cache")
        cache_text = gr.Markdown(value=get_trt_cache_text())
        clear_cache_btn = gr.Button("🗑️ Clear TensorRT Cache", size="sm")

        refresh_models_btn.click(
            lambda: get_model_status(),
            outputs=[model_table],
        )
        verify_btn.click(
            verify_models,
            outputs=[verify_report],
        )
        download_btn.click(
            _download_and_refresh,
            inputs=[download_dropdown, force_checkbox],
            outputs=[download_log, model_table, cache_text],
            show_progress="full",
        )
        clear_cache_btn.click(
            lambda: (clear_trt_cache(), get_model_status()),
            outputs=[cache_text, model_table],
        )

    return {
        "model_table": model_table,
        "download_dropdown": download_dropdown,
        "download_btn": download_btn,
        "download_log": download_log,
        "cache_text": cache_text,
    }
