"""Gradio interface for RAGBook — composition root."""

import gradio as gr

from src.ui.api_client import ApiClient
from src.ui.constants import (
    APP_HEADER,
    APP_TITLE,
    HEALTH_POLL_SECONDS,
    TAB_COLLECTIONS,
    TAB_SEARCH,
    TAB_UPLOAD,
)
from src.ui.tabs import collections_tab, search_tab, upload_tab
from src.ui.warmup_banner import banner_state


GRADIO_CSS = """
.gradio-dropdown label > div,
.gradio-dropdown .wrap,
.gradio-dropdown .wrap-selection {
    cursor: pointer !important;
    pointer-events: all !important;
}
/* Gradio puts elem_classes on both the outer .block wrapper and the inner
   .prose div, so the banner rules are scoped to the wrapper and the inner
   copy is neutralised — otherwise every border and padding is drawn twice. */
.block.warmup-banner-info,
.block.warmup-banner-error {
    padding: 8px 12px !important;
    border-radius: 4px !important;
    border-left-width: 4px !important;
    border-left-style: solid !important;
}
.block.warmup-banner-info {
    background-color: #e7f3ff !important;
    border-left-color: #1c7ed6 !important;
}
.block.warmup-banner-error {
    background-color: #fff5f5 !important;
    border-left-color: #e03131 !important;
}
.prose.warmup-banner-info,
.prose.warmup-banner-error {
    background: none !important;
    border: 0 !important;
    padding: 0 !important;
}
.block.search-status {
    padding: 8px 12px !important;
    border-radius: 4px !important;
    border-left-width: 4px !important;
    border-left-style: solid !important;
    background-color: #fff8e1 !important;
    border-left-color: #f59f00 !important;
}
.prose.search-status {
    background: none !important;
    border: 0 !important;
    padding: 0 !important;
}
"""


def create_gradio_app(api_base_url: str) -> gr.Blocks:
    """Assemble the Gradio Blocks interface from individual tabs."""

    client = ApiClient(api_base_url)

    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as app:
        gr.Markdown(APP_HEADER)

        warmup_banner = gr.Markdown(visible=False, elem_classes=["warmup-banner-info"])
        warmup_timer = gr.Timer(HEALTH_POLL_SECONDS)

        with gr.Tab(TAB_SEARCH):
            search_load_fn, search_outputs = search_tab.create(client)

        with gr.Tab(TAB_COLLECTIONS):
            collections_load_fn, table = collections_tab.create(client)

        with gr.Tab(TAB_UPLOAD):
            upload_load_fn, upload_dropdown = upload_tab.create(client)

        def _poll_health():
            health = client.health()
            text, timer_active, unchanged, variant = banner_state(health)
            if unchanged:
                return gr.update(), gr.Timer(active=True)
            return (
                gr.update(value=text, visible=text is not None, elem_classes=[variant]),
                gr.Timer(active=timer_active),
            )

        warmup_timer.tick(fn=_poll_health, outputs=[warmup_banner, warmup_timer])
        app.load(fn=_poll_health, outputs=[warmup_banner, warmup_timer])

        app.load(fn=collections_load_fn, outputs=[table])
        app.load(fn=search_load_fn, outputs=search_outputs)
        app.load(fn=upload_load_fn, outputs=[upload_dropdown])

    return app
