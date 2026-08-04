"""Gradio interface for RAGBook — composition root."""

import gradio as gr

from src.ui.api_client import ApiClient
from src.ui.constants import (
    APP_HEADER,
    APP_TITLE,
    TAB_COLLECTIONS,
    TAB_SEARCH,
    TAB_UPLOAD,
)
from src.ui.tabs import collections_tab, search_tab, upload_tab


_CSS = """
.gradio-dropdown label > div,
.gradio-dropdown .wrap,
.gradio-dropdown .wrap-selection {
    cursor: pointer !important;
    pointer-events: all !important;
}
"""


def create_gradio_app(api_base_url: str) -> gr.Blocks:
    """Assemble the Gradio Blocks interface from individual tabs."""

    client = ApiClient(api_base_url)

    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft(), css=_CSS) as app:
        gr.Markdown(APP_HEADER)

        with gr.Tab(TAB_SEARCH):
            search_load_fn, search_dropdown = search_tab.create(client)

        with gr.Tab(TAB_COLLECTIONS):
            collections_load_fn, table = collections_tab.create(client)

        with gr.Tab(TAB_UPLOAD):
            upload_load_fn, upload_dropdown = upload_tab.create(client)

        app.load(fn=collections_load_fn, outputs=[table])
        app.load(fn=search_load_fn, outputs=[search_dropdown])
        app.load(fn=upload_load_fn, outputs=[upload_dropdown])

    return app
