"""Search tab."""

import tempfile

import gradio as gr

from src.ui.api_client import ApiClient
from src.ui.constants import (
    CHUNK_PREVIEW_MAX_CHARS,
    DEFAULT_SEARCH_STRATEGY,
    DOWNLOAD_FILE_PREFIX,
    SEARCH_STRATEGIES,
    TOP_K_DEFAULT,
    TOP_K_MAX,
    TOP_K_MIN,
)


def create(client: ApiClient) -> tuple[callable, gr.Dropdown]:
    """Render the Search tab."""

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Your question",
                placeholder="Ask something about your documents...",
                lines=2,
            )
            with gr.Row():
                gr.Column(scale=4)  # spacer: pushes the button to the right
                with gr.Column(scale=1, min_width=140):
                    search_btn = gr.Button("Search", variant="primary")
            answer_output = gr.Markdown(label="Answer")
            sources_output = gr.Dataframe(
                headers=["Content", "Score", "Source file"],
                label="Sources",
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                gr.Column(scale=4)  # spacer: pushes the button to the right
                with gr.Column(scale=1, min_width=140):
                    download_btn = gr.Button("Download answer", size="sm")
            download_file = gr.File(label="Download", visible=False)

        with gr.Column(scale=1, min_width=260):
            search_collection = gr.Dropdown(
                label="Collection",
                choices=[],
                value=None,
                allow_custom_value=False,
                interactive=True,
            )
            with gr.Row():
                gr.Column(scale=1)  # spacer: pushes the button to the right
                with gr.Column(scale=1, min_width=100):
                    refresh_btn = gr.Button("Refresh", size="sm")
            search_strategy = gr.Radio(
                choices=SEARCH_STRATEGIES,
                value=DEFAULT_SEARCH_STRATEGY,
                label="Search strategy",
            )
            top_k_slider = gr.Slider(
                minimum=TOP_K_MIN,
                maximum=TOP_K_MAX,
                value=TOP_K_DEFAULT,
                step=1,
                label="Top K results",
            )

    last_answer = gr.State("")

    def refresh_collections():
        return gr.update(choices=client.collection_choices())

    refresh_btn.click(fn=refresh_collections, outputs=[search_collection])

    def do_search(query, collection_id, strategy, top_k):
        if not query.strip():
            yield "", gr.update(value=[]), ""
            return
        if not collection_id:
            yield "**Please select a collection to search.**", gr.update(value=[]), ""
            return

        accumulated = ""
        sources_rows: list = []

        try:
            for event, data in client.search_stream(
                query,
                collection_id=collection_id,
                top_k=int(top_k),
                strategy=strategy,
            ):
                if event == "sources":
                    sources_rows = [
                        [
                            s["chunk_content"][:CHUNK_PREVIEW_MAX_CHARS],
                            f"{s['score']:.4f}",
                            s["document_filename"],
                        ]
                        for s in data
                    ]
                    yield "", gr.update(value=sources_rows), ""
                elif event == "delta":
                    accumulated += data["text"]
                    yield accumulated, gr.update(value=sources_rows), ""
                elif event == "done":
                    yield accumulated, gr.update(value=sources_rows), accumulated
                    return
                elif event == "error":
                    detail = data.get("detail", "unknown error")
                    yield accumulated + f"\n\n**Error:** {detail}", gr.update(value=sources_rows), ""
                    return
        except Exception as e:
            yield f"**Error:** {e}", gr.update(value=[]), ""

    search_btn.click(
        fn=do_search,
        inputs=[query_input, search_collection, search_strategy, top_k_slider],
        outputs=[answer_output, sources_output, last_answer],
    )

    def download_answer(answer_text):
        if not answer_text:
            return gr.update(visible=False)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix=DOWNLOAD_FILE_PREFIX
        )
        tmp.write(answer_text)
        tmp.close()
        return gr.update(value=tmp.name, visible=True)

    download_btn.click(fn=download_answer, inputs=[last_answer], outputs=[download_file])

    return refresh_collections, search_collection
