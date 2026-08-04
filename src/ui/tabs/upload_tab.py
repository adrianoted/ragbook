"""Upload Documents tab."""

from pathlib import Path

import gradio as gr

from src.ui.api_client import ApiClient
from src.ui.constants import NO_COLLECTION_PLACEHOLDER, SUPPORTED_FILE_TYPES


def create(client: ApiClient) -> tuple[callable, gr.Dropdown]:
    """Render the Upload Documents tab inside a gr.Tab context."""

    with gr.Row():
        with gr.Column(scale=2):
            upload_files = gr.File(
                label="Drop files here",
                file_count="multiple",
                file_types=SUPPORTED_FILE_TYPES,
            )
        with gr.Column(scale=1):
            upload_collection = gr.Dropdown(
                label="Collection",
                choices=[],
                value=None,
                allow_custom_value=True,
                interactive=True,
            )
            refresh_btn = gr.Button("Refresh collections", size="sm")
            upload_btn = gr.Button("Upload", variant="primary")

    upload_output = gr.Dataframe(
        headers=["Filename", "Type", "Chunks", "Collection", "Status"],
        label="Uploaded documents",
        interactive=False,
    )

    def refresh_collections():
        return gr.update(choices=client.collection_choices())

    refresh_btn.click(fn=refresh_collections, outputs=[upload_collection])

    def upload_documents(files, collection_id):
        if not files:
            return gr.update(value=[])

        rows = []
        for file_obj in files:
            file_path = file_obj if isinstance(file_obj, str) else file_obj.name
            filename = Path(file_path).name
            try:
                r = client.ingest(file_path, collection_id or None)
                if r.get("already_ingested"):
                    status = f"già presente ({r['num_chunks']} chunk)"
                else:
                    status = "✓"
                rows.append([
                    r["filename"],
                    Path(filename).suffix,
                    r["num_chunks"],
                    r.get("collection_id") or NO_COLLECTION_PLACEHOLDER,
                    status,
                ])
            except Exception as e:
                rows.append([filename, "ERROR", 0, str(e), "✗"])
        return gr.update(value=rows)

    upload_btn.click(
        fn=upload_documents,
        inputs=[upload_files, upload_collection],
        outputs=[upload_output],
    )

    return refresh_collections, upload_collection
