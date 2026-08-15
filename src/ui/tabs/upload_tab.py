"""Upload Documents tab."""

import time
from pathlib import Path

import gradio as gr

from src.ui.api_client import ApiClient
from src.ui.constants import (
    DELETE_LABEL_ARMED,
    DELETE_LABEL_IDLE,
    DELETE_STATUS_ERROR_PREFIX,
    DELETE_STATUS_SUCCESS,
    DOCUMENTS_TABLE_HEADERS,
    NO_COLLECTION_PLACEHOLDER,
    SUPPORTED_FILE_TYPES,
    UPLOAD_STATUS_DUPLICATE,
    UPLOAD_STATUS_ERROR_PREFIX,
    UPLOAD_STATUS_QUEUED,
)
from src.ui.upload_events import job_status_text, poll_ingest_job

DELETE_COLUMN_INDEX = len(DOCUMENTS_TABLE_HEADERS) - 1


def create(
    client: ApiClient,
) -> tuple[callable, gr.Dropdown, gr.Dataframe, callable, gr.State]:
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
        column_widths=["45%", "10%", "10%", "20%", "15%"],
        min_width=50,
        label="Uploaded documents",
        interactive=False,
    )

    documents_table = gr.Dataframe(
        headers=DOCUMENTS_TABLE_HEADERS,
        column_widths=["55%", "10%", "25%", "60px"],
        min_width=50,
        label="Documents in collection",
        interactive=False,
    )
    documents_msg = gr.Markdown()

    documents_state = gr.State([])
    armed = gr.State(None)

    def refresh_collections():
        return gr.update(choices=client.collection_choices())

    def _document_rows(docs, armed_id=None):
        rows = []
        for d in docs:
            action = DELETE_LABEL_ARMED if d["id"] == armed_id else DELETE_LABEL_IDLE
            rows.append([d["filename"], d["document_type"], d["created_at"][:19], action])
        return rows

    def load_documents(collection_id):
        if not collection_id:
            return gr.update(value=[]), []
        docs = client.list_documents(collection_id)
        return gr.update(value=_document_rows(docs)), docs

    def on_documents_select(evt: gr.SelectData, docs, armed_id, collection_id):
        row_idx, col_idx = evt.index
        if col_idx != DELETE_COLUMN_INDEX or row_idx >= len(docs):
            return gr.update(value=_document_rows(docs)), docs, None, gr.update()

        doc = docs[row_idx]
        if doc["id"] != armed_id:
            rows = _document_rows(docs, armed_id=doc["id"])
            return gr.update(value=rows), docs, doc["id"], gr.update()

        try:
            client.delete_document(collection_id, doc["id"])
            msg = DELETE_STATUS_SUCCESS.format(filename=doc["filename"])
        except Exception as e:
            msg = f"{DELETE_STATUS_ERROR_PREFIX}{e}"

        table_update, new_docs = load_documents(collection_id)
        return table_update, new_docs, None, msg

    refresh_btn.click(fn=refresh_collections, outputs=[upload_collection])
    upload_collection.change(
        fn=load_documents, inputs=[upload_collection], outputs=[documents_table, documents_state]
    )
    documents_table.select(
        fn=on_documents_select,
        inputs=[documents_state, armed, upload_collection],
        outputs=[documents_table, documents_state, armed, documents_msg],
    )

    def upload_documents(files, collection_id):
        # Generator: submit each file to /ingest/async, then poll the job so the
        # table advances live and a long ingest no longer dies on the client
        # timeout. Serial by design — one polling loop holds one Gradio worker.
        if not files:
            yield gr.update(value=[])
            return

        # The dropdown carries ids, not names; resolve once so the table shows
        # the collection the user actually picked instead of a raw UUID.
        names = {cid: name for name, cid in client.collection_choices()}

        rows = []
        for file_obj in files:
            file_path = file_obj if isinstance(file_obj, str) else file_obj.name
            filename = Path(file_path).name
            suffix = Path(filename).suffix
            try:
                accepted = client.ingest_async(file_path, collection_id or None)
            except Exception as e:
                # Keep today's behaviour: record the error row and move on.
                rows.append([filename, "ERROR", 0, str(e), UPLOAD_STATUS_ERROR_PREFIX.strip()])
                yield gr.update(value=list(rows))
                continue

            resolved_id = accepted.get("collection_id") or collection_id
            collection = names.get(resolved_id, resolved_id)
            if accepted.get("already_ingested"):
                # Server dedup: nothing scheduled, show the final duplicate row.
                num_chunks = accepted["num_chunks"]
                rows.append([
                    accepted["filename"],
                    suffix,
                    num_chunks,
                    collection or NO_COLLECTION_PLACEHOLDER,
                    UPLOAD_STATUS_DUPLICATE.format(n=num_chunks),
                ])
                yield gr.update(value=list(rows))
                continue

            # New ingest: queued row, then live progress as the job is polled.
            # The current file is always the last row; finished files stay above.
            rows.append([
                filename,
                suffix,
                0,
                collection or NO_COLLECTION_PLACEHOLDER,
                UPLOAD_STATUS_QUEUED,
            ])
            yield gr.update(value=list(rows))

            start = time.monotonic()
            for job in poll_ingest_job(client, accepted["job_id"]):
                # Chunks shows the total as soon as chunking reports it; the
                # status line carries the phase, never a counter.
                chunks = job.get("num_chunks") or job.get("chunks_total", 0)
                rows[-1] = [
                    filename,
                    suffix,
                    chunks,
                    collection or NO_COLLECTION_PLACEHOLDER,
                    job_status_text(job, time.monotonic() - start),
                ]
                yield gr.update(value=list(rows))

    upload_btn.click(
        fn=upload_documents,
        inputs=[upload_files, upload_collection],
        outputs=[upload_output],
    ).then(
        fn=load_documents, inputs=[upload_collection], outputs=[documents_table, documents_state]
    )

    return refresh_collections, upload_collection, documents_table, load_documents, documents_state
