"""Collections management tab."""

import gradio as gr

from src.ui.api_client import ApiClient
from src.ui.constants import (
    COLLECTIONS_TABLE_HEADERS,
    DELETE_COLLECTION_LABEL_ARMED,
    DELETE_COLLECTION_STATUS_SUCCESS,
    DELETE_LABEL_IDLE,
    DELETE_STATUS_ERROR_PREFIX,
)

DELETE_COLUMN_INDEX = len(COLLECTIONS_TABLE_HEADERS) - 1


def create(client: ApiClient) -> tuple[callable, gr.Dataframe, gr.State]:
    """Render the Collections tab. Returns (load_fn, table, state) for app.load."""

    with gr.Row():
        with gr.Column(scale=2):
            collections_table = gr.Dataframe(
                headers=COLLECTIONS_TABLE_HEADERS,
                label="Collections",
                interactive=False,
            )
            collections_msg = gr.Markdown()
            refresh_btn = gr.Button("Refresh", size="sm")
        with gr.Column(scale=1):
            gr.Markdown("### Create collection")
            new_name = gr.Textbox(label="Name")
            new_desc = gr.Textbox(label="Description", lines=2)
            create_btn = gr.Button("Create", variant="primary")
            create_msg = gr.Markdown()

    collections_state = gr.State([])
    armed = gr.State(None)

    def _collection_rows(collections, armed_id=None):
        rows = []
        for c in collections:
            action = (
                DELETE_COLLECTION_LABEL_ARMED if c["id"] == armed_id else DELETE_LABEL_IDLE
            )
            rows.append([c["id"], c["name"], c["description"], c["created_at"], action])
        return rows

    def load_collections():
        collections = client.list_collections()
        return gr.update(value=_collection_rows(collections)), collections

    def on_select(evt: gr.SelectData, collections, armed_id):
        row_idx, col_idx = evt.index
        if col_idx != DELETE_COLUMN_INDEX or row_idx >= len(collections):
            # Any click outside the trash column disarms — two rows armed at
            # once must never be possible.
            return gr.update(value=_collection_rows(collections)), collections, None, gr.update()

        collection = collections[row_idx]
        if collection["id"] != armed_id:
            rows = _collection_rows(collections, armed_id=collection["id"])
            return gr.update(value=rows), collections, collection["id"], gr.update()

        try:
            client.delete_collection(collection["id"])
            msg = DELETE_COLLECTION_STATUS_SUCCESS.format(name=collection["name"])
        except Exception as e:
            msg = f"{DELETE_STATUS_ERROR_PREFIX}{e}"

        table_update, new_collections = load_collections()
        return table_update, new_collections, None, msg

    collections_table.select(
        fn=on_select,
        inputs=[collections_state, armed],
        outputs=[collections_table, collections_state, armed, collections_msg],
    )

    refresh_btn.click(fn=load_collections, outputs=[collections_table, collections_state])

    def create_collection(name, description):
        if not name.strip():
            return "Name is required.", gr.update(), gr.update()
        try:
            result = client.create_collection(name, description)
            table_update, collections = load_collections()
            return f"Created: **{result['name']}**", table_update, collections
        except Exception as e:
            return f"Error: {e}", gr.update(), gr.update()

    create_btn.click(
        fn=create_collection,
        inputs=[new_name, new_desc],
        outputs=[create_msg, collections_table, collections_state],
    )

    return load_collections, collections_table, collections_state
