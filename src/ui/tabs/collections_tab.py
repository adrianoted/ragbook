"""Collections management tab."""

import gradio as gr

from src.ui.api_client import ApiClient


def create(client: ApiClient) -> tuple[callable, gr.Dataframe]:
    """Render the Collections tab. Returns (load_fn, output) for app.load."""

    with gr.Row():
        with gr.Column(scale=2):
            collections_table = gr.Dataframe(
                headers=["ID", "Name", "Description", "Created"],
                label="Collections",
                interactive=False,
            )
            refresh_btn = gr.Button("Refresh", size="sm")
        with gr.Column(scale=1):
            gr.Markdown("### Create collection")
            new_name = gr.Textbox(label="Name")
            new_desc = gr.Textbox(label="Description", lines=2)
            create_btn = gr.Button("Create", variant="primary")
            create_msg = gr.Markdown()

            gr.Markdown("### Delete collection")
            del_id = gr.Textbox(label="Collection ID to delete")
            delete_btn = gr.Button("Delete", variant="stop")
            delete_msg = gr.Markdown()

    def load_collections():
        raw = client.list_collections()
        rows = [[c["id"], c["name"], c["description"], c["created_at"]] for c in raw]
        return gr.update(value=rows)

    def on_select(evt: gr.SelectData, data):
        row_idx = evt.index[0]
        return data.iloc[row_idx, 0] if data is not None and len(data) > row_idx else ""

    collections_table.select(fn=on_select, inputs=[collections_table], outputs=[del_id])

    refresh_btn.click(fn=load_collections, outputs=[collections_table])

    def create_collection(name, description):
        if not name.strip():
            return "Name is required.", gr.update()
        try:
            result = client.create_collection(name, description)
            return f"Created: **{result['name']}**", load_collections()
        except Exception as e:
            return f"Error: {e}", gr.update()

    create_btn.click(
        fn=create_collection,
        inputs=[new_name, new_desc],
        outputs=[create_msg, collections_table],
    )

    def delete_collection(cid):
        if not cid.strip():
            return "Enter a collection ID.", gr.update()
        try:
            client.delete_collection(cid)
            return "Deleted.", load_collections()
        except Exception as e:
            return f"Error: {e}", gr.update()

    delete_btn.click(
        fn=delete_collection,
        inputs=[del_id],
        outputs=[delete_msg, collections_table],
    )

    return load_collections, collections_table
