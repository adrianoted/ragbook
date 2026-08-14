"""Search tab."""

import tempfile

import gradio as gr

from src.ui.api_client import ApiClient
from src.ui.constants import (
    DEFAULT_SEARCH_STRATEGY,
    DOWNLOAD_FILE_PREFIX,
    FUSION_CHOICES,
    LLM_NUM_CTX_INFO,
    LLM_THINK_INFO,
    RERANKER_INFO,
    SEARCH_STATUS_VARIANT,
    SEARCH_STRATEGIES,
    TOP_K_DEFAULT,
    TOP_K_MAX,
    TOP_K_MIN,
    TUNING_FALLBACK_DEFAULTS,
    TUNING_FALLBACK_RANGES,
    TUNING_LABELS,
)
from src.ui.search_events import initial_outputs, next_outputs


def create(client: ApiClient) -> tuple[callable, list]:
    """Render the Search tab.

    Returns the load function and the list of components it feeds, in the exact
    order the function yields them, so the caller can wire ``app.load``.
    """

    def _fallback_slider(key: str, info: str | None = None) -> gr.Slider:
        """A slider seeded with the offline fallbacks; the server overrides it at load."""
        lo, hi, step = TUNING_FALLBACK_RANGES[key]
        return gr.Slider(
            minimum=lo,
            maximum=hi,
            value=TUNING_FALLBACK_DEFAULTS[key],
            step=step,
            label=TUNING_LABELS[key],
            info=info,
        )

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
            status_line = gr.Markdown(visible=False, elem_classes=[SEARCH_STATUS_VARIANT])
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
            with gr.Accordion("Advanced", open=False):
                min_score_slider = _fallback_slider("min_score")
                fusion_radio = gr.Radio(
                    choices=FUSION_CHOICES,
                    value=TUNING_FALLBACK_DEFAULTS["fusion"],
                    label=TUNING_LABELS["fusion"],
                )
                hybrid_vector_weight_slider = _fallback_slider("hybrid_vector_weight")
                max_results_per_document_slider = _fallback_slider("max_results_per_document")
                reranker_enabled_checkbox = gr.Checkbox(
                    value=TUNING_FALLBACK_DEFAULTS["reranker_enabled"],
                    label=TUNING_LABELS["reranker_enabled"],
                    info=RERANKER_INFO,
                )
                llm_temperature_slider = _fallback_slider("llm_temperature")
                llm_think_checkbox = gr.Checkbox(
                    value=TUNING_FALLBACK_DEFAULTS["llm_think"],
                    label=TUNING_LABELS["llm_think"],
                    info=LLM_THINK_INFO,
                )
                llm_num_ctx_slider = _fallback_slider("llm_num_ctx", info=LLM_NUM_CTX_INFO)
                reset_btn = gr.Button("Reset to defaults", size="sm")

    last_answer = gr.State("")
    # Server-side defaults, loaded from GET /api/config; drives "Reset to defaults".
    server_defaults = gr.State(dict(TUNING_FALLBACK_DEFAULTS))

    # Controls fed by the load function, in return order. server_defaults must stay
    # last: gradio_app wires app.load against exactly this list.
    tuning_controls = [
        min_score_slider,
        fusion_radio,
        hybrid_vector_weight_slider,
        max_results_per_document_slider,
        reranker_enabled_checkbox,
        llm_temperature_slider,
        llm_think_checkbox,
        llm_num_ctx_slider,
    ]
    load_outputs = [search_collection, *tuning_controls, server_defaults]

    def load_search_tab():
        """Page-load function: fill the collections dropdown and seed the tuning controls.

        Reads GET /api/config here (not at build time, when the server is not yet
        listening). Returns one update per component in ``load_outputs`` order.

        Only ``app.load`` calls this. The Refresh button uses
        ``refresh_collection_choices`` instead, so re-listing collections never
        discards the tuning values the user has just set.
        """
        config = client.get_config()
        defaults = config.get("defaults", {})
        ranges = config.get("ranges", {})
        # When the server is unreachable (config == {}) keep the Ollama-only
        # controls visible: search still works with server-side defaults.
        ollama_only = config.get("llm_provider") == "ollama" if config else True

        def _slider(key, visible=None):
            rng = ranges.get(key) or {}
            fb_min, fb_max, fb_step = TUNING_FALLBACK_RANGES[key]
            kwargs = {
                "value": defaults.get(key, TUNING_FALLBACK_DEFAULTS[key]),
                "minimum": rng.get("min", fb_min),
                "maximum": rng.get("max", fb_max),
                "step": rng.get("step", fb_step),
            }
            if visible is not None:
                kwargs["visible"] = visible
            return gr.update(**kwargs)

        def _value(key, visible=None):
            kwargs = {"value": defaults.get(key, TUNING_FALLBACK_DEFAULTS[key])}
            if visible is not None:
                kwargs["visible"] = visible
            return gr.update(**kwargs)

        stored = defaults or dict(TUNING_FALLBACK_DEFAULTS)

        return (
            gr.update(choices=client.collection_choices()),
            _slider("min_score"),
            _value("fusion"),
            _slider("hybrid_vector_weight"),
            _slider("max_results_per_document"),
            _value("reranker_enabled"),
            _slider("llm_temperature"),
            _value("llm_think", visible=ollama_only),
            _slider("llm_num_ctx", visible=ollama_only),
            stored,
        )

    def refresh_collection_choices():
        """Refresh button: re-list the collections and touch nothing else."""
        return gr.update(choices=client.collection_choices())

    refresh_btn.click(fn=refresh_collection_choices, outputs=[search_collection])

    def reset_to_defaults(defaults):
        """Reset the eight controls to the server defaults (fallbacks if offline)."""
        d = defaults or TUNING_FALLBACK_DEFAULTS
        return tuple(
            gr.update(value=d.get(key, TUNING_FALLBACK_DEFAULTS[key]))
            for key in (
                "min_score",
                "fusion",
                "hybrid_vector_weight",
                "max_results_per_document",
                "reranker_enabled",
                "llm_temperature",
                "llm_think",
                "llm_num_ctx",
            )
        )

    reset_btn.click(fn=reset_to_defaults, inputs=[server_defaults], outputs=tuning_controls)

    def do_search(
        query,
        collection_id,
        strategy,
        top_k,
        min_score,
        fusion,
        hybrid_vector_weight,
        max_results_per_document,
        reranker_enabled,
        llm_temperature,
        llm_think,
        llm_num_ctx,
    ):
        if not query.strip():
            yield "", gr.update(value=[]), "", gr.update(value="", visible=False)
            return
        if not collection_id:
            yield (
                "**Please select a collection to search.**",
                gr.update(value=[]),
                "",
                gr.update(value="", visible=False),
            )
            return

        state = initial_outputs()

        try:
            for event, data in client.search_stream(
                query,
                collection_id=collection_id,
                top_k=int(top_k),
                strategy=strategy,
                min_score=min_score,
                fusion=fusion,
                hybrid_vector_weight=hybrid_vector_weight,
                max_results_per_document=int(max_results_per_document),
                reranker_enabled=reranker_enabled,
                llm_temperature=llm_temperature,
                llm_think=llm_think,
                llm_num_ctx=int(llm_num_ctx),
            ):
                new_state = next_outputs(event, data, state)
                if new_state is None:
                    continue
                state = new_state
                yield (
                    state.answer,
                    gr.update(value=state.sources_rows),
                    state.last_answer,
                    gr.update(value=state.status, visible=bool(state.status)),
                )
                if state.finished:
                    return
        except Exception as e:
            yield f"**Error:** {e}", gr.update(value=[]), "", gr.update(value="", visible=False)
            return

        # The stream ended without a terminal event: `parse_sse_lines` silently
        # drops partial frames, so a truncated response never reaches `done`.
        # Settle on what was accumulated and clear the status line, otherwise the
        # "generating" pill stays up forever with the button spinner already off.
        yield (
            state.answer,
            gr.update(value=state.sources_rows),
            state.answer,
            gr.update(value="", visible=False),
        )

    search_btn.click(
        fn=do_search,
        inputs=[
            query_input,
            search_collection,
            search_strategy,
            top_k_slider,
            *tuning_controls,
        ],
        outputs=[answer_output, sources_output, last_answer, status_line],
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

    # show_progress="hidden": download_file is born visible=False, so it only mounts
    # while this event is already in flight and misses the "complete" status that
    # would clear its tracker — leaving a "processing | N.Ns" pill up forever
    # (Gradio 6.9.0). Suppressing the tracker for this event sidesteps the race.
    download_btn.click(
        fn=download_answer,
        inputs=[last_answer],
        outputs=[download_file],
        show_progress="hidden",
    )

    return load_search_tab, load_outputs
