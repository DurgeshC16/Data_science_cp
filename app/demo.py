"""Streamlit dashboard for interactively reviewing the Cascade-XID pipeline."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from src.config import load_config
from src.data.loader import normalise_columns
from src.pipeline.cascade import CascadePipeline, load_cascade_from_artifacts


def _cost_estimate(
    stats: dict[str, Any], prompt_usd_per_million: float, output_usd_per_million: float
) -> dict[str, float] | None:
    """Estimate observed cascade spend and all-flow LLM spend from entered token prices.

    Args:
        stats: run statistics emitted by ``CascadePipeline``.
        prompt_usd_per_million: user-supplied prompt-token price.
        output_usd_per_million: user-supplied output-token price.

    Returns:
        Cost estimates and estimated savings, or ``None`` without measured LLM calls.
    """
    calls = int(stats.get("llm_calls", 0))
    total = int(stats.get("total_flows", 0))
    if calls == 0 or total == 0:
        return None
    prompt_tokens = int(stats.get("tokens", {}).get("prompt", 0))
    output_tokens = int(stats.get("tokens", {}).get("output", 0))
    observed = (
        prompt_tokens * prompt_usd_per_million
        + output_tokens * output_usd_per_million
    ) / 1_000_000
    # Baseline assumes one LLM call per flow with the mean observed per-call
    # prompt/completion tokens. This is an explicitly user-selected estimate.
    all_flow_prompt = total * prompt_tokens / calls
    all_flow_output = total * output_tokens / calls
    all_flow = (
        all_flow_prompt * prompt_usd_per_million
        + all_flow_output * output_usd_per_million
    ) / 1_000_000
    return {
        "cascade_usd": observed,
        "llm_every_flow_usd": all_flow,
        "estimated_savings_usd": all_flow - observed,
    }


def _records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a display-only frame from validated records."""
    return pd.DataFrame(records)


def main() -> None:
    """Render the upload, routing, analyst-report, and cost-estimate dashboard."""
    st.set_page_config(page_title="Cascade-XID Demo", layout="wide")
    config = load_config()
    st.title("Cascade-XID — analyst review demo")
    st.caption(
        "Research prototype. Model outputs and draft ATT&CK mappings require analyst review."
    )

    with st.sidebar:
        st.header("Run configuration")
        dataset = st.selectbox("Model artifact dataset", ["cicids2017", "ciciot2023"])
        gate_profile = st.selectbox("Gate profile", ["p05", "p10", "p15"], index=1)
        max_flows = st.number_input(
            "Maximum uploaded rows",
            min_value=1,
            max_value=int(config.eval.sample_cap),
            value=min(200, int(config.eval.sample_cap)),
            help="This demo cap is limited by eval.sample_cap.",
        )
        concurrency = st.number_input(
            "Tier-2 concurrency",
            min_value=1,
            max_value=32,
            value=int(config.pipeline.tier2_concurrency),
        )
        st.subheader("Optional cost assumptions")
        prompt_price_text = st.text_input(
            "Prompt USD per 1M tokens (leave blank to disable)", value=""
        )
        output_price_text = st.text_input(
            "Output USD per 1M tokens (leave blank to disable)", value=""
        )

    uploaded = st.file_uploader("Upload a flow CSV", type=["csv"])
    if uploaded is None:
        st.info(
            "Upload a CSV to choose rows and run the cascade. No curated sample flow "
            "records are currently available in this repository."
        )
        st.warning(
            "The clean repository does not include real flow data or trained model "
            "artifacts. After upload, this page reports the exact missing artifact "
            "rather than substituting synthetic predictions."
        )
        return

    try:
        source = normalise_columns(pd.read_csv(uploaded))
    except Exception as error:
        st.error(f"Could not read CSV: {error}")
        return
    if "flow_id" not in source:
        source.insert(0, "flow_id", [f"upload-{i:08d}" for i in range(len(source))])
    if source.empty:
        st.error("The uploaded CSV contains no rows.")
        return
    if len(source) > int(max_flows):
        st.warning(
            f"Only the first {int(max_flows)} rows are eligible; eval.sample_cap="
            f"{config.eval.sample_cap}."
        )
        source = source.iloc[: int(max_flows)].reset_index(drop=True)

    st.subheader("Choose flows from the uploaded CSV")
    options = source["flow_id"].astype(str).tolist()
    selected_ids = st.multiselect(
        "Rows to score (empty means all eligible rows)", options=options, default=[]
    )
    selected = source[
        source["flow_id"].astype(str).isin(selected_ids)
    ] if selected_ids else source
    st.dataframe(selected.head(20), use_container_width=True)

    if st.button("Run cascade", type="primary"):
        try:
            pipeline: CascadePipeline = load_cascade_from_artifacts(
                dataset,
                gate_profile,
                config=config,
                tier2_concurrency=int(concurrency),
            )
        except Exception as error:
            st.error(f"Cascade artifacts are not ready: {error}")
            return

        rows = selected.to_dict(orient="records")
        progress = st.progress(0.0)
        live = st.empty()
        records = []
        escalated = 0
        for index, record in enumerate(
            pipeline.run_stream(rows, batch_size=int(config.pipeline.batch_size)),
            start=1,
        ):
            records.append(record.model_dump(mode="json"))
            escalated += int(record.tier_reached == 2)
            progress.progress(index / len(rows))
            live.metric(
                "Live escalation rate",
                f"{100.0 * escalated / index:.1f}%",
                f"{escalated} / {index} flows escalated",
            )
        st.session_state["cascade_records"] = records
        st.session_state["cascade_flow_details"] = pipeline.flow_details
        st.session_state["cascade_run_stats"] = json.loads(
            pipeline.run_stats_path.read_text(encoding="utf-8")
        )

    records = st.session_state.get("cascade_records")
    if not records:
        return
    stats = st.session_state.get("cascade_run_stats", {})
    record_frame = _records_to_frame(records)
    total = int(stats.get("total_flows", len(records)))
    escalated_count = int(stats.get("escalated", 0))
    columns = st.columns(4)
    columns[0].metric("Total flows", total)
    columns[1].metric("Escalated", escalated_count)
    columns[2].metric("LLM calls", int(stats.get("llm_calls", 0)))
    columns[3].metric(
        "Escalation rate", f"{100.0 * escalated_count / max(total, 1):.1f}%"
    )
    st.dataframe(record_frame, use_container_width=True)

    st.download_button(
        "Download JSONL records",
        data="".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        file_name="cascade_results.jsonl",
        mime="application/jsonl",
    )

    selected_flow_id = st.selectbox(
        "Review a flow", record_frame["flow_id"].astype(str).tolist()
    )
    record = next(row for row in records if str(row["flow_id"]) == selected_flow_id)
    details = st.session_state.get("cascade_flow_details", {}).get(selected_flow_id, {})
    st.subheader("Flow decision")
    st.write(
        "**Tier-1 verdict:**",
        details.get("tier1_verdict", record["final_classification"]),
    )
    st.progress(float(record["confidence"]))
    st.caption(f"Tier-1 confidence: {float(record['confidence']):.1%}")
    st.write(
        "**Gate decision:**",
        "Escalated to Tier 2" if record["tier_reached"] == 2 else "Accepted by Tier 1",
    )

    if record["tier_reached"] == 2:
        st.subheader("Analyst report card")
        st.write(record["explanation"])
        technique_id = record.get("attack_technique")
        if technique_id:
            name = ""
            tier2_output = details.get("tier2_output") or {}
            technique = tier2_output.get("attack_technique")
            if isinstance(technique, dict):
                name = technique.get("name", "")
            st.markdown(
                f"**ATT&CK technique:** [{technique_id} {name}]"
                f"(https://attack.mitre.org/techniques/{technique_id.replace('.', '/')}/)"
            )
        else:
            st.write("**ATT&CK technique:** none validated")
        st.write("**Recommended action:**", record["recommended_action"])
        st.write("**Evidence used:**", record["evidence_used"] or "No retrieved evidence")
        shap_features = details.get("shap_features") or []
        if shap_features:
            chart_frame = pd.DataFrame(
                {
                    "Feature": [item["feature"] for item in shap_features],
                    "SHAP contribution toward Attack": [
                        item["shap_value"] for item in shap_features
                    ],
                }
            ).set_index("Feature")
            st.bar_chart(chart_frame)
        else:
            st.info("No SHAP explanation was available for this model/flow.")
        passages = details.get("passages") or []
        if passages:
            with st.expander("Retrieved reference passages"):
                for passage in passages:
                    st.markdown(
                        f"**{passage.get('technique_id') or passage.get('cve_id') or passage.get('document_id')}**"
                    )
                    st.write(passage.get("text", ""))
    else:
        st.write(record["explanation"])
        st.write("**Recommended action:**", record["recommended_action"])

    st.subheader("Estimated cost saved vs. LLM on every flow")
    try:
        prompt_price = float(prompt_price_text)
        output_price = float(output_price_text)
    except (TypeError, ValueError):
        st.info(
            "Cost estimate not shown: enter prompt and output USD-per-million-token "
            "prices. These rates are user-supplied assumptions, not measured prices."
        )
    else:
        estimate = _cost_estimate(stats, prompt_price, output_price)
        if estimate is None:
            st.info("Cost estimate not measured: this run has no uncached LLM calls.")
        else:
            st.caption(
                "All-flow baseline assumes one call per flow using the mean observed "
                "per-call prompt/completion tokens. This is an estimate, not a measured bill."
            )
            cost_columns = st.columns(3)
            cost_columns[0].metric("Cascade estimate", f"${estimate['cascade_usd']:.6f}")
            cost_columns[1].metric(
                "LLM-every-flow estimate", f"${estimate['llm_every_flow_usd']:.6f}"
            )
            cost_columns[2].metric(
                "Estimated savings", f"${estimate['estimated_savings_usd']:.6f}"
            )


if __name__ == "__main__":
    main()
