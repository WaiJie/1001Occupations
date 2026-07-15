import plotly.express as px
import streamlit as st

from app import config, data


def _explore_map_select():
    state = st.session_state.get("explore_map")
    if state is None:
        return
    sel = state.get("selection", {}) if isinstance(state, dict) else getattr(state, "selection", {})
    pts = (sel or {}).get("points", []) if isinstance(sel, dict) else getattr(sel, "points", []) or []
    if not pts:
        return
    try:
        code = int(pts[0]["customdata"][0])
    except (KeyError, IndexError, TypeError):
        return
    st.session_state.dialog_code = code
    st.session_state.dialog_open = True


def render_explore():
    st.subheader("Search Occupations")
    st.markdown("Search by code or title.")
    map_merged = data.umap_df.merge(
        data.occ_stats_df[["occ_code", "job_post_count"]], left_on="code", right_on="occ_code", how="left"
    )
    map_merged["group"] = map_merged["major_code"].map(config.MAJOR_LABELS)
    occ_options = map_merged.dropna(subset=["title"]).sort_values("title")
    occ_labels = {int(r["code"]): f"{int(r['code'])} — {r['title']}" for _, r in occ_options.iterrows()}
    col_occ_search, col_occ_go = st.columns([4, 1])
    with col_occ_search:
        occ_search = st.selectbox("Search by code / title", options=list(occ_labels.keys()),
            format_func=lambda c: occ_labels.get(c, ""),
            placeholder="Search...", label_visibility="collapsed", index=None, key="explore_occ_search")
    with col_occ_go:
        if st.button("View", disabled=not occ_search, width="stretch"):
            st.session_state.dialog_code = int(occ_search)
            st.session_state.dialog_open = True
            st.rerun()

    st.subheader("Occupation Map")
    st.markdown("Click any point to view its profile.")
    map_filtered = data.umap_df[data.umap_df["major_code"].apply(lambda x: True)]
    map_merged = map_filtered.merge(
        data.occ_stats_df[["occ_code", "job_post_count"]], left_on="code", right_on="occ_code", how="left"
    )
    map_merged["group"] = map_merged["major_code"].map(config.MAJOR_LABELS)
    fig = px.scatter(
        map_merged, x="x", y="y",
        color="group",
        color_discrete_map={config.MAJOR_LABELS[k]: v for k, v in config.MAJOR_COLORS.items()},
        hover_data={"x": False, "y": False, "code": True, "title": True, "group": True, "job_post_count": True},
        custom_data=["code", "title"],
        labels={"color": "Major Group"}, height=500,
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="white")), selector=dict(mode="markers"))
    # Keep selected/unselected visually identical so a click doesn't darken/dim
    # the map — the click just pops the occupation modal.
    fig.update_traces(
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=1)),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(font=dict(size=10), orientation="h", y=-0.15),
        xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, key="explore_map", on_select=_explore_map_select)
