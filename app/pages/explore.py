import plotly.express as px
import streamlit as st

from app import config, data


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
        if st.button("View", disabled=not occ_search, use_container_width=True):
            code = int(occ_search)
            st.query_params["occupation"] = str(code)
            st.session_state.selected_code = code
            st.session_state.page = "occupation"
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
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(font=dict(size=10), orientation="h", y=-0.15),
        xaxis=dict(visible=False), yaxis=dict(visible=False))
    event = st.plotly_chart(fig, key="explore_map", on_select="rerun")
    if event and event.selection and event.selection.points:
        point = event.selection.points[0]
        try:
            code = int(point["customdata"][0])
            st.query_params["occupation"] = str(code)
            st.session_state.selected_code = code
            st.session_state.page = "occupation"
            st.rerun()
        except (KeyError, IndexError, TypeError):
            pass
