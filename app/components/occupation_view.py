import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import config, data
import app.api.endpoints as api
import app.utils as u


def top_n_similar(code, n=5):
    row = data.df[data.df["occupation_code"] == code]
    if row.empty:
        return []
    title_text = row.iloc[0]["occupation_title"]
    results = api.search_occupations(title_text, n + 1, occupation_code=code)
    similar = []
    for r in results:
        if int(r["code"]) == code:
            continue
        similar.append((int(r["code"]), r["title"], r["profile_score"]))
        if len(similar) == n:
            break
    return similar


def show_similarity_cards(code):
    st.markdown(f"##### Top 5 Related Occupations {config.SOURCE_CALC}", unsafe_allow_html=True)
    st.markdown("Occupations with similar responsibilities and characteristics.")
    similar = top_n_similar(code, n=5)
    if similar:
        cols = st.columns(5)
        for ci, (scode, stitle, ssim) in enumerate(similar):
            with cols[ci]:
                with st.container(border=True):
                    st.markdown(f"##### {scode}")
                    st.markdown(f"<div style='min-height:3em'>{stitle}</div>", unsafe_allow_html=True)
                    st.metric("Similarity", f"{ssim:.1%}")
                    if st.button("Open", key=f"sim_{scode}"):
                        st.query_params["occupation"] = str(int(scode))
                        st.session_state.selected_code = int(scode)
                        st.session_state.page = "occupation"
                        st.rerun()
    else:
        st.info("No similar occupations found.")


def show_semantic_neighbourhood(code, title):
    st.markdown(f"##### Semantic Neighbourhood {config.SOURCE_CALC}", unsafe_allow_html=True)
    st.markdown("Visualises functional similarity between occupations based on their tasks and responsibilities. Nearby occupations often perform similar work, while connections may span across the map when occupations share similar functions despite belonging to different industries.")
    n_neighbours = 30
    row = data.df[data.df["occupation_code"] == code]
    query = row.iloc[0]["occupation_title"] if not row.empty else title
    results = api.search_occupations(query, n_neighbours + 1, occupation_code=code)
    neighbour_codes = set()
    for r in results:
        c = int(r["code"])
        if c != code:
            neighbour_codes.add(c)
    neighbours = data.umap_df[data.umap_df["code"].isin(neighbour_codes)]
    others = data.umap_df[~data.umap_df["code"].isin(neighbour_codes)]
    row_pt = data.umap_df[data.umap_df["code"] == code].iloc[0]
    st.markdown(f"Semantically closest to **{len(neighbours)}** other occupations.")
    st.markdown("<span style='color:#e74c3c'>★</span> Current &nbsp; <span style='color:#2d5a87'>●</span> Neighbour &nbsp; <span style='color:#c0c0c0'>●</span> Other", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=others["x"], y=others["y"],
        mode="markers",
        marker=dict(color="#e8e8e8", size=4, line=dict(width=0)),
        hoverinfo="skip",
    ))
    line_x, line_y = [], []
    sx, sy = row_pt["x"], row_pt["y"]
    for _, nbr in neighbours.iterrows():
        line_x.extend([sx, nbr["x"], None])
        line_y.extend([sy, nbr["y"], None])
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y,
        mode="lines",
        line=dict(color="rgba(0,0,0,0.07)", width=0.8),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=neighbours["x"], y=neighbours["y"],
        mode="markers",
        marker=dict(color="#2d5a87", size=6, line=dict(width=0.5, color="white")),
        customdata=neighbours[["code", "title"]],
        hovertemplate="%{customdata[1]}<br>%{customdata[0]}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[sx], y=[sy],
        mode="markers",
        marker=dict(color="#e74c3c", size=18, symbol="star", line=dict(width=2, color="white")),
        customdata=[[code, title]],
        hovertemplate="%{customdata[1]}<br>%{customdata[0]}<extra></extra>",
    ))
    all_pts = pd.concat([neighbours, row_pt.to_frame().T])
    x_min, x_max = all_pts["x"].min(), all_pts["x"].max()
    y_min, y_max = all_pts["y"].min(), all_pts["y"].max()
    x_pad = max((x_max - x_min) * 0.3, 0.5)
    y_pad = max((y_max - y_min) * 0.3, 0.5)
    fig.update_layout(
        height=400, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[x_min - x_pad, x_max + x_pad]),
        yaxis=dict(visible=False, range=[y_min - y_pad, y_max + y_pad],
                   scaleanchor="x", scaleratio=1),
        showlegend=False,
    )
    event = st.plotly_chart(fig, key=f"neighbourhood_{code}", on_select="rerun")
    if event and event.selection and event.selection.points:
        point = event.selection.points[0]
        try:
            clicked_code = int(point["customdata"][0])
            if clicked_code != code:
                st.query_params["occupation"] = str(clicked_code)
                st.session_state.selected_code = clicked_code
                st.session_state.page = "occupation"
                st.rerun()
        except (KeyError, IndexError, TypeError):
            pass


def show_occupation_page(row):
    if st.button("⬅ Back"):
        del st.query_params["occupation"]
        st.session_state.page = "home"
        st.session_state.nav_target = {"search": "Explore Occupations", "map": "Explore Occupations", "profile": "My Profile", "job": "Find Jobs"}.get(st.session_state.return_to, "Home")
        st.rerun()
    code = int(row["occupation_code"])
    title = row["occupation_title"]
    titles = [u.proper_case(row[f"{l}_group_title"]) for l in ["major", "sub_major", "minor", "unit"]]
    titles.append(u.proper_case(row["occupation_title"]))
    st.markdown(f"SSOC: {' → '.join(titles)}")
    stat = data.occ_stats_df[data.occ_stats_df["occ_code"] == code]
    job_post_count = int(stat["job_post_count"].values[0]) if not stat.empty else 0

    col_title, col_count = st.columns([3, 1])
    with col_title:
        st.markdown(f"## {code} — {title}")
    with col_count:
        st.metric("Job Posts", f"{job_post_count:,}")
    st.markdown("Explore the responsibilities, skills, career context and related occupations for this role.")

    occ_stat = data.occ_stats_df[data.occ_stats_df["occ_code"] == code]
    if not occ_stat.empty:
        st.markdown(f"<div style='display:flex;align-items:center;gap:4px;margin-bottom:4px'><span style='font-weight:600'>Salary & Experience</span>{config.SOURCE_JOBS}</div>", unsafe_allow_html=True)
        s = occ_stat.iloc[0]
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            st.metric("Avg Salary Min", f"SGD ${s['avg_sal_min']:,.0f}" if pd.notna(s['avg_sal_min']) else "—")
        with mc2:
            st.metric("Avg Salary Max", f"SGD ${s['avg_sal_max']:,.0f}" if pd.notna(s['avg_sal_max']) else "—")
        with mc3:
            st.metric("Median Salary", f"SGD ${s['median_sal']:,.0f}" if pd.notna(s['median_sal']) else "—")
        with mc4:
            st.metric("Avg Experience", f"{s['mean_exp']:.1f} yr" if pd.notna(s['mean_exp']) else "—")
        with mc5:
            st.metric("Median Experience", f"{s['median_exp']:.1f} yr" if pd.notna(s['median_exp']) else "—")

    if st.session_state.profile["preferred_code"] == code:
        st.button("⭐ Preferred Occupation", disabled=True)
    else:
        if st.button("⭐ Set as Preferred Occupation"):
            st.session_state.profile["preferred_code"] = code
            st.session_state.profile["preferred_title"] = title
            st.session_state.jobs_dirty = True
            st.rerun()
    st.markdown(f"##### Overview {config.SOURCE_SSOC}", unsafe_allow_html=True)
    st.write(row["detailed_definitions"])
    st.divider()
    st.markdown(f"##### What They Do {config.SOURCE_SSOC}", unsafe_allow_html=True)
    tasks = u.parse_csv_list(row["tasks"])
    if tasks:
        for t in tasks:
            st.markdown(f"- {t}")
    else:
        st.info("No tasks listed.")
    st.divider()
    occ_rows = data.occ_skills_tools_df[data.occ_skills_tools_df["occ_code"] == code].sort_values("rank")
    if not occ_rows.empty:
        st.markdown(f"##### Skills & Tools in Demand (from job posts) {config.SOURCE_JOBS}", unsafe_allow_html=True)
        col_top, col_var = st.columns([1, 1])
        with col_top:
            top_n = st.number_input("Show top", min_value=1, max_value=len(occ_rows), value=min(20, len(occ_rows)), key=f"topn_{code}")
        with col_var:
            show_variants = st.checkbox("Show variant spellings", key=f"var_{code}")
        for _, r in occ_rows.head(top_n).iterrows():
            st.markdown(f"**{r['main_term']}** ({r['total_count']} job posts)")
            if show_variants and r["total_count"] > 0:
                variants = json.loads(r["variants"])
                if len(variants) > 1:
                    sorted_v = sorted(variants, key=lambda x: -x[1])
                    for vt, vc in sorted_v:
                        if vt != r["main_term"]:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;· {vt} ({vc})")
        st.divider()

    row_text = row["occupation_title"] + " " + (row.get("tasks", "") or "")
    top_jobs = []
    try:
        top_jobs = api.search_jobs(row_text, 5)
    except Exception:
        try:
            top_jobs = api.search_jobs(row_text, 5)
        except Exception:
            st.warning("Could not load representative job posts. Please try again later.")
    if top_jobs:
        st.markdown(f"##### Top Representative Job Posts {config.SOURCE_JOBS}", unsafe_allow_html=True)
        st.markdown("Highest-similarity job postings matched to this occupation.")
        for rank, jr in enumerate(top_jobs, 1):
            nj = u.normalize_job(jr)
            preview = u._strip_html(nj.get("preview") or nj["description"][:200] or "")
            company = nj["company"]
            url = nj["url"]
            sim = nj["profile_score"]
            with st.container(border=True):
                st.markdown(f"**#{rank}** &nbsp; {jr['title']}  —  Similarity: **{sim:.1%}**")
                if company:
                    st.markdown(f"<span style='color:#1b2a4a;font-weight:700'>{company}</span>", unsafe_allow_html=True)
                if preview:
                    st.caption(preview[:200] + ("..." if len(preview) > 200 else ""))
                if url:
                    st.markdown(f"[View original posting]({url})")
    st.divider()

    notes = row["notes"]
    if not pd.isna(notes) and notes.strip("—–- ") not in ("", "—", "–", "-", "nan"):
        st.markdown(f"##### Notes {config.SOURCE_SSOC}", unsafe_allow_html=True)
        st.write(notes)
        st.divider()
    st.markdown(f"##### Common Job Titles {config.SOURCE_SSOC}", unsafe_allow_html=True)
    examples = u.parse_csv_list(row["examples_of_job_classified_here"])
    if examples:
        for e in examples:
            st.markdown(f"- {e}")
    else:
        st.info("None listed.")
    st.divider()
    show_semantic_neighbourhood(code, title)
    st.divider()
    show_similarity_cards(code)
