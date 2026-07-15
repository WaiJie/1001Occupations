from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

from app import config, data
import app.api.endpoints as api
import app.utils as u
from app.components.job_card import render_job_card


def compute_job_matches(resume_weight):
    resume_text = st.session_state.profile["resume"].strip()
    occ_code = st.session_state.profile["preferred_code"]
    jf = st.session_state.job_filters
    filters = {
        "sal_min": jf["sal_min"],
        "sal_max": jf["sal_max"],
        "max_exp": jf["max_exp"],
        "job_status": jf["job_status_filter"],
        "posted_within": jf["posted_within"],
        "source": jf["source"],
        "work_arrangement": jf["work_arrangement"],
    }

    results = api.search_jobs(
        resume_text, 100, occupation_code=occ_code,
        resume_weight=resume_weight, filters=filters,
    )

    enriched = []
    for rank, r in enumerate(results):
        n = u.normalize_job(r)
        snippets = u.make_snippets(resume_text, n["embedding_text"]) if rank < 10 else []
        enriched.append({
            "rank": rank + 1,
            "title": n["title"],
            "company": n["company"],
            "url": n["url"],
            "min_exp": n["min_exp"],
            "sal_min": n["sal_min"],
            "sal_max": n["sal_max"],
            "preview": n["preview"],
            "combined_desc": n["description"],
            "skills": n["skills"],
            "tools": n["tools"],
            "score": n["profile_score"],
            "resume_score": n["resume_score"],
            "occ_score": n["occupation_score"],
            "snippets": snippets,
            "job_status": n["job_status"],
            "posted_date": n["posted_date"],
            "expiry_date": n["expiry_date"],
            "is_closed": n["is_closed"],
        })

    return enriched


def render_job_filters():
    """Job filters live in st.session_state.job_filters. Widgets are wrapped in
    a Streamlit form so editing a single control does NOT trigger an API call;
    results are only recomputed when the user clicks 'Apply Filters'."""
    jf = st.session_state.job_filters
    with st.form("job_filters_form"):
        st.subheader("Filters")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.slider("Max Experience (years)", 0, 20, key="max_exp_years")
        with col_s2:
            st.markdown("**Salary Range (SGD/month)**")
            st.number_input("Min Salary", min_value=0, max_value=100000, step=500, key="sal_min_val")
            st.number_input("Max Salary", min_value=0, max_value=100000, step=500, key="sal_max_val")
        st.selectbox("Job Status", ["Open", "Closed", "All"], key="job_status_sel")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.selectbox(
                "Posted Within", [None, 1, 3, 7, 14, 30, 90],
                format_func=lambda x: "Any time" if x is None else f"Last {x} days",
                key="posted_within_sel",
            )
        with col_f2:
            st.text_input("Source (optional)", key="source_sel")
        st.text_input("Work Arrangement (optional)", key="work_arr_sel")

        submitted = st.form_submit_button("Apply Filters", type="primary", width="stretch")

    if submitted:
        jf["max_exp"] = st.session_state.max_exp_years
        jf["sal_min"] = st.session_state.sal_min_val
        jf["sal_max"] = st.session_state.sal_max_val
        jf["job_status_filter"] = st.session_state.job_status_sel
        jf["posted_within"] = st.session_state.posted_within_sel
        jf["source"] = st.session_state.source_sel
        jf["work_arrangement"] = st.session_state.work_arr_sel
        st.session_state.jobs_dirty = True


def render_jobs():
    st.markdown("Set your filters below, then review matching jobs based on your resume and target occupation from **My Profile**.")
    resume_weight = config.get_mode(st.session_state.profile["career_direction"])[2]

    ready = bool(st.session_state.profile["resume"].strip() and st.session_state.profile["preferred_code"])

    if ready:
        render_job_filters()

    if ready and st.session_state.get("jobs_dirty", False):
        st.session_state.jobs_dirty = False
        with st.spinner("Computing job matches..."):
            st.session_state.job_match_results = compute_job_matches(resume_weight)
        st.session_state.job_page = 1

    if st.session_state.get("job_match_results"):
        all_results = st.session_state.job_match_results
        total_all = len(all_results)
        st.subheader(f"Top {total_all} Matching Jobs")

        label = config.get_mode(st.session_state.profile["career_direction"])[1]
        st.info(f"**Overall Match** computed with **{label}** weighting ({resume_weight*100:.0f}% resume / {(1-resume_weight)*100:.0f}% occupation)")

        per_page = 10
        total_pages = max(1, (len(all_results) + per_page - 1) // per_page)
        st.session_state.job_page = min(st.session_state.job_page, total_pages)
        page = st.session_state.job_page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("◀", disabled=page <= 1, width="stretch"):
                st.session_state.job_page = page - 1
                st.rerun()
        with col_info:
            st.markdown(f"<div style='text-align:center'>Page {page} of {total_pages}</div>", unsafe_allow_html=True)
        with col_next:
            if st.button("▶", disabled=page >= total_pages, width="stretch"):
                st.session_state.job_page = page + 1
                st.rerun()

        resume_text = st.session_state.profile["resume"].strip()
        page_results = list(all_results[start_idx:end_idx])
        missing = [(i, r) for i, r in enumerate(page_results) if not r.get("snippets") and r.get("combined_desc")]
        if missing:
            with st.spinner("Computing evidence..."):
                def compute_snippets(i, r):
                    try:
                        return i, u.make_snippets(resume_text, r["combined_desc"])
                    except Exception:
                        return i, []
                with ThreadPoolExecutor(max_workers=5) as ex:
                    futs = {ex.submit(compute_snippets, i, r): i for i, r in missing}
                    for f in as_completed(futs):
                        i, snippets = f.result()
                        page_results[i]["snippets"] = snippets
        for idx, res in enumerate(page_results):
            render_job_card(res, idx, start_idx, resume_text)

        if st.button("Refresh results", type="primary", width="stretch"):
            with st.status("Finding matching jobs...", expanded=False) as s:
                s.write("Computing...")
                st.session_state.job_match_results = compute_job_matches(resume_weight)
                s.update(label="Done", state="complete")
            st.session_state.job_page = 1
            st.rerun()
    elif ready:
        st.info("Click below to find matching jobs.")
        if st.button("Find matching jobs", type="primary", width="stretch"):
            with st.status("Finding matching jobs...", expanded=False) as s:
                s.write("Computing...")
                st.session_state.job_match_results = compute_job_matches(resume_weight)
                s.update(label="Done", state="complete")
            st.session_state.job_page = 1
            st.rerun()
    else:
        st.info("Set your resume and target occupation in My Profile to find matching jobs.")
