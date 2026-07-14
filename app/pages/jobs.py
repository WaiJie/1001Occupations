from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

from app import config, data
import app.api.endpoints as api
import app.utils as u
from app.components.job_card import render_job_card


def compute_job_matches(resume_weight):
    resume_text = st.session_state.profile["resume"].strip()
    occ_code = st.session_state.profile["preferred_code"]
    filters = {
        "sal_min": st.session_state.profile["sal_min"],
        "sal_max": st.session_state.profile["sal_max"],
        "max_exp": st.session_state.profile["max_exp"],
        "job_status": st.session_state.profile["job_status_filter"],
        "posted_within": st.session_state.profile["posted_within"],
        "source": st.session_state.profile["source"],
        "work_arrangement": st.session_state.profile["work_arrangement"],
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


def render_jobs():
    st.markdown("Recommendations based on your profile settings under the **My Profile** tab.")
    resume_weight = config.get_mode(st.session_state.profile["career_direction"])[2]

    ready = bool(st.session_state.profile["resume"].strip() and st.session_state.profile["preferred_code"])

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
