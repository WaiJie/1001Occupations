import streamlit as st

from app import config
import app.api.endpoints as api
import app.utils as u
from app.components.job_card import render_byo_card


def _resume_texts():
    texts = []
    if st.session_state.profile["resume"].strip():
        texts.append(st.session_state.profile["resume"].strip())
    if st.session_state.profile["resume2"].strip():
        texts.append(st.session_state.profile["resume2"].strip())
    return texts


def _aggregate(scores_list):
    """Pick the best resume match (highest profile_score) as the headline."""
    best = max(scores_list, key=lambda s: s.get("profile_score", 0))
    return {
        "score": best.get("profile_score", 0),
        "resume_score": best.get("resume_score", 0),
        "occ_score": best.get("occupation_score", 0),
        "scores": scores_list,
    }


def compute_byo_match(description, resume_weight):
    resume_texts = _resume_texts()
    occ_code = st.session_state.profile["preferred_code"]

    batch = api.match_profiles_batch(resume_texts, description.strip(), occ_code, resume_weight)

    snippets = u.make_snippets(st.session_state.profile["resume"].strip(), description)

    try:
        occ_results = api.search_occupations(description.strip()[:1000], 5)
        similar = []
        for r in occ_results:
            similar.append({"code": int(r["code"]), "title": r["title"], "score": r["profile_score"]})
    except Exception:
        similar = []

    agg = _aggregate(batch if isinstance(batch, list) else [batch])
    return {
        "description": description.strip(),
        "score": agg["score"],
        "resume_score": agg["resume_score"],
        "occ_score": agg["occ_score"],
        "scores": agg["scores"],
        "snippets": snippets,
        "similar_occ": similar,
    }


def render_byo():
    st.markdown("""
**Bring Your Own Job** lets you evaluate job descriptions from any source — LinkedIn, company career pages, job boards — against your profile.

Paste a job description below and we'll score it based on your **resume(s)**, **target occupation**, and **career direction** set in **My Profile**. If you added a second resume, each job is scored against both and the best match is shown.
""")

    if st.session_state.get("byo_clear"):
        st.session_state.byo_input = ""
        st.session_state.byo_clear = False

    st.text_area(
        "Job Description",
        key="byo_input",
        placeholder="Paste the full job description here...",
        height=200,
        label_visibility="collapsed",
    )

    ready = bool(st.session_state.profile["resume"].strip() and st.session_state.profile["preferred_code"])

    if not ready:
        st.info("Set your resume and target occupation in **My Profile** before evaluating jobs.")
    else:
        resume_weight = config.get_mode(st.session_state.profile["career_direction"])[2]
        label = config.get_mode(st.session_state.profile["career_direction"])[1]

        if st.session_state.byo_input.strip():
            if st.button("Add to List", type="primary", width="stretch"):
                with st.status("Analysing...", expanded=False) as s:
                    s.write("Computing...")
                    result = compute_byo_match(st.session_state.byo_input, resume_weight)
                    st.session_state.byo_jobs.append(result)
                    st.session_state.byo_clear = True
                    s.update(label="Done", state="complete")
                st.rerun()

        if st.session_state.byo_jobs:
            st.divider()

            sorted_jobs = sorted(st.session_state.byo_jobs, key=lambda x: x["score"], reverse=True)
            for rank, job in enumerate(sorted_jobs, 1):
                job["_rank"] = rank
                render_byo_card(job, label, resume_weight)

            col_refresh, col_clear = st.columns(2)
            with col_refresh:
                if st.button("Re-analyse All", type="primary", width="stretch"):
                    with st.status("Re-analysing all jobs...") as s:
                        new_jobs = []
                        for i, job in enumerate(st.session_state.byo_jobs):
                            s.write(f"Analysing {i + 1} of {len(st.session_state.byo_jobs)}...")
                            new_jobs.append(compute_byo_match(job["description"], resume_weight))
                        st.session_state.byo_jobs = new_jobs
                        s.update(label="Done", state="complete")
                    st.rerun()
            with col_clear:
                if st.button("Clear All", width="stretch"):
                    st.session_state.byo_jobs = []
                    st.rerun()
