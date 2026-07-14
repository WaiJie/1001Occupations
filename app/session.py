import streamlit as st


def init_session():
    """Initialise all session_state defaults. Call once at startup."""
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "selected_code" not in st.session_state:
        st.session_state.selected_code = None
    if "match_results" not in st.session_state:
        st.session_state.match_results = None
    if "return_to" not in st.session_state:
        st.session_state.return_to = "Home"
    if "nav_tab" not in st.session_state:
        st.session_state.nav_tab = "Home"
    if "resume_input" not in st.session_state:
        st.session_state.resume_input = ""
    if "job_match_results" not in st.session_state:
        st.session_state.job_match_results = None
    if "job_page" not in st.session_state:
        st.session_state.job_page = 1
    if "nav_target" not in st.session_state:
        st.session_state.nav_target = None
    if "profile" not in st.session_state:
        st.session_state.profile = {
            "resume": "",
            "resume2": "",
            "preferred_code": None,
            "preferred_title": None,
            "career_direction": 0,
            "max_exp": 20,
            "sal_min": 0,
            "sal_max": 50000,
            "job_status_filter": "Open",
            "posted_within": None,
            "source": "",
            "work_arrangement": "",
        }
        # Backfill any keys missing from a previously-saved profile dict.
        _profile_defaults = {
            "resume": "", "resume2": "",
            "preferred_code": None, "preferred_title": None,
            "career_direction": 0, "max_exp": 20,
            "sal_min": 0, "sal_max": 50000, "job_status_filter": "Open",
            "posted_within": None, "source": "", "work_arrangement": "",
        }
        for k, v in _profile_defaults.items():
            st.session_state.profile.setdefault(k, v)
    if "job_status_filter" not in st.session_state.profile:
        st.session_state.profile["job_status_filter"] = "Open"
    if "career_dir" not in st.session_state:
        st.session_state.career_dir = st.session_state.profile["career_direction"]
    if "max_exp_years" not in st.session_state:
        st.session_state.max_exp_years = st.session_state.profile["max_exp"]
    if "sal_min_val" not in st.session_state:
        st.session_state.sal_min_val = st.session_state.profile["sal_min"]
    if "sal_max_val" not in st.session_state:
        st.session_state.sal_max_val = st.session_state.profile["sal_max"]
    if "posted_within_sel" not in st.session_state:
        st.session_state.posted_within_sel = st.session_state.profile["posted_within"]
    if "source_sel" not in st.session_state:
        st.session_state.source_sel = st.session_state.profile["source"]
    if "work_arr_sel" not in st.session_state:
        st.session_state.work_arr_sel = st.session_state.profile["work_arrangement"]
    if "jobs_dirty" not in st.session_state:
        st.session_state.jobs_dirty = True
    if "prev_tab" not in st.session_state:
        st.session_state.prev_tab = "Home"
    if "byo_input" not in st.session_state:
        st.session_state.byo_input = ""
    if "byo_jobs" not in st.session_state:
        st.session_state.byo_jobs = []
    if "byo_page" not in st.session_state:
        st.session_state.byo_page = 1
