import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINK_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SENTENCE_TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*use_return_dict.*")

import streamlit as st

from app import config, data
from app import session
import app.api.endpoints as api
from app.components.occupation_view import show_occupation_page
from app.pages import home, profile, explore, jobs, byo

st.set_page_config(page_title="1001 Occupations", page_icon="💼", layout="wide")
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: stretch !important;
}
div[data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
    align-self: stretch !important;
    flex: 1 1 0px !important;
}
div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}
div[data-testid="column"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderBox"] {
    flex: 1 !important;
    padding: 0.5rem 1rem 1rem 1rem;
}
div[role="radiogroup"] {
    background: #e8f0fe !important;
    border-radius: 10px !important;
    padding: 4px !important;
}
div[role="radiogroup"] label {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 0.4rem 1rem !important;
    border-radius: 8px !important;
}
div[role="radiogroup"] label[aria-checked="true"] {
    background: #1a73e8 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

session.init_session()

params = st.query_params
if "occupation" in params:
    try:
        st.session_state.page = "occupation"
        st.session_state.selected_code = int(params["occupation"])
    except ValueError:
        pass

# ── Branding & Navigation ─────────────────────────────
col_title, col_nav = st.columns([2, 3])
with col_title:
    st.markdown("<h1 style='font-size:2rem; font-weight:900; margin:0; padding:0; line-height:1;'>1001 Occupations</h1><span style='color:#888; font-size:0.85rem;'>Explore Singapore's SSOC 2024 framework</span>", unsafe_allow_html=True)
with col_nav:
    if st.session_state.page != "occupation":
        if st.session_state.nav_target:
            st.session_state.nav_tab = st.session_state.nav_target
            st.session_state.nav_target = None
        st.segmented_control("Navigation", ["Home", "My Profile", "Explore Occupations", "Find Jobs", "Bring Your Own Job"], key="nav_tab", label_visibility="collapsed")

st.divider()

if st.session_state.page == "occupation":
    row = data.df[data.df["occupation_code"] == st.session_state.selected_code]
    if not row.empty:
        show_occupation_page(row.iloc[0])
else:
    if st.session_state.nav_tab != st.session_state.prev_tab:
        if st.session_state.nav_tab == "My Profile":
            st.session_state.career_dir = st.session_state.profile["career_direction"]
        elif st.session_state.nav_tab == "Find Jobs":
            st.session_state.max_exp_years = st.session_state.job_filters["max_exp"]
            st.session_state.sal_min_val = st.session_state.job_filters["sal_min"]
            st.session_state.sal_max_val = st.session_state.job_filters["sal_max"]
            st.session_state.posted_within_sel = st.session_state.job_filters["posted_within"]
            st.session_state.source_sel = st.session_state.job_filters["source"]
            st.session_state.work_arr_sel = st.session_state.job_filters["work_arrangement"]
        st.session_state.prev_tab = st.session_state.nav_tab

    tab = st.session_state.nav_tab
    if tab == "Home":
        home.render_home()
    elif tab == "My Profile":
        profile.render_profile()
    elif tab == "Explore Occupations":
        explore.render_explore()
    elif tab == "Find Jobs":
        jobs.render_jobs()
    elif tab == "Bring Your Own Job":
        byo.render_byo()
