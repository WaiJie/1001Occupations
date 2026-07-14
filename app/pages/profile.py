import streamlit as st

from app import config
import app.api.endpoints as api


def render_profile():
    st.subheader("Resume")
    if st.session_state.profile["resume"] and not st.session_state.resume_input:
        st.session_state.resume_input = st.session_state.profile["resume"]
    if st.session_state.profile["resume2"] and not st.session_state.get("resume2_input"):
        st.session_state.resume2_input = st.session_state.profile["resume2"]

    st.text_area(
        "Paste your resume, job description, or career summary",
        key="resume_input",
        placeholder="e.g. Experienced data scientist with 5 years in machine learning, Python, and statistical modelling...",
        height=200,
    )
    st.text_area(
        "Resume 2 (optional) — used to compare Bring-Your-Own jobs against a second profile",
        key="resume2_input",
        placeholder="e.g. A second resume or alternative career summary to compare against...",
        height=200,
    )

    if st.button("Save Resume", type="primary"):
        st.session_state.profile["resume"] = st.session_state.resume_input
        st.session_state.profile["resume2"] = st.session_state.resume2_input
        st.session_state.jobs_dirty = True
        if st.session_state.profile["resume"].strip():
            with st.spinner("Matching your resume to occupations..."):
                api_results = api.search_occupations(
                    st.session_state.profile["resume"].strip(), 6,
                )
                st.session_state.match_results = []
                for r in api_results:
                    st.session_state.match_results.append({
                        "rank": len(st.session_state.match_results) + 1,
                        "code": int(r["code"]),
                        "title": r["title"],
                        "score": r["profile_score"],
                    })
                if not st.session_state.profile["preferred_code"] and st.session_state.match_results:
                    top_res = st.session_state.match_results[0]
                    st.session_state.profile["preferred_code"] = top_res["code"]
                    st.session_state.profile["preferred_title"] = top_res["title"]
                    st.session_state._just_auto_selected = True
                    st.session_state.jobs_dirty = True

    if st.session_state.profile["resume"].strip():
        st.caption(f"Resume 1: {len(st.session_state.profile['resume'].split())} words ✓ Uploaded")
        if st.session_state.profile["resume2"].strip():
            st.caption(f"Resume 2: {len(st.session_state.profile['resume2'].split())} words ✓ Uploaded")
        st.divider()

        st.subheader("Top Occupation Matches")
        if st.session_state.get("_just_auto_selected"):
            st.info(
                f"We auto-selected your top match "
                f"**{st.session_state.profile['preferred_code']} — {st.session_state.profile['preferred_title']}** "
                f"as your target occupation. Change it below if you prefer another."
            )
            st.session_state._just_auto_selected = False
        if st.session_state.match_results:
            for row_i in range(0, len(st.session_state.match_results), 3):
                cols = st.columns(3)
                for ci, res in enumerate(st.session_state.match_results[row_i:row_i + 3]):
                    with cols[ci]:
                        with st.container(border=True):
                            st.markdown(f"##### {res['title']}")
                            st.markdown(f"Code: {res['code']}  \nMatch: {res['score']:.0%}")
                            sel = st.session_state.profile["preferred_code"] == res['code']
                            col_sel, col_view = st.columns(2)
                            with col_sel:
                                if st.button("Select", key=f"sel_{res['code']}", width="stretch", disabled=sel,
                                             type="primary" if sel else "secondary"):
                                    st.session_state.profile["preferred_code"] = res['code']
                                    st.session_state.profile["preferred_title"] = res['title']
                                    st.session_state.jobs_dirty = True
                                    st.rerun()
                            with col_view:
                                if st.button("View", key=f"view_{res['code']}", width="stretch"):
                                    st.query_params["occupation"] = str(int(res['code']))
                                    st.session_state.selected_code = int(res['code'])
                                    st.session_state.page = "occupation"
                                    st.rerun()
        else:
            st.info("Save your resume to see occupation matches.")

        st.markdown("Want to try something different?")
        if st.button("Browse occupations", type="primary", width="stretch"):
            st.session_state.nav_target = "Explore Occupations"
            st.rerun()

        st.divider()
        col_target, col_career = st.columns([1, 1])
        with col_target:
            st.subheader("Target Occupation")
            if st.session_state.profile["preferred_code"]:
                st.markdown(f"**{st.session_state.profile['preferred_code']}** — {st.session_state.profile['preferred_title']}")
                col_clear, col_view = st.columns(2)
                with col_clear:
                    if st.button("Clear target occupation"):
                        st.session_state.profile["preferred_code"] = None
                        st.session_state.profile["preferred_title"] = None
                        st.session_state.jobs_dirty = True
                        st.rerun()
                with col_view:
                    if st.button("View occupation", use_container_width=True):
                        st.query_params["occupation"] = str(int(st.session_state.profile["preferred_code"]))
                        st.session_state.selected_code = int(st.session_state.profile["preferred_code"])
                        st.session_state.page = "occupation"
                        st.rerun()
            else:
                st.warning("Select a target occupation from your matches above.")
        with col_career:
            st.subheader("Career Direction")
            st.caption("Slide towards **Career Transition** to prioritise your target occupation over your current experience when finding matching jobs.")
            help_lines = ["Controls how much weight to put on your current experience vs your target occupation."]
            for _, lbl, rw in config.MODES:
                help_lines.append(f"  - {lbl}: {rw*100:.0f}% resume / {(1-rw)*100:.0f}% occupation")
            lbl_left, slider, lbl_right = st.columns([1, 4, 1])
            with lbl_left:
                st.markdown("<div style='text-align:center;padding-top:4px;font-size:0.85rem;color:#666'>Exact Match</div>", unsafe_allow_html=True)
            with slider:
                st.select_slider(
                    "Career Direction",
                    options=[m[0] for m in config.MODES],
                    format_func=lambda x: config.get_mode(x)[1],
                    key="career_dir",
                    label_visibility="collapsed",
                    help="\n".join(help_lines),
                )
            with lbl_right:
                st.markdown("<div style='text-align:center;padding-top:4px;font-size:0.85rem;color:#666'>Career Transition</div>", unsafe_allow_html=True)
            if st.session_state.profile["career_direction"] != st.session_state.career_dir:
                st.session_state.jobs_dirty = True
            st.session_state.profile["career_direction"] = st.session_state.career_dir

