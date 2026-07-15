import streamlit as st

from app import config


def render_home():
    col_img_left, col_img, col_img_right = st.columns([1, 2, 1])
    with col_img:
        st.image("assets/Landing_page.png", width="stretch")
    st.markdown("<p style='text-align:center; font-size:1.2rem; color:#555;'>Upload your resume to discover occupations and find matching jobs.</p>", unsafe_allow_html=True)

    btn_label = "Go to My Profile" if st.session_state.profile["resume"].strip() else "Upload Resume"
    if st.button(btn_label, type="primary", width="stretch"):
        st.session_state.nav_target = "My Profile"
        st.rerun()

    st.divider()

    if st.session_state.profile["resume"].strip() and st.session_state.profile["preferred_code"]:
        st.subheader("Continue")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Target Occupation**")
            st.info(f"**{st.session_state.profile['preferred_code']}** — {st.session_state.profile['preferred_title']}")
        with c2:
            st.markdown("**Resume**")
            st.info("Uploaded")
        with c3:
            label = config.get_mode(st.session_state.profile["career_direction"])[1]
            st.markdown("**Career Direction**")
            st.info(label)
        if st.button("Continue Finding Jobs", type="primary", width="stretch"):
            st.session_state.nav_target = "Find Jobs"
            st.rerun()
