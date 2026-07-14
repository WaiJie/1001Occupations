import streamlit as st

from app import config
import app.api.endpoints as api
import app.utils as u


def render_job_card(res, idx, start_idx, resume_text):
    with st.container(border=True):
        is_closed = res.get("is_closed", False)
        status = res.get("job_status", "")
        if is_closed:
            display_status = "Closed"
            color = "#e74c3c"
        elif status:
            display_status = status
            color = "#27ae60" if status.lower() in ("open", "re-open") else "#e74c3c"
        else:
            display_status = ""
            color = ""
        badge = f" <span style='background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.75rem;vertical-align:middle'>{display_status}</span>" if display_status else ""
        col_title, col_btns = st.columns([4, 1])
        with col_title:
            st.markdown(f"**#{start_idx + idx + 1}** &nbsp; {res['title']}{badge}", unsafe_allow_html=True)
            if res["company"]:
                st.markdown(f"<span style='color:#1b2a4a;font-weight:700;font-size:0.95rem'>{res['company']}</span>", unsafe_allow_html=True)
            date_parts = []
            if res.get("posted_date"):
                date_parts.append(f"Posted: {res['posted_date']}")
            if res.get("expiry_date"):
                date_parts.append(f"Expires: {res['expiry_date']}")
            if date_parts:
                st.markdown(f"<span style='color:#888;font-size:0.85rem'>{' &nbsp;·&nbsp; '.join(date_parts)}</span>", unsafe_allow_html=True)
            info_parts = []
            if res["min_exp"] is not None:
                info_parts.append(f"📋 {int(res['min_exp'])} yr{'s' if res['min_exp'] != 1 else ''} exp")
            if res["sal_min"] is not None and res["sal_max"] is not None:
                info_parts.append(f"💰 SGD {int(res['sal_min']):,} – {int(res['sal_max']):,}")
            if info_parts:
                st.markdown(" &nbsp;|&nbsp; ".join(info_parts))
            st.markdown(f"🎯 Match: **{res['score']:.1%}** &nbsp; Resume: {res['resume_score']:.1%} &nbsp; Occupation: {res['occ_score']:.1%}")
        with col_btns:
            if res.get("url"):
                state_key = f"explain_result_{id(res)}"
                st.link_button("🚀 Apply", res["url"], type="primary", use_container_width=True)
                if st.button("✨ Explain", key=f"explain_btn_{id(res)}", use_container_width=True):
                    with st.spinner("Analysing..."):
                        desc = res.get("combined_desc", "")[:2000]
                        prompt = f"In a few sentences, explain why this job is a good match. State which skills align and note any obvious gaps.\n\nResume:\n{resume_text[:2000]}\n\nJob Description:\n{desc}"
                        explanation, used_fallback = api.explain_match(prompt)
                        if used_fallback:
                            st.toast("External LLM limit reached. Using fallback model.", icon="⚠️")
                        if isinstance(explanation, dict):
                            explanation = explanation.get("response", str(explanation))
                        st.session_state[state_key] = str(explanation)
                    st.rerun()
        if res.get("url") and st.session_state.get(f"explain_result_{id(res)}"):
            st.info(st.session_state[f"explain_result_{id(res)}"])
        if res.get("skills") or res.get("tools"):
            seen = set()
            combined = []
            for item in (res.get("skills") or [])[:5] + (res.get("tools") or [])[:5]:
                if item not in seen:
                    seen.add(item)
                    combined.append(item)
            st.markdown(f"**Skills & Tools:** <span style='color:#e67e22'>{' · '.join(combined)}</span>", unsafe_allow_html=True)
        if res.get("snippets"):
            st.markdown("**Evidence from job description:**")
            for sent, _ in res["snippets"]:
                st.markdown(f"- {sent}")


def render_byo_card(job, resume_weight_label, resume_weight):
    with st.container(border=True):
        col_title, col_btns = st.columns([4, 1])
        with col_title:
            st.markdown(f"**#{job['_rank']}** &nbsp; Match: **{job['score']:.1%}**")
            scores = job.get("scores") or [{
                "profile_score": job["score"],
                "resume_score": job["resume_score"],
                "occupation_score": job["occ_score"],
            }]
            if len(scores) > 1:
                parts = [f"R{i}: {sc['profile_score']:.1%}" for i, sc in enumerate(scores, 1)]
                st.markdown("Per-resume: " + " &nbsp;·&nbsp; ".join(parts))
            st.markdown(f"Resume: {job['resume_score']:.1%} &nbsp; Occupation: {job['occ_score']:.1%} &nbsp; Weighting: {resume_weight_label} ({resume_weight*100:.0f}% / {(1-resume_weight)*100:.0f}%)")
            preview = u._strip_html(job["description"][:200] + ("..." if len(job["description"]) > 200 else ""))
            st.caption(preview)
        with col_btns:
            state_key = f"byo_explain_{id(job)}"
            if st.button("✨ Explain", key=state_key + "_btn", use_container_width=True):
                with st.spinner("Analysing..."):
                    desc = job["description"][:2000]
                    prompt = f"In a few sentences, explain why this job is a good match. State which skills align and note any obvious gaps.\n\nResume:\n{st.session_state.profile['resume'].strip()[:2000]}\n\nJob Description:\n{desc}"
                    explanation, used_fallback = api.explain_match(prompt)
                    if used_fallback:
                        st.toast("External LLM limit reached. Using fallback model.", icon="⚠️")
                    if isinstance(explanation, dict):
                        explanation = explanation.get("response", str(explanation))
                    st.session_state[state_key] = str(explanation)
                st.rerun()
            if st.button("✕ Remove", key=f"byo_del_{id(job)}", use_container_width=True):
                st.session_state.byo_jobs = [j for j in st.session_state.byo_jobs if id(j) != id(job)]
                st.rerun()
        if st.session_state.get(state_key):
            st.info(st.session_state[state_key])
        if job.get("similar_occ"):
            best = job["similar_occ"][0]
            st.markdown(f"**Closest occupation:** {best['code']} — {best['title']} &nbsp; <span style='color:#888'>({best['score']:.1%} similarity)</span>", unsafe_allow_html=True)
        if job.get("snippets"):
            st.markdown("**Evidence from job description:**")
            for sent, _ in job["snippets"]:
                st.markdown(f"- {sent}")
