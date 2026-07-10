import os, warnings, json, re, ast
from datetime import date
os.environ["HF_HUB_DISABLE_SYMLINK_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SENTENCE_TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*use_return_dict.*")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from gradio_client import Client

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

DATA_PATH = "data/ssoc2024_flat_wide.xlsx"
COORDS_PATH = "data/ssoc_umap_coords.csv"
META_PATH = "data/ssoc_metadata.csv"
OCC_SKILLS_TOOLS_PATH = "data/occupation_skills_tools.csv"

MAJOR_COLORS = {
    1: "#e41a1c", 2: "#377eb8", 3: "#4daf4a", 4: "#984ea3",
    5: "#ff7f00", 6: "#fdcf27", 7: "#a65628", 8: "#f781bf",
    9: "#999999",
}

MAJOR_LABELS = {
    1: "1 - LEGISLATORS, SENIOR OFFICIALS AND MANAGERS",
    2: "2 - PROFESSIONALS",
    3: "3 - ASSOCIATE PROFESSIONALS AND TECHNICIANS",
    4: "4 - CLERICAL SUPPORT WORKERS",
    5: "5 - SERVICES AND SALES WORKERS",
    6: "6 - AGRICULTURAL AND FISHERY WORKERS",
    7: "7 - CRAFTSMEN AND RELATED TRADES WORKERS",
    8: "8 - PLANT AND MACHINE OPERATORS AND ASSEMBLERS",
    9: "9 - CLEANERS, LABOURERS AND RELATED WORKERS",
}

# ── Load ─────────────────────────────────────────────────
@st.cache_data
def load_occupations():
    df = pd.read_excel(DATA_PATH)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    return df

@st.cache_data
def load_coords():
    return pd.read_csv(COORDS_PATH)

@st.cache_data
def load_occupation_metadata():
    return pd.read_csv(META_PATH)

@st.cache_data
def load_occupation_skills_tools():
    return pd.read_csv(OCC_SKILLS_TOOLS_PATH)

@st.cache_data
def load_occupation_stats():
    return pd.read_csv("data/occupation_stats.csv")

df = load_occupations()
umap_df = load_coords()
emb_meta = load_occupation_metadata()
occ_skills_tools_df = load_occupation_skills_tools()
occ_stats_df = load_occupation_stats()

def global_top_skills(n=50, major_filter="All"):
    data = occ_skills_tools_df
    major_map = {
        "All": lambda x: True,
        "PMET": lambda x: x in (1, 2, 3),
        "Non-PMET": lambda x: x not in (1, 2, 3),
        "1 - Managers": lambda x: x == 1,
        "2 - Professionals": lambda x: x == 2,
        "3 - Associate Professionals": lambda x: x == 3,
    }
    fn = major_map.get(major_filter, lambda x: True)
    occ_codes = emb_meta[emb_meta["major_code"].apply(fn)]["code"]
    data = data[data["occ_code"].isin(occ_codes)]
    return (
        data.groupby("main_term")["total_count"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )

@st.cache_resource
def get_api_client():
    token = st.secrets["HF_TOKEN"]
    return Client("Wjchua/1001Occupations", token=token)

# ── Session state ────────────────────────────────────────
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
        "preferred_code": None,
        "preferred_title": None,
        "career_direction": 0,
        "max_exp": 20,
        "sal_min": 0,
        "sal_max": 50000,
        "job_status_filter": "Open",
    }
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


params = st.query_params
if "occupation" in params:
    try:
        st.session_state.page = "occupation"
        st.session_state.selected_code = int(params["occupation"])
    except ValueError:
        pass

# ── Helpers ──────────────────────────────────────────────
def proper_case(s):
    import re
    return re.sub(r'\([^)]*\)|[^\s(]+', lambda m: m.group(0) if m.group(0).startswith("(") else m.group(0).title(), s)

def parse_csv_list(val):
    if pd.isna(val):
        return []
    import csv
    from io import StringIO
    reader = csv.reader(StringIO(str(val)), skipinitialspace=True)
    return [s.strip('" ') for s in next(reader, []) if s.strip()]

def api_semantic_search(text, dataset="occupations", top_k=5, occupation_code=None, resume_weight=1.0):
    client = get_api_client()
    return client.predict(
        text, dataset, top_k,
        occupation_code if occupation_code is not None else None,
        resume_weight,
        api_name="/semantic_search",
    )

def api_profile_match(resume_text, job_description, occupation_code=None, resume_weight=1.0):
    client = get_api_client()
    return client.predict(
        resume_text, job_description,
        occupation_code if occupation_code is not None else None,
        resume_weight,
        api_name="/profile_match",
    )

def api_retrieve_evidence(profile_vec, combined_desc, top_k=3):
    client = get_api_client()
    return client.predict(
        profile_vec, combined_desc, float(top_k),
        api_name="/retrieve_evidence",
    )

def api_call_external_llm(message, system_prompt="You are a helpful career assistant.", temperature=0.2, max_new_tokens=512):
    client = get_api_client()
    return client.predict(
        message, system_prompt, temperature, max_new_tokens,
        api_name="/call_external_llm",
    )

def api_call_llm(message, system_prompt="You are a helpful career assistant.", temperature=0.2, max_new_tokens=512):
    client = get_api_client()
    return client.predict(
        message, system_prompt, temperature, max_new_tokens,
        api_name="/call_llm",
    )

def explain_match(prompt):
    try:
        return api_call_external_llm(prompt), False
    except Exception:
        try:
            return api_call_llm(prompt), True
        except Exception:
            return None, False

def top_n_similar(code, n=5):
    row = df[df["occupation_code"] == code]
    if row.empty:
        return []
    title_text = row.iloc[0]["occupation_title"]
    results = api_semantic_search(title_text, "occupations", n + 1, occupation_code=code)
    similar = []
    for r in results:
        if int(r["code"]) == code:
            continue
        similar.append((int(r["code"]), r["title"], r["profile_score"]))
        if len(similar) == n:
            break
    return similar

def show_similarity_cards(code):
    st.markdown(f"##### Top 5 Related Occupations {SOURCE_CALC}", unsafe_allow_html=True)
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
    st.markdown(f"##### Semantic Neighbourhood {SOURCE_CALC}", unsafe_allow_html=True)
    st.markdown("Visualises functional similarity between occupations based on their tasks and responsibilities. Nearby occupations often perform similar work, while connections may span across the map when occupations share similar functions despite belonging to different industries.")
    n_neighbours = 30
    row = df[df["occupation_code"] == code]
    query = row.iloc[0]["occupation_title"] if not row.empty else title
    results = api_semantic_search(query, "occupations", n_neighbours + 1, occupation_code=code)
    neighbour_codes = set()
    for r in results:
        c = int(r["code"])
        if c != code:
            neighbour_codes.add(c)
    neighbours = umap_df[umap_df["code"].isin(neighbour_codes)]
    others = umap_df[~umap_df["code"].isin(neighbour_codes)]
    row_pt = umap_df[umap_df["code"] == code].iloc[0]
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

SOURCE_SSOC = "<span style='font-size:0.7rem;color:#e67e22;border:1px solid #e67e22;border-radius:3px;padding:0 6px;margin-left:8px;white-space:nowrap'>SSOC 2024</span>"
SOURCE_JOBS = "<span style='font-size:0.7rem;color:#1a73e8;border:1px solid #1a73e8;border-radius:3px;padding:0 6px;margin-left:8px;white-space:nowrap'>Job Posts</span>"
SOURCE_CALC = "<span style='font-size:0.7rem;color:#888;border:1px solid #888;border-radius:3px;padding:0 6px;margin-left:8px;white-space:nowrap'>Calculated</span>"

def show_occupation_page(row):
    if st.button("⬅ Back"):
        del st.query_params["occupation"]
        st.session_state.page = "home"
        st.session_state.nav_target = {"search": "Explore Occupations", "map": "Explore Occupations", "profile": "My Profile", "job": "Find Jobs"}.get(st.session_state.return_to, "Home")
        st.rerun()
    code = int(row["occupation_code"])
    title = row["occupation_title"]
    titles = [proper_case(row[f"{l}_group_title"]) for l in ["major", "sub_major", "minor", "unit"]]
    titles.append(proper_case(row["occupation_title"]))
    st.markdown(f"SSOC: {' → '.join(titles)}")
    stat = occ_stats_df[occ_stats_df["occ_code"] == code]
    job_post_count = int(stat["job_post_count"].values[0]) if not stat.empty else 0

    col_title, col_count = st.columns([3, 1])
    with col_title:
        st.markdown(f"## {code} — {title}")
    with col_count:
        st.metric("Job Posts", f"{job_post_count:,}")
    st.markdown("Explore the responsibilities, skills, career context and related occupations for this role.")

    occ_stat = occ_stats_df[occ_stats_df["occ_code"] == code]
    if not occ_stat.empty:
        st.markdown(f"<div style='display:flex;align-items:center;gap:4px;margin-bottom:4px'><span style='font-weight:600'>Salary & Experience</span>{SOURCE_JOBS}</div>", unsafe_allow_html=True)
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
    st.markdown(f"##### Overview {SOURCE_SSOC}", unsafe_allow_html=True)
    st.write(row["detailed_definitions"])
    st.divider()
    st.markdown(f"##### What They Do {SOURCE_SSOC}", unsafe_allow_html=True)
    tasks = parse_csv_list(row["tasks"])
    if tasks:
        for t in tasks:
            st.markdown(f"- {t}")
    else:
        st.info("No tasks listed.")
    st.divider()
    occ_rows = occ_skills_tools_df[occ_skills_tools_df["occ_code"] == code].sort_values("rank")
    if not occ_rows.empty:
        st.markdown(f"##### Skills & Tools in Demand (from job posts) {SOURCE_JOBS}", unsafe_allow_html=True)
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
        top_jobs = api_semantic_search(row_text, "jobs", 5)
    except Exception:
        try:
            top_jobs = api_semantic_search(row_text, "jobs", 5)
        except Exception:
            st.warning("Could not load representative job posts. Please try again later.")
    if top_jobs:
        st.markdown(f"##### Top Representative Job Posts {SOURCE_JOBS}", unsafe_allow_html=True)
        st.markdown("Highest-similarity job postings matched to this occupation.")
        for rank, jr in enumerate(top_jobs, 1):
            nj = normalize_job(jr)
            preview = _strip_html(nj.get("preview") or nj["description"][:200] or "")
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
        st.markdown(f"##### Notes {SOURCE_SSOC}", unsafe_allow_html=True)
        st.write(notes)
        st.divider()
    st.markdown(f"##### Common Job Titles {SOURCE_SSOC}", unsafe_allow_html=True)
    examples = parse_csv_list(row["examples_of_job_classified_here"])
    if examples:
        for e in examples:
            st.markdown(f"- {e}")
    else:
        st.info("None listed.")
    st.divider()
    show_semantic_neighbourhood(code, title)
    st.divider()
    show_similarity_cards(code)

# ── Job matching ─────────────────────────────────────────
def _strip_html(text):
    text = re.sub(r'<[^>]+>', ' ', str(text))
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&nbsp;', ' ')
    return re.sub(r'\s+', ' ', text).strip()

def make_snippets(resume_text, combined_desc, top_k=3):
    resume_words = set(re.findall(r'\w+', resume_text.lower()))
    clean_desc = _strip_html(combined_desc)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_desc) if len(s.strip()) > 20]
    if not sents:
        sents = [clean_desc[:200]]
    scored = []
    for s in sents:
        sw = set(re.findall(r'\w+', s.lower()))
        intersect = len(resume_words & sw)
        union = len(resume_words | sw)
        score = intersect / union if union > 0 else 0
        scored.append((s, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]

def _parse_gliner_list(val):
    if not val:
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []

def normalize_job(r):
    n = {
        "title": r.get("title", ""),
        "company": r.get("postedCompany__name") or r.get("company") or "",
        "url": r.get("metadata__jobDetailsUrl") or r.get("url") or "",
        "min_exp": r.get("minimumYearsExperience") or r.get("min_exp"),
        "sal_min": r.get("salary__minimum") or r.get("sal_min"),
        "sal_max": r.get("salary__maximum") or r.get("sal_max"),
        "preview": r.get("preview") or "",
        "description": r.get("description") or r.get("combined_desc") or "",
        "embedding_text": r.get("embedding_text") or r.get("description") or r.get("combined_desc") or "",
        "skills": _parse_gliner_list(r.get("gliner_skills")) or (r.get("skills") or []),
        "tools": _parse_gliner_list(r.get("gliner_tools")) or (r.get("tools") or []),
        "profile_score": r.get("profile_score", 0),
        "resume_score": r.get("resume_score", 0),
        "occupation_score": r.get("occupation_score", 0),
        "uuid": r.get("uuid") or r.get("job_id"),
        "job_status": r.get("status__jobStatus") or "",
        "posted_date": r.get("metadata__newPostingDate") or r.get("metadata__originalPostingDate") or "",
        "expiry_date": r.get("metadata__expiryDate") or "",
    }
    raw_status = n["job_status"].lower()
    if raw_status in ("closed", "removed", "filled"):
        n["is_closed"] = True
    elif n["expiry_date"]:
        try:
            exp = date.fromisoformat(n["expiry_date"])
            n["is_closed"] = exp < date.today()
        except Exception:
            n["is_closed"] = False
    else:
        n["is_closed"] = False
    return n

def compute_job_matches(resume_weight):
    resume_text = st.session_state.profile["resume"].strip()
    occ_code = st.session_state.profile["preferred_code"]

    results = api_semantic_search(
        resume_text, "jobs", 100,
        occupation_code=occ_code,
        resume_weight=resume_weight,
    )

    enriched = []
    for rank, r in enumerate(results):
        n = normalize_job(r)
        snippets = make_snippets(resume_text, n["embedding_text"]) if rank < 10 else []
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

def compute_byo_match(description, resume_weight):
    resume_text = st.session_state.profile["resume"].strip()
    occ_code = st.session_state.profile["preferred_code"]

    profile_result = api_profile_match(
        resume_text, description,
        occupation_code=occ_code,
        resume_weight=resume_weight,
    )

    snippets = make_snippets(resume_text, description)

    try:
        occ_results = api_semantic_search(description.strip()[:1000], "occupations", 5)
        similar = []
        for r in occ_results:
            similar.append({"code": int(r["code"]), "title": r["title"], "score": r["profile_score"]})
    except Exception:
        similar = []

    return {
        "description": description.strip(),
        "score": profile_result.get("profile_score", 0),
        "resume_score": profile_result.get("resume_score", 0),
        "occ_score": profile_result.get("occupation_score", 0),
        "snippets": snippets,
        "similar_occ": similar,
    }

# ── Branding & Navigation ─────────────────────────────
MODES = [(0, "Exact Match", 1.00), (25, "Career Fit", 0.90), (50, "Balanced", 0.80), (75, "Career Pivot", 0.60), (100, "Career Transition", 0.40)]

def _get_mode(career_direction):
    return min(MODES, key=lambda m: abs(m[0] - career_direction))

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
    row = df[df["occupation_code"] == st.session_state.selected_code]
    if not row.empty:
        show_occupation_page(row.iloc[0])
else:

    if st.session_state.nav_tab != st.session_state.prev_tab:
        if st.session_state.nav_tab == "My Profile":
            st.session_state.career_dir = st.session_state.profile["career_direction"]
            st.session_state.max_exp_years = st.session_state.profile["max_exp"]
        st.session_state.prev_tab = st.session_state.nav_tab

    if st.session_state.nav_tab == "Home":
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
                label = _get_mode(st.session_state.profile["career_direction"])[1]
                st.markdown("**Career Direction**")
                st.info(label)
            if st.button("Continue Finding Jobs", type="primary", width="stretch"):
                st.session_state.nav_target = "Find Jobs"
                st.rerun()


    elif st.session_state.nav_tab == "My Profile":
        st.subheader("Resume")
        if st.session_state.profile["resume"] and not st.session_state.resume_input:
            st.session_state.resume_input = st.session_state.profile["resume"]
        st.text_area(
            "Paste your resume, job description, or career summary",
            key="resume_input",
            placeholder="e.g. Experienced data scientist with 5 years in machine learning, Python, and statistical modelling...",
            height=200,
        )
        if st.button("Save Resume", type="primary"):
            st.session_state.profile["resume"] = st.session_state.resume_input
            st.session_state.jobs_dirty = True
            if st.session_state.profile["resume"].strip():
                with st.spinner("Matching your resume to occupations..."):
                    api_results = api_semantic_search(
                        st.session_state.profile["resume"].strip(),
                        "occupations", 5,
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
                        st.session_state.jobs_dirty = True

        if st.session_state.profile["resume"].strip():
            st.caption(f"Resume: {len(st.session_state.profile['resume'].split())} words ✓ Uploaded")
            st.divider()

            st.subheader("Top Occupation Matches")
            if st.session_state.match_results:
                for row_i in range(0, len(st.session_state.match_results), 3):
                    cols = st.columns(3)
                    for ci, res in enumerate(st.session_state.match_results[row_i:row_i + 3]):
                        with cols[ci]:
                            with st.container(border=True):
                                st.markdown(f"##### {res['title']}")
                                st.markdown(f"Code: {res['code']}  \nMatch: {res['score']:.0%}")
                                sel = st.session_state.profile["preferred_code"] == res['code']
                                if st.button("Select", key=f"sel_{res['code']}", width="stretch", disabled=sel,
                                             type="primary" if sel else "secondary"):
                                    st.session_state.profile["preferred_code"] = res['code']
                                    st.session_state.profile["preferred_title"] = res['title']
                                    st.session_state.jobs_dirty = True
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
                    if st.button("Clear target occupation"):
                        st.session_state.profile["preferred_code"] = None
                        st.session_state.profile["preferred_title"] = None
                        st.session_state.jobs_dirty = True
                        st.rerun()
                else:
                    st.warning("Select a target occupation from your matches above.")
            with col_career:
                st.subheader("Career Direction")
                st.caption("Slide towards **Career Transition** to prioritise your target occupation over your current experience when finding matching jobs.")
                help_lines = ["Controls how much weight to put on your current experience vs your target occupation."]
                for _, lbl, rw in MODES:
                    help_lines.append(f"  - {lbl}: {rw*100:.0f}% resume / {(1-rw)*100:.0f}% occupation")
                lbl_left, slider, lbl_right = st.columns([1, 4, 1])
                with lbl_left:
                    st.markdown("<div style='text-align:center;padding-top:4px;font-size:0.85rem;color:#666'>Exact Match</div>", unsafe_allow_html=True)
                with slider:
                    st.select_slider(
                        "Career Direction",
                        options=[m[0] for m in MODES],
                        format_func=lambda x: _get_mode(x)[1],
                        key="career_dir",
                        label_visibility="collapsed",
                        help="\n".join(help_lines),
                    )
                with lbl_right:
                    st.markdown("<div style='text-align:center;padding-top:4px;font-size:0.85rem;color:#666'>Career Transition</div>", unsafe_allow_html=True)
                if st.session_state.profile["career_direction"] != st.session_state.career_dir:
                    st.session_state.jobs_dirty = True
                st.session_state.profile["career_direction"] = st.session_state.career_dir

        st.divider()
        st.subheader("Preferences")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.slider("Max Experience (years)", 0, 20, key="max_exp_years")
            if st.session_state.profile["max_exp"] != st.session_state.max_exp_years:
                st.session_state.jobs_dirty = True
            st.session_state.profile["max_exp"] = st.session_state.max_exp_years
        with col_s2:
            st.markdown("**Salary Range (SGD/month)**")
            sal_min = st.number_input("Min Salary", min_value=0, max_value=100000, value=st.session_state.profile["sal_min"], step=500, key="sal_min_val")
            sal_max = st.number_input("Max Salary", min_value=0, max_value=100000, value=st.session_state.profile["sal_max"], step=500, key="sal_max_val")
            if st.session_state.profile["sal_min"] != sal_min or st.session_state.profile["sal_max"] != sal_max:
                st.session_state.jobs_dirty = True
            st.session_state.profile["sal_min"] = sal_min
            st.session_state.profile["sal_max"] = sal_max
        job_status = st.selectbox("Job Status", ["Open", "Closed", "All"], key="job_status_sel")
        if st.session_state.profile["job_status_filter"] != job_status:
            st.session_state.jobs_dirty = True
        st.session_state.profile["job_status_filter"] = job_status

    elif st.session_state.nav_tab == "Explore Occupations":
        pmet_filter = st.segmented_control(
            "Filter by major group", ["All", "PMET", "Non-PMET", "1 - Managers", "2 - Professionals", "3 - Associate Professionals"],
            key="pmet_filter", default="All",
        )
        filter_map = {
            "All": lambda x: True,
            "PMET": lambda x: x in (1, 2, 3),
            "Non-PMET": lambda x: x not in (1, 2, 3),
            "1 - Managers": lambda x: x == 1,
            "2 - Professionals": lambda x: x == 2,
            "3 - Associate Professionals": lambda x: x == 3,
        }
        st.subheader("Search Occupations")
        st.markdown("Search by code or title.")
        map_merged = umap_df.merge(
            occ_stats_df[["occ_code", "job_post_count"]], left_on="code", right_on="occ_code", how="left"
        )
        map_merged["group"] = map_merged["major_code"].map(MAJOR_LABELS)
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
        st.divider()

        st.subheader("Occupation Map")
        st.markdown("Click any point to view its profile.")
        map_filtered = umap_df[umap_df["major_code"].apply(filter_map.get(pmet_filter, lambda x: True))]
        map_merged = map_filtered.merge(
            occ_stats_df[["occ_code", "job_post_count"]], left_on="code", right_on="occ_code", how="left"
        )
        map_merged["group"] = map_merged["major_code"].map(MAJOR_LABELS)
        fig = px.scatter(
            map_merged, x="x", y="y",
            color="group",
            color_discrete_map={MAJOR_LABELS[k]: v for k, v in MAJOR_COLORS.items()},
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

        st.divider()
        st.subheader("Job Posts by Occupation")
        st.markdown("Each job post is matched to the SSOC occupation it is most similar to.")
        occ_chart = occ_stats_df.merge(emb_meta[["code", "title", "major_code"]], left_on="occ_code", right_on="code", how="left")
        occ_chart = occ_chart.sort_values("job_post_count", ascending=False)

        mask = occ_chart["major_code"].apply(filter_map.get(pmet_filter, lambda x: True))
        filtered = occ_chart[mask]

        show_n = st.selectbox("Show top", [20, 50, 100, 200, len(filtered)], index=1, key="eo_topn")
        top_chart = filtered.head(show_n)
        fig2 = px.bar(
            top_chart, x="job_post_count", y="title", orientation="h",
            labels={"job_post_count": "Job Posts", "title": "Occupation"},
            height=max(400, show_n * 20),
            color="job_post_count", color_continuous_scale="Blues",
        )
        fig2.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig2, width="stretch")
        display_cols = ["occ_code", "title", "job_post_count", "avg_sal_min", "avg_sal_max", "median_sal", "mean_exp", "median_exp"]
        display = filtered[display_cols].rename(columns={
            "occ_code": "Code", "title": "Occupation", "job_post_count": "Job Posts",
            "avg_sal_min": "Avg Sal Min", "avg_sal_max": "Avg Sal Max", "median_sal": "Median Sal",
            "mean_exp": "Avg Exp (yr)", "median_exp": "Median Exp (yr)",
        })
        st.dataframe(display, width="stretch", hide_index=True,
                     column_config={
                         "Avg Sal Min": st.column_config.NumberColumn(format="$%.0f"),
                         "Avg Sal Max": st.column_config.NumberColumn(format="$%.0f"),
                         "Median Sal": st.column_config.NumberColumn(format="$%.0f"),
                         "Avg Exp (yr)": st.column_config.NumberColumn(format="%.1f"),
                         "Median Exp (yr)": st.column_config.NumberColumn(format="%.1f"),
                         "Job Posts": st.column_config.NumberColumn(format="%d"),
                     })

        st.divider()
        st.subheader(f"Top 50 Skills — {pmet_filter}")
        gtop = global_top_skills(50, major_filter=pmet_filter)
        gtop["rank"] = range(1, len(gtop) + 1)
        st.dataframe(
            gtop[["rank", "main_term", "total_count"]],
            column_config={
                "rank": "Rank",
                "main_term": "Skill",
                "total_count": st.column_config.NumberColumn("Job Posts", format="%d"),
            },
            hide_index=True, width="stretch",
        )

    elif st.session_state.nav_tab == "Find Jobs":
        st.markdown("Recommendations based on your profile settings under the **My Profile** tab.")
        resume_weight = _get_mode(st.session_state.profile["career_direction"])[2]

        ready = bool(st.session_state.profile["resume"].strip() and st.session_state.profile["preferred_code"])

        if ready and st.session_state.get("jobs_dirty", False):
            st.session_state.jobs_dirty = False
            with st.spinner("Computing job matches..."):
                st.session_state.job_match_results = compute_job_matches(resume_weight)
            st.session_state.job_page = 1

        if st.session_state.get("job_match_results"):
            all_results = st.session_state.job_match_results
            job_status_filter = st.session_state.profile.get("job_status_filter", "Open")
            max_exp_filter = st.session_state.profile["max_exp"]
            sal_min_filter = st.session_state.profile["sal_min"]
            sal_max_filter = st.session_state.profile["sal_max"]
            results = [
                r for r in all_results
                if (r["min_exp"] is None or r["min_exp"] <= max_exp_filter)
                and (r["sal_min"] is None or r["sal_max"] is None or
                     (r["sal_min"] <= sal_max_filter and r["sal_max"] >= sal_min_filter))
            ]
            if job_status_filter == "Open":
                results = [r for r in results if not r.get("is_closed", False)]
            elif job_status_filter == "Closed":
                results = [r for r in results if r.get("is_closed", False)]
            total_all = len(all_results)
            st.subheader(f"Top {len(results)} Matching Jobs" + (f" (filtered from {total_all})" if len(results) < total_all else ""))

            label = next(m[1] for m in MODES if m[0] == st.session_state.profile["career_direction"])
            st.info(f"**Overall Match** computed with **{label}** weighting ({resume_weight*100:.0f}% resume / {(1-resume_weight)*100:.0f}% occupation)")

            per_page = 10
            total_pages = max(1, (len(results) + per_page - 1) // per_page)
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

            from concurrent.futures import ThreadPoolExecutor, as_completed
            lazy_resume_text = st.session_state.profile["resume"].strip()
            page_results = list(results[start_idx:end_idx])
            missing = [(i, r) for i, r in enumerate(page_results) if not r.get("snippets") and r.get("combined_desc")]
            if missing:
                with st.spinner("Computing evidence..."):
                    def compute_snippets(i, r):
                        try:
                            return i, make_snippets(lazy_resume_text, r["combined_desc"])
                        except Exception:
                            return i, []
                    with ThreadPoolExecutor(max_workers=5) as ex:
                        futs = {ex.submit(compute_snippets, i, r): i for i, r in missing}
                        for f in as_completed(futs):
                            i, snippets = f.result()
                            page_results[i]["snippets"] = snippets
            for idx, res in enumerate(page_results):
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
                                    resume = st.session_state.profile["resume"].strip()[:2000]
                                    desc = res.get("combined_desc", "")[:2000]
                                    prompt = f"In a few sentences, explain why this job is a good match. State which skills align and note any obvious gaps.\n\nResume:\n{resume}\n\nJob Description:\n{desc}"
                                    explanation, used_fallback = explain_match(prompt)
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

    elif st.session_state.nav_tab == "Bring Your Own Job":
        st.markdown("""
**Bring Your Own Job** lets you evaluate job descriptions from any source — LinkedIn, company career pages, job boards — against your profile.

Paste a job description below and we'll score it based on your **resume**, **target occupation**, and **career direction** set in **My Profile**. Jobs are added to a ranked list so you can compare them at a glance.
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
            resume_weight = _get_mode(st.session_state.profile["career_direction"])[2]
            label = next(m[1] for m in MODES if m[0] == st.session_state.profile["career_direction"])

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
                    with st.container(border=True):
                        col_title, col_btns = st.columns([4, 1])
                        with col_title:
                            st.markdown(f"**#{rank}** &nbsp; Match: **{job['score']:.1%}**")
                            st.markdown(f"Resume: {job['resume_score']:.1%} &nbsp; Occupation: {job['occ_score']:.1%} &nbsp; Weighting: {label} ({resume_weight*100:.0f}% / {(1-resume_weight)*100:.0f}%)")
                            preview = _strip_html(job["description"][:200] + ("..." if len(job["description"]) > 200 else ""))
                            st.caption(preview)
                        with col_btns:
                            state_key = f"byo_explain_{id(job)}"
                            if st.button("✨ Explain", key=state_key + "_btn", use_container_width=True):
                                with st.spinner("Analysing..."):
                                    resume = st.session_state.profile["resume"].strip()[:2000]
                                    desc = job["description"][:2000]
                                    prompt = f"In a few sentences, explain why this job is a good match. State which skills align and note any obvious gaps.\n\nResume:\n{resume}\n\nJob Description:\n{desc}"
                                    explanation, used_fallback = explain_match(prompt)
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


