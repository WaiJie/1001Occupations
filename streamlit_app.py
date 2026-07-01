import os, warnings
os.environ["HF_HUB_DISABLE_SYMLINK_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SENTENCE_TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*use_return_dict.*")

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
</style>
""", unsafe_allow_html=True)

DATA_PATH = "data/ssoc2024_flat_wide.xlsx"
COORDS_PATH = "data/ssoc_umap_coords.csv"
EMB_PATH = "data/ssoc_embeddings.npy"
META_PATH = "data/ssoc_metadata.csv"
JOB_EMB_PATH1 = "data/job_embeddings_part1.npy"
JOB_EMB_PATH2 = "data/job_embeddings_part2.npy"
JOB_META_PATH1 = "data/job_metadata_part1.csv"
JOB_META_PATH2 = "data/job_metadata_part2.csv"
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
def load_embeddings():
    emb = np.load(EMB_PATH)
    meta = pd.read_csv(META_PATH)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / norms
    return emb_norm, meta

@st.cache_data
def load_job_embeddings():
    emb = np.concatenate([np.load(JOB_EMB_PATH1), np.load(JOB_EMB_PATH2)])
    meta = pd.concat([pd.read_csv(JOB_META_PATH1), pd.read_csv(JOB_META_PATH2)], ignore_index=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / norms
    return emb_norm, meta

df = load_occupations()
umap_df = load_coords()
emb_norm, emb_meta = load_embeddings()
job_emb_norm, job_meta = load_job_embeddings()
code_to_idx = {row["code"]: i for i, row in emb_meta.iterrows()}

@st.cache_data
def load_occupation_skills_tools():
    return pd.read_csv(OCC_SKILLS_TOOLS_PATH)

@st.cache_data
def load_occupation_stats():
    return pd.read_csv("data/occupation_stats.csv")

occ_skills_tools_df = load_occupation_skills_tools()
occ_stats_df = load_occupation_stats()

@st.cache_resource
def get_sim_matrix():
    return emb_norm @ emb_norm.T

sim_matrix = get_sim_matrix()

@st.cache_resource
def load_model():
    print("[load_model] Starting...", flush=True)
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        print(f"[load_model] torch available: {torch.cuda.is_available()}", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[load_model] Device: {device}", flush=True)
        print("[load_model] Loading SentenceTransformer...", flush=True)
        model = SentenceTransformer(
            "jinaai/jina-embeddings-v5-text-nano",
            trust_remote_code=True,
            revision="refs/pr/11",
            device=device,
            model_kwargs={"torch_dtype": torch.bfloat16, "default_task": "text-matching"},
        )
        print("[load_model] Done", flush=True)
        return model
    except Exception as e:
        import traceback
        print(f"[load_model ERROR]: {e}", flush=True)
        traceback.print_exc()
        raise

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
        "career_direction": 33,
        "max_exp": 20,
    }
if "career_dir" not in st.session_state:
    st.session_state.career_dir = st.session_state.profile["career_direction"]
if "max_exp_years" not in st.session_state:
    st.session_state.max_exp_years = st.session_state.profile["max_exp"]
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

def top_n_similar(code, n=5):
    idx = code_to_idx.get(code)
    if idx is None:
        return []
    vec = emb_norm[idx]
    sims = emb_norm @ vec
    top = np.argsort(sims)[-(n + 1):][::-1]
    results = []
    for i in top:
        if i == idx:
            continue
        results.append((emb_meta.iloc[i]["code"], emb_meta.iloc[i]["title"], sims[i]))
        if len(results) == n:
            break
    return results

def show_similarity_cards(code):
    st.subheader("Top 5 Related Occupations")
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
    st.subheader("Semantic Neighbourhood")
    st.markdown("Visualises functional similarity between occupations based on their tasks and responsibilities. Nearby occupations often perform similar work, while connections may span across the map when occupations share similar functions despite belonging to different industries.")
    idx = code_to_idx[code]
    sims = sim_matrix[idx]
    sorted_indices = np.argsort(sims)[::-1][1:]
    sorted_sims = sims[sorted_indices]
    gap_threshold = 0.03
    max_candidates = 40
    cut = max_candidates
    for i in range(min(max_candidates, len(sorted_sims) - 1)):
        if sorted_sims[i] - sorted_sims[i + 1] > gap_threshold:
            cut = i + 1
            break
    n_neighbours = max(5, cut)
    neighbour_indices = sorted_indices[:n_neighbours]
    neighbour_codes = set(emb_meta.iloc[neighbour_indices]["code"].values)
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
    event = st.plotly_chart(fig, key=f"neighbourhood_{code}", width="stretch", on_select="rerun")
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
    titles = [proper_case(row[f"{l}_group_title"]) for l in ["major", "sub_major", "minor", "unit"]]
    titles.append(proper_case(row["occupation_title"]))
    st.markdown(f"SSOC: {' → '.join(titles)}")
    occ_codes, _, _ = compute_job_occupation_matches()
    job_post_count = int(np.sum(occ_codes == code))

    col_title, col_count = st.columns([3, 1])
    with col_title:
        st.markdown(f"## {code} — {title}")
    with col_count:
        st.metric("Job Posts", f"{job_post_count:,}")
    st.markdown("Explore the responsibilities, skills, career context and related occupations for this role.")

    occ_stat = occ_stats_df[occ_stats_df["occ_code"] == code]
    if not occ_stat.empty:
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
    st.subheader("Overview")
    st.write(row["detailed_definitions"])
    st.divider()
    st.subheader("What They Do")
    tasks = parse_csv_list(row["tasks"])
    if tasks:
        for t in tasks:
            st.markdown(f"- {t}")
    else:
        st.info("No tasks listed.")
    st.divider()
    occ_rows = occ_skills_tools_df[occ_skills_tools_df["occ_code"] == code].sort_values("rank")
    if not occ_rows.empty:
        st.subheader("Skills & Tools in Demand (from job posts)")
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
    occ_codes_all, _, best_sim_all = compute_job_occupation_matches()
    job_mask = occ_codes_all == code
    if job_mask.sum() > 0:
        job_indices = np.where(job_mask)[0]
        top5_idx = job_indices[np.argsort(best_sim_all[job_indices])[::-1][:5]]
        st.subheader("Top Representative Job Posts")
        st.markdown("Highest-similarity job postings matched to this occupation.")
        for rank, ji in enumerate(top5_idx, 1):
            jr = job_meta.iloc[ji]
            sim = best_sim_all[ji]
            preview = jr["preview"] if pd.notna(jr["preview"]) else ""
            company = jr["company"] if pd.notna(jr["company"]) else ""
            url = jr["url"] if pd.notna(jr["url"]) else ""
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
        st.subheader("Notes")
        st.write(notes)
        st.divider()
    st.subheader("Common Job Titles")
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
def compute_job_matches(resume_weight):
    import re
    model = load_model()

    resume_vec = model.encode([f"Tasks: {st.session_state.profile['resume'].strip()}"], convert_to_numpy=True, task="text-matching")
    resume_vec_norm = resume_vec / np.linalg.norm(resume_vec)

    occ_idx = code_to_idx.get(st.session_state.profile["preferred_code"])
    occ_vec = emb_norm[occ_idx]

    combined = resume_weight * resume_vec_norm[0] + (1 - resume_weight) * occ_vec
    combined_norm = combined / np.linalg.norm(combined)

    sims = job_emb_norm @ combined_norm
    top = np.argsort(sims)[-100:][::-1]

    all_sentences = []
    job_sent_ranges = []
    for idx in top:
        desc = job_meta.iloc[idx]["combined_desc"]
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", desc) if len(s.strip()) > 20]
        if not sents:
            sents = [desc[:200]]
        start = len(all_sentences)
        all_sentences.extend(sents)
        job_sent_ranges.append((start, len(all_sentences)))

    sent_embs = model.encode(all_sentences, convert_to_numpy=True, task="text-matching")
    resume_emb_norm = resume_vec_norm[0] / np.linalg.norm(resume_vec_norm[0])
    sent_sims = sent_embs @ resume_emb_norm

    job_vecs = job_emb_norm[top]
    resume_sims_all = job_vecs @ resume_vec_norm[0]
    occ_sims_all = job_vecs @ occ_vec

    results = []
    for rank, (idx, (s_start, s_end)) in enumerate(zip(top, job_sent_ranges)):
        job_sent_scores = sent_sims[s_start:s_end]
        top_k = min(3, len(job_sent_scores))
        top_sent_idx = np.argsort(job_sent_scores)[-top_k:][::-1]
        snippets = [(all_sentences[s_start + si], float(job_sent_scores[si])) for si in top_sent_idx]
        row = job_meta.iloc[idx]
        skills_list, tools_list = [], []
        try:
            skills_list = json.loads(row["skills"]) if isinstance(row["skills"], str) else row["skills"]
        except Exception:
            pass
        try:
            tools_list = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
        except Exception:
            pass
        results.append({
            "rank": rank + 1,
            "title": row["title"],
            "company": row["company"] if pd.notna(row["company"]) else "",
            "url": row["url"] if pd.notna(row["url"]) else "",
            "min_exp": int(row["min_exp"]) if pd.notna(row["min_exp"]) else None,
            "sal_min": int(row["sal_min"]) if pd.notna(row["sal_min"]) else None,
            "sal_max": int(row["sal_max"]) if pd.notna(row["sal_max"]) else None,
            "preview": row["preview"],
            "skills": skills_list,
            "tools": tools_list,
            "score": float(sims[idx]),
            "resume_score": float(resume_sims_all[rank]),
            "occ_score": float(occ_sims_all[rank]),
            "snippets": snippets,
        })

    return results

def compute_byo_match(description, resume_weight):
    import re
    model = load_model()

    job_vec = model.encode([f"Tasks: {description.strip()}"], convert_to_numpy=True, task="text-matching")
    job_vec_norm = job_vec / np.linalg.norm(job_vec)

    resume_vec = model.encode([f"Tasks: {st.session_state.profile['resume'].strip()}"], convert_to_numpy=True, task="text-matching")
    resume_vec_norm = resume_vec / np.linalg.norm(resume_vec)

    occ_idx = code_to_idx.get(st.session_state.profile["preferred_code"])
    occ_vec = emb_norm[occ_idx]

    combined = resume_weight * resume_vec_norm[0] + (1 - resume_weight) * occ_vec
    combined_norm = combined / np.linalg.norm(combined)

    overall_score = float(job_vec_norm[0] @ combined_norm)
    resume_score = float(job_vec_norm[0] @ resume_vec_norm[0])
    occ_score = float(job_vec_norm[0] @ occ_vec)

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", description) if len(s.strip()) > 20]
    if not sents:
        sents = [description[:200]]

    sent_embs = model.encode(sents, convert_to_numpy=True, task="text-matching")
    sent_sims = sent_embs @ resume_vec_norm[0]
    top_k = min(3, len(sents))
    top_sent_idx = np.argsort(sent_sims)[-top_k:][::-1]
    snippets = [(sents[si], float(sent_sims[si])) for si in top_sent_idx]

    return {
        "description": description.strip(),
        "score": overall_score,
        "resume_score": resume_score,
        "occ_score": occ_score,
        "snippets": snippets,
    }

# ── Job analytics ────────────────────────────────────────
@st.cache_data
def compute_job_occupation_matches():
    n_jobs = len(job_emb_norm)
    n_occ = len(emb_norm)
    best_idx = np.empty(n_jobs, dtype=int)
    best_sim = np.empty(n_jobs, dtype=float)
    batch_size = 1000
    for i in range(0, n_jobs, batch_size):
        batch = job_emb_norm[i:i + batch_size]
        sims = batch @ emb_norm.T
        best_idx[i:i + batch_size] = np.argmax(sims, axis=1)
        best_sim[i:i + batch_size] = np.max(sims, axis=1)
    occ_codes = emb_meta.iloc[best_idx]["code"].values
    occ_titles = emb_meta.iloc[best_idx]["title"].values
    return occ_codes, occ_titles, best_sim

# ── Branding ─────────────────────────────────────────────
st.markdown("<div style='display:flex; align-items:baseline; gap:12px; margin:0 0 0 0; padding:0;'><h1 style='font-size:2rem; font-weight:900; margin:0; padding:0; line-height:1;'>1001 Occupations</h1><span style='color:#888; font-size:0.85rem;'>Explore Singapore's SSOC 2024 framework</span></div>", unsafe_allow_html=True)
st.divider()

# ── Navigation & Tab Router ──────────────────────────────
MODES = [(0, "Career Fit", 0.90), (33, "Balanced", 0.80), (67, "Career Pivot", 0.60), (100, "Career Transition", 0.40)]

if st.session_state.page == "occupation":
    row = df[df["occupation_code"] == st.session_state.selected_code]
    if not row.empty:
        show_occupation_page(row.iloc[0])
else:
    if st.session_state.nav_target:
        st.session_state.nav_tab = st.session_state.nav_target
        st.session_state.nav_target = None
    st.segmented_control("Navigation", ["Home", "My Profile", "Explore Occupations", "Find Jobs", "Bring Your Own Job"], key="nav_tab")

    st.markdown("---")

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
                label = next(m[1] for m in MODES if m[0] == st.session_state.profile["career_direction"])
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
                    model = load_model()
                    vec = model.encode([f"Tasks: {st.session_state.profile['resume'].strip()}"], convert_to_numpy=True, task="text-matching")
                    vec_norm = vec / np.linalg.norm(vec)
                    sims = (emb_norm @ vec_norm[0]).flatten()
                    top = np.argsort(sims)[-5:][::-1]
                    st.session_state.match_results = []
                    for idx in top:
                        code = int(emb_meta.iloc[idx]["code"])
                        title = emb_meta.iloc[idx]["title"]
                        st.session_state.match_results.append({
                            "rank": len(st.session_state.match_results) + 1,
                            "code": code, "title": title, "score": float(sims[idx]),
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
        else:
            st.info("Save your resume above to get started.")

        st.divider()
        st.subheader("Career Direction")
        help_lines = ["Controls how much weight to put on your current experience vs your target occupation."]
        for _, label, rw in MODES:
            help_lines.append(f"  - {label}: {rw*100:.0f}% resume / {(1-rw)*100:.0f}% occupation")
        st.select_slider(
            "Career Direction",
            options=[m[0] for m in MODES],
            format_func=lambda x: next(m[1] for m in MODES if m[0] == x),
            key="career_dir",
            help="\n".join(help_lines),
        )
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
            st.markdown("**Salary Range** — *Coming soon*")

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
        st.markdown("Search by title — results appear below.")
        search_q = st.text_input("Search occupation title", placeholder="e.g. data scientist", label_visibility="collapsed")
        if search_q:
            matches = df[df["occupation_title"].str.contains(search_q, case=False, na=False)]
            for _, row in matches.head(20).iterrows():
                code = int(row["occupation_code"])
                with st.container(border=True):
                    st.markdown(f"**{code}** — {row['occupation_title']}")
                    if st.button("View", key=f"eo_s_{code}", width="stretch"):
                        st.query_params["occupation"] = str(code)
                        st.session_state.selected_code = code
                        st.session_state.page = "occupation"
                        st.rerun()
                if len(matches) > 20:
                    st.caption(f"... and {len(matches) - 20} more")
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
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(font=dict(size=10)))
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
        fig = px.bar(
            top_chart, x="job_post_count", y="title", orientation="h",
            labels={"job_post_count": "Job Posts", "title": "Occupation"},
            height=max(400, show_n * 20),
            color="job_post_count", color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch")

        st.subheader(f"{pmet_filter}: {len(filtered)} Occupations")
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

    elif st.session_state.nav_tab == "Find Jobs":
        st.markdown("Recommendations based on your profile settings under the **My Profile** tab.")
        resume_weight = next(m[2] for m in MODES if m[0] == st.session_state.profile["career_direction"])

        ready = bool(st.session_state.profile["resume"].strip() and st.session_state.profile["preferred_code"])

        if ready and st.session_state.get("jobs_dirty", False):
            st.session_state.jobs_dirty = False
            with st.spinner("Computing job matches..."):
                st.session_state.job_match_results = compute_job_matches(resume_weight)
            st.session_state.job_page = 1

        if st.session_state.get("job_match_results"):
            all_results = st.session_state.job_match_results
            max_exp_filter = st.session_state.profile["max_exp"]
            results = [r for r in all_results if r["min_exp"] is None or r["min_exp"] <= max_exp_filter]
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

            for res in results[start_idx:end_idx]:
                with st.container(border=True):
                    st.markdown(f"**#{res['rank']}** &nbsp; {res['title']} &nbsp;—&nbsp; Match: **{res['score']:.1%}**")
                    st.markdown(f"Resume: {res['resume_score']:.1%} &nbsp; Occupation: {res['occ_score']:.1%}")
                    meta_parts = []
                    if res["company"]:
                        meta_parts.append(f"<span style='color:#1b2a4a;font-weight:700'>{res['company']}</span>")
                    if res["min_exp"] is not None:
                        meta_parts.append(f"Exp: {int(res['min_exp'])} yr{'s' if res['min_exp'] != 1 else ''}")
                    if res["sal_min"] is not None and res["sal_max"] is not None:
                        meta_parts.append(f"SGD {int(res['sal_min']):,} to SGD {int(res['sal_max']):,}")
                    if res["url"]:
                        meta_parts.append(f"[Apply]({res['url']})")
                    if meta_parts:
                        st.markdown(" | ".join(meta_parts), unsafe_allow_html=True)
                    if res.get("skills") or res.get("tools"):
                        seen = set()
                        combined = []
                        for item in (res.get("skills") or [])[:5] + (res.get("tools") or [])[:5]:
                            if item not in seen:
                                seen.add(item)
                                combined.append(item)
                        st.markdown(f"<span style='color:#e67e22'>{' · '.join(combined)}</span>", unsafe_allow_html=True)
                    if res.get("snippets"):
                        with st.expander("Evidence from job description", expanded=False):
                            for sent, sc in res["snippets"]:
                                st.markdown(f"- {sent} ({sc:.0%})")

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
            resume_weight = next(m[2] for m in MODES if m[0] == st.session_state.profile["career_direction"])
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
                        col_main, col_remove = st.columns([10, 1])
                        with col_main:
                            st.markdown(f"**#{rank}** &nbsp; Match: **{job['score']:.1%}**")
                            st.markdown(f"Resume: {job['resume_score']:.1%} &nbsp; Occupation: {job['occ_score']:.1%} &nbsp; Weighting: {label} ({resume_weight*100:.0f}% / {(1-resume_weight)*100:.0f}%)")
                            preview = job["description"][:200] + ("..." if len(job["description"]) > 200 else "")
                            st.caption(preview)
                        with col_remove:
                            if st.button("✕", key=f"byo_del_{id(job)}", help="Remove"):
                                st.session_state.byo_jobs = [j for j in st.session_state.byo_jobs if id(j) != id(job)]
                                st.rerun()

                        with st.expander("Evidence from job description", expanded=False):
                            for sent, sc in job["snippets"]:
                                st.markdown(f"- {sent} ({sc:.0%})")

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


