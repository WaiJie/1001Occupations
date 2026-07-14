import pandas as pd
import streamlit as st

from app import config


@st.cache_data
def load_occupations():
    df = pd.read_excel(config.DATA_PATH)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    return df


@st.cache_data
def load_coords():
    return pd.read_csv(config.COORDS_PATH)


@st.cache_data
def load_occupation_metadata():
    return pd.read_csv(config.META_PATH)


@st.cache_data
def load_occupation_skills_tools():
    return pd.read_csv(config.OCC_SKILLS_TOOLS_PATH)


@st.cache_data
def load_occupation_stats():
    return pd.read_csv("data/occupation_stats.csv")


df = load_occupations()
umap_df = load_coords()
emb_meta = load_occupation_metadata()
occ_skills_tools_df = load_occupation_skills_tools()
occ_stats_df = load_occupation_stats()
