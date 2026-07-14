import streamlit as st
from gradio_client import Client

from app import config


@st.cache_resource
def get_api_client():
    token = st.secrets["HF_TOKEN"]
    return Client(config.API_SPACE, token=token)
