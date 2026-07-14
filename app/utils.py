import ast
import csv
import io
import re
from datetime import date

import pandas as pd
import streamlit as st


def proper_case(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    return re.sub(
        r'\([^)]*\)|[^\s(]+',
        lambda m: m.group(0) if m.group(0).startswith("(") else m.group(0).title(),
        s,
    )


def parse_csv_list(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    reader = csv.reader(io.StringIO(str(val)), skipinitialspace=True)
    return [s.strip('" ') for s in next(reader, []) if s.strip()]


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


def _as_list(val):
    """Coerce a skills/tools value into a list (handles JSON strings)."""
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.strip():
        parsed = _parse_gliner_list(val)
        if parsed:
            return parsed
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
        "skills": _as_list(_parse_gliner_list(r.get("gliner_skills")) or r.get("skills")),
        "tools": _as_list(_parse_gliner_list(r.get("gliner_tools")) or r.get("tools")),
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
