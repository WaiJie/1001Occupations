from app.api.client import get_api_client


def _code(occupation_code):
    """Normalise occupation_code to the int the API dropdown expects (or None).

    The space builds its occupation dropdown from (label, int(code)) tuples, so
    the accepted choice values are ints, not strings.
    """
    if occupation_code is None:
        return None
    return int(occupation_code)


# ── Occupation search (new dedicated endpoint) ──────────
def search_occupations(text, top_k=5, occupation_code=None, resume_weight=1.0):
    client = get_api_client()
    return client.predict(
        text,
        float(top_k),
        _code(occupation_code),
        float(resume_weight),
        api_name="/_occupation_search",
    )


# ── Job search (new dedicated endpoint, with filters) ──
def search_jobs(text, top_k=5, occupation_code=None, resume_weight=1.0, filters=None):
    client = get_api_client()
    f = filters or {}
    return client.predict(
        text,
        float(top_k),
        _code(occupation_code),
        float(resume_weight),
        float(f.get("sal_min", 0)),
        float(f.get("sal_max", 1000000)),
        float(f.get("max_exp", 20)),
        f.get("job_status", "All"),
        f.get("source", ""),
        f.get("posted_within", None),
        f.get("work_arrangement", ""),
        api_name="/_job_search",
    )


# ── Single profile match (BYO "Add to List") ────────────
def match_profile(resume_text, job_description, occupation_code=None, resume_weight=1.0):
    client = get_api_client()
    return client.predict(
        resume_text,
        job_description,
        _code(occupation_code),
        float(resume_weight),
        api_name="/profile_match",
    )


# ── Batch profile match (BYO "Re-analyse All" / 2 resumes) ──
def match_profiles_batch(resume_texts, job_description, occupation_code=None, resume_weight=1.0):
    client = get_api_client()
    return client.predict(
        resume_texts,
        job_description,
        _code(occupation_code),
        float(resume_weight),
        api_name="/multi_profile_match",
    )


# ── LLM explain (unchanged) ─────────────────────────────
def call_external_llm(message, system_prompt="You are a helpful career assistant.", temperature=0.2, max_new_tokens=512):
    client = get_api_client()
    return client.predict(
        message, system_prompt, float(temperature), float(max_new_tokens),
        api_name="/call_external_llm",
    )


def call_llm(message, system_prompt="You are a helpful career assistant.", temperature=0.2, max_new_tokens=512):
    client = get_api_client()
    return client.predict(
        message, system_prompt, float(temperature), float(max_new_tokens),
        api_name="/call_llm",
    )


def explain_match(prompt):
    try:
        return call_external_llm(prompt), False
    except Exception:
        try:
            return call_llm(prompt), True
        except Exception:
            return None, False
