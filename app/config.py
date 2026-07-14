import os

# ── Data paths ──────────────────────────────────────────
DATA_PATH = "data/ssoc2024_flat_wide.xlsx"
COORDS_PATH = "data/ssoc_umap_coords.csv"
META_PATH = "data/ssoc_metadata.csv"
OCC_SKILLS_TOOLS_PATH = "data/occupation_skills_tools.csv"

# ── API ─────────────────────────────────────────────────
API_SPACE = "Wjchua/1001OccupationsAPI"

# ── Visual constants ────────────────────────────────────
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

# Career direction presets: (slider value, label, resume_weight)
MODES = [
    (0, "Exact Match", 1.00),
    (25, "Career Fit", 0.90),
    (50, "Balanced", 0.80),
    (75, "Career Pivot", 0.60),
    (100, "Career Transition", 0.40),
]


def get_mode(career_direction):
    return min(MODES, key=lambda m: abs(m[0] - career_direction))


# Source badges used in section headers
SOURCE_SSOC = "<span style='font-size:0.7rem;color:#e67e22;border:1px solid #e67e22;border-radius:3px;padding:0 6px;margin-left:8px;white-space:nowrap'>SSOC 2024</span>"
SOURCE_JOBS = "<span style='font-size:0.7rem;color:#1a73e8;border:1px solid #1a73e8;border-radius:3px;padding:0 6px;margin-left:8px;white-space:nowrap'>Job Posts</span>"
SOURCE_CALC = "<span style='font-size:0.7rem;color:#888;border:1px solid #888;border-radius:3px;padding:0 6px;margin-left:8px;white-space:nowrap'>Calculated</span>"
