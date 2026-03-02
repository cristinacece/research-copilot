# -*- coding: utf-8 -*-
"""
CSS styles for the academic theme.
Navy blue / gold palette, serif fonts for abstracts.
"""

ACADEMIC_CSS = """
<style>
/* ── Global ── */
:root {
    --navy:  #1a2744;
    --gold:  #c9a84c;
    --cream: #faf7f0;
    --slate: #4a5568;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--navy) !important;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--gold) !important;
}

/* Main header */
h1 {
    color: var(--navy);
    font-family: 'Georgia', serif;
    border-bottom: 3px solid var(--gold);
    padding-bottom: 0.3rem;
}
h2, h3 {
    color: var(--navy);
}

/* Abstract / blockquote blocks */
.abstract-block {
    background-color: var(--cream);
    border-left: 4px solid var(--gold);
    padding: 0.8rem 1.2rem;
    font-family: 'Georgia', serif;
    font-size: 0.92rem;
    line-height: 1.6;
    color: var(--slate);
    border-radius: 0 6px 6px 0;
    margin: 0.5rem 0;
}

/* Citation blocks */
.citation-block {
    background-color: #f0f4ff;
    border: 1px solid #c0ccee;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    font-family: 'Georgia', serif;
    color: var(--slate);
    margin: 0.3rem 0;
}

/* Similarity badge */
.sim-badge {
    display: inline-block;
    background: var(--gold);
    color: var(--navy);
    font-weight: bold;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin-left: 6px;
}

/* Strategy selector label */
.strategy-label {
    font-size: 0.8rem;
    color: var(--slate);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, var(--navy), #2d4a8a);
    color: white;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    text-align: center;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: bold;
    color: var(--gold);
}
.metric-card .label {
    font-size: 0.85rem;
    opacity: 0.85;
}
</style>
"""
