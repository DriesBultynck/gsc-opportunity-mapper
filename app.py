import io
import json
import zipfile
import re
from collections import Counter
from urllib.parse import urlparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(page_title="GSC Opportunity Mapper (Exec-friendly)", layout="wide")


# ----------------------------
# Helpers
# ----------------------------
def parse_ctr(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan


def normalize_query(q: str) -> str:
    q = str(q).lower().strip()
    q = re.sub(r"[’']", "", q)
    q = re.sub(r"[^a-z0-9\s\-]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def url_to_tokens(url: str):
    try:
        p = urlparse(str(url))
        path = p.path.strip("/")
    except:
        path = str(url)
    if path == "":
        return ["home"]
    toks = re.split(r"[\/\-\_]+", path)
    toks = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in toks if t]
    toks = [t for t in toks if t and t not in {"amp", "utm", "http", "https", "www", "com"}]
    return toks


def slug_from_url(u: str) -> str:
    try:
        p = urlparse(str(u))
        s = p.path.strip("/")
        return s if s else "home"
    except:
        return str(u)[:60]


def pos_bucket(p):
    if pd.isna(p):
        return "unknown"
    if p <= 3:
        return "1-3"
    if p <= 6:
        return "4-6"
    if p <= 10:
        return "7-10"
    if p <= 20:
        return "11-20"
    return "21+"


CTR_FLOOR = {"1-3": 0.15, "4-6": 0.08, "7-10": 0.04, "11-20": 0.015, "21+": 0.005, "unknown": 0.01}


def expected_ctr_with_floor(p, bucket_median, overall_median):
    b = pos_bucket(p)
    med = bucket_median.get(b, overall_median)
    if pd.isna(med):
        med = overall_median if not pd.isna(overall_median) else 0.01
    return max(med, CTR_FLOOR.get(b, 0.01))


# ----------------------------
# Intent rules (client-agnostic, brand input)
# ----------------------------
def build_intent_rules(brand_terms, custom_rules=None):
    """
    brand_terms: list of strings like ["acme", "acme ltd"]
    custom_rules: optional dict {intent: [regex,...]} to REPLACE defaults for that intent
    """
    bt = [re.escape(t.strip().lower()) for t in brand_terms if t and t.strip()]
    brand_regexes = []
    for term in bt:
        brand_regexes.append(rf"\b{term}\b")

    default_rules = {
        "navigational": brand_regexes + [
            r"\blogin\b",
            r"\bsign[\s\-]?in\b",
            r"\bsign[\s\-]?up\b",
            r"\bcontact\b",
            r"\bcustomer service\b",
            r"\bphone number\b",
        ],
        "transactional": [
            r"\bbuy\b",
            r"\border\b",
            r"\bpricing\b",
            r"\bprice\b",
            r"\bcost\b",
            r"\bquote\b",
            r"\bbook(ing)?\b",
            r"\bappointment\b",
            r"\brequest\b",
            r"\bdemo\b",
            r"\bhire\b",
            r"\bagency\b",
            r"\bservice(s)?\b",
            r"\bconsultant\b",
            r"\bcompany\b",
            r"\bnear me\b",
            r"\b(in|near)\s+[a-z]{3,}\b",
        ],
        "commercial": [
            r"\bbest\b",
            r"\btop\b",
            r"\breview(s)?\b",
            r"\bvs\b",
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\balternative(s)?\b",
            r"\brating(s)?\b",
        ],
        "informational": [
            r"\bhow\b",
            r"\bwhat\b",
            r"\bwhy\b",
            r"\bguide\b",
            r"\btutorial\b",
            r"\bdefinition\b",
            r"\bmeaning\b",
            r"\bexample(s)?\b",
            r"\btips\b",
        ],
    }

    if custom_rules:
        for k, v in custom_rules.items():
            if k in default_rules and isinstance(v, list) and v:
                default_rules[k] = v  # replace entirely

    return default_rules


def compile_rules(intent_rules):
    compiled = {}
    for intent, patterns in intent_rules.items():
        compiled[intent] = [re.compile(p, flags=re.IGNORECASE) for p in patterns if p and p.strip()]
    return compiled


def classify_intent(q_norm, compiled_rules):
    matched = []
    for intent, regs in compiled_rules.items():
        for rgx in regs:
            if rgx.search(q_norm):
                matched.append(intent)
                break

    if "navigational" in matched:
        intent = "navigational"
    elif "transactional" in matched:
        intent = "transactional"
    elif "commercial" in matched:
        intent = "commercial"
    elif "informational" in matched:
        intent = "informational"
    else:
        intent = "informational"

    conf = min(0.55 + 0.15 * len(matched), 0.95)
    return intent, conf, matched


# ----------------------------
# Charts (bigger + labels)
# ----------------------------
def chart_leaderboard_stacked(df_pages, topn=20, title_suffix="", label_col="slug"):
    d = df_pages.sort_values("opportunity_clicks", ascending=False).head(topn).copy()
    labels = d[label_col].astype(str)

    actual = d["actual_clicks"].fillna(0).values
    opp = d["opportunity_clicks"].fillna(0).values

    fig = plt.figure(figsize=(14, 9))
    y = np.arange(len(d))

    plt.barh(y, actual, label="Actual clicks")
    plt.barh(y, opp, left=actual, label="Opportunity clicks (est.)")

    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Clicks (actual + opportunity)")
    plt.title(f"Top {topn} pages: actual vs opportunity{title_suffix}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return fig


def chart_scatter_labeled(df_pages, topn=50, label_top=10, title_suffix="", label_col="slug"):
    d = df_pages.sort_values("opportunity_clicks", ascending=False).head(topn).copy()

    fig = plt.figure(figsize=(14, 8))
    plt.scatter(d["avg_position"], d["opportunity_clicks"])

    plt.xlabel("Avg position (lower is better)")
    plt.ylabel("Opportunity clicks (estimated)")
    plt.title(f"Impact vs Effort (top {topn}; labels = top {label_top}){title_suffix}")

    if label_top and label_top > 0:
        d_lab = d.head(label_top)
        for _, r in d_lab.iterrows():
            plt.annotate(str(r[label_col]), (r["avg_position"], r["opportunity_clicks"]))

    plt.tight_layout()
    return fig


def chart_ctr_vs_position_labeled(df_pages, topn=200, label_top=12, title_suffix="", label_col="slug"):
    d = df_pages.sort_values("impressions", ascending=False).head(topn).copy()

    fig = plt.figure(figsize=(14, 8))
    plt.scatter(d["avg_position"], d["ctr"])

    plt.xlabel("Avg position")
    plt.ylabel("CTR")
    plt.title(f"CTR vs Position (top {topn} by impressions; labels = top {label_top}){title_suffix}")

    if label_top and label_top > 0:
        d_lab = d.sort_values("opportunity_clicks", ascending=False).head(label_top)
        for _, r in d_lab.iterrows():
            plt.annotate(str(r[label_col]), (r["avg_position"], r["ctr"]))

    plt.tight_layout()
    return fig


# ----------------------------
# GPT brief generators (JSON + Markdown)
# ----------------------------
def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except:
        return None


def safe_int(x):
    try:
        if pd.isna(x):
            return 0
        return int(round(float(x)))
    except:
        return 0


def md_table(headers, rows, max_col_width=80):
    # Simple markdown table generator (no external libs)
    def trunc(s):
        s = str(s)
        return s if len(s) <= max_col_width else (s[: max_col_width - 1] + "…")

    hdr = "| " + " | ".join([trunc(h) for h in headers]) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join(["| " + " | ".join([trunc(c) for c in r]) + " |" for r in rows])
    return "\n".join([hdr, sep, body])


def build_gpt_brief(
    q_out: pd.DataFrame,
    cluster_out: pd.DataFrame,
    page_opps_all: pd.DataFrame,
    page_opps_intent: pd.DataFrame,
    brand_terms: list,
    client_name: str = "",
    min_impressions: int = 100,
    top_pages: int = 20,
    top_clusters: int = 20,
    queries_per_cluster: int = 5,
):
    # Filter pages for headline KPIs & top lists
    pages_f = page_opps_all.copy()
    pages_f = pages_f[pages_f["impressions"].fillna(0) >= float(min_impressions)].copy()
    pages_f["slug"] = pages_f["slug"].fillna(pages_f["page"].apply(slug_from_url))
    pages_f["ctr"] = pages_f["ctr"].fillna(pages_f.get("page_ctr"))
    pages_f["avg_position"] = pages_f["avg_position"].fillna(pages_f.get("page_position"))

    total_clicks = pages_f["actual_clicks"].fillna(0).sum()
    total_opp = pages_f["opportunity_clicks"].fillna(0).sum()
    uplift_pct = (total_opp / total_clicks * 100.0) if total_clicks > 0 else 0.0

    # Top pages
    top_pages_df = pages_f.sort_values("opportunity_clicks", ascending=False).head(top_pages).copy()
    top_pages_list = []
    for _, r in top_pages_df.iterrows():
        top_pages_list.append(
            {
                "url": str(r.get("page")),
                "slug": str(r.get("slug")),
                "mapped_impressions": safe_int(r.get("impressions")),
                "mapped_clicks": safe_int(r.get("actual_clicks")),
                "opportunity_clicks_est": safe_int(r.get("opportunity_clicks")),
                "avg_position": safe_float(r.get("avg_position")),
                "ctr": safe_float(r.get("ctr")),
                "page_export_clicks": safe_int(r.get("page_clicks")),
                "page_export_impressions": safe_int(r.get("page_impressions")),
                "page_export_ctr": safe_float(r.get("page_ctr")),
                "page_export_position": safe_float(r.get("page_position")),
            }
        )

    # Top clusters (use opportunity_clicks first; then priority_score if present)
    cl = cluster_out.copy()
    sort_cols = ["opportunity_clicks"]
    if "priority_score" in cl.columns:
        sort_cols = ["opportunity_clicks", "priority_score"]
    cl_top = cl.sort_values(sort_cols, ascending=False).head(top_clusters).copy()

    # Sample queries per cluster (top impressions)
    q_samp = (
        q_out.sort_values(["cluster_id", "impressions"], ascending=[True, False])
        .groupby("cluster_id")
        .head(queries_per_cluster)
        .copy()
    )
    samples = {}
    for cid, sub in q_samp.groupby("cluster_id"):
        samples[str(cid)] = [
            {
                "query": str(r["query"]),
                "intent": str(r.get("intent")),
                "impressions": safe_int(r.get("impressions")),
                "clicks": safe_int(r.get("clicks")),
                "position": safe_float(r.get("position")),
                "ctr": safe_float(r.get("ctr")),
                "opportunity_clicks_est": safe_int(r.get("opportunity_clicks")),
                "primary_page": str(r.get("primary_page")) if "primary_page" in sub.columns else None,
            }
            for _, r in sub.iterrows()
        ]

    top_clusters_list = []
    for _, r in cl_top.iterrows():
        cid = str(r.get("cluster_id"))
        top_clusters_list.append(
            {
                "cluster_id": cid,
                "topic_label": str(r.get("topic_label")),
                "dominant_intent": str(r.get("dominant_intent")),
                "mapped_page": str(r.get("primary_page")),
                "match_score": safe_float(r.get("match_score")),
                "queries": safe_int(r.get("queries")),
                "clicks": safe_int(r.get("clicks")),
                "impressions": safe_int(r.get("impressions")),
                "avg_position": safe_float(r.get("avg_position")),
                "opportunity_clicks_est": safe_int(r.get("opportunity_clicks")),
                "priority_band": str(r.get("priority_band")),
                "recommended_action": str(r.get("recommended_action")),
                "sample_queries": samples.get(cid, []),
            }
        )

    # Cannibalisation list
    cann = cluster_out[cluster_out.get("cannibalisation_risk", False) == True].copy()
    cann = cann.sort_values(["opportunity_clicks"], ascending=False).head(50)
    cann_list = []
    for _, r in cann.iterrows():
        cann_list.append(
            {
                "cluster_id": str(r.get("cluster_id")),
                "topic_label": str(r.get("topic_label")),
                "dominant_intent": str(r.get("dominant_intent")),
                "primary_page": str(r.get("primary_page")),
                "runner_up_page": str(r.get("runner_up_page")),
                "match_score": safe_float(r.get("match_score")),
                "runner_up_score": safe_float(r.get("runner_up_score")),
                "recommended_action": str(r.get("recommended_action")),
            }
        )

    # Intent breakdown totals + top pages per intent (mapped)
    intent_breakdown = {}
    for intent in ["informational", "commercial", "transactional", "navigational"]:
        di = page_opps_intent[page_opps_intent["intent"] == intent].copy()
        if len(di) == 0:
            intent_breakdown[intent] = {"totals": {}, "top_pages": []}
            continue
        di = di[di["impressions"].fillna(0) >= float(min_impressions)].copy()
        di["slug"] = di["slug"].fillna(di["page"].apply(slug_from_url))
        totals = {
            "mapped_clicks": safe_int(di["actual_clicks"].fillna(0).sum()),
            "mapped_impressions": safe_int(di["impressions"].fillna(0).sum()),
            "opportunity_clicks_est": safe_int(di["opportunity_clicks"].fillna(0).sum()),
        }
        di_top = di.sort_values("opportunity_clicks", ascending=False).head(10)
        top_list = []
        for _, r in di_top.iterrows():
            top_list.append(
                {
                    "url": str(r.get("page")),
                    "slug": str(r.get("slug")),
                    "mapped_impressions": safe_int(r.get("impressions")),
                    "mapped_clicks": safe_int(r.get("actual_clicks")),
                    "opportunity_clicks_est": safe_int(r.get("opportunity_clicks")),
                    "avg_position": safe_float(r.get("avg_position")),
                    "ctr": safe_float(r.get("ctr")),
                }
            )
        intent_breakdown[intent] = {"totals": totals, "top_pages": top_list}

    brief = {
        "meta": {
            "client_name": client_name.strip(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "brand_terms": brand_terms,
            "min_impressions_filter": int(min_impressions),
            "definitions": {
                "mapped_clicks/impressions": "Aggregated from query→cluster→page mapping (not necessarily equal to page totals in GSC Pages export).",
                "opportunity_clicks_est": "Estimated extra clicks based on CTR gap vs expected CTR by position (heuristic baseline).",
                "intent": "Rule-based classification (brand terms affect navigational).",
            },
            "rules_of_use_for_llm": [
                "Use ONLY the data in this file. Do not invent numbers.",
                "If something is missing/unclear, list assumptions explicitly.",
                "Output should be an SEO action plan grounded in the top pages/clusters here.",
            ],
        },
        "kpis": {
            "mapped_clicks_total": safe_int(total_clicks),
            "opportunity_clicks_est_total": safe_int(total_opp),
            "estimated_uplift_pct": round(float(uplift_pct), 2),
        },
        "top_pages": top_pages_list,
        "top_clusters": top_clusters_list,
        "cannibalisation": cann_list,
        "intent_breakdown": intent_breakdown,
    }

    # Markdown companion
    md_lines = []
    md_lines.append("# SEO Opportunity Brief (for Custom GPT)")
    if client_name.strip():
        md_lines.append(f"**Client:** {client_name.strip()}")
    md_lines.append(f"**Generated (UTC):** {brief['meta']['generated_at_utc']}")
    md_lines.append(f"**Brand terms:** {', '.join(brand_terms) if brand_terms else '(none provided)'}")
    md_lines.append(f"**Min impressions filter:** {min_impressions}")
    md_lines.append("")
    md_lines.append("## Headline KPIs (mapped)")
    md_lines.append(
        f"- Mapped clicks: **{brief['kpis']['mapped_clicks_total']:,}**\n"
        f"- Opportunity clicks (est.): **{brief['kpis']['opportunity_clicks_est_total']:,}**\n"
        f"- Estimated upside: **{brief['kpis']['estimated_uplift_pct']:.2f}%**"
    )
    md_lines.append("")
    md_lines.append("## Top pages (by opportunity clicks)")
    tp_rows = []
    for p in brief["top_pages"]:
        tp_rows.append(
            [
                p["slug"],
                f"{p['mapped_clicks']:,}",
                f"{p['opportunity_clicks_est']:,}",
                f"{p['mapped_impressions']:,}",
                f"{p['avg_position'] if p['avg_position'] is not None else ''}",
                f"{round(p['ctr']*100,2) if p['ctr'] is not None else ''}%",
            ]
        )
    md_lines.append(
        md_table(
            ["page", "clicks", "opp clicks", "impr", "pos", "ctr"],
            tp_rows,
            max_col_width=60
        )
    )
    md_lines.append("")
    md_lines.append("## Top clusters (by opportunity clicks)")
    tc_rows = []
    for c in brief["top_clusters"][: min(len(brief["top_clusters"]), 15)]:
        tc_rows.append(
            [
                c["topic_label"],
                c["dominant_intent"],
                slug_from_url(c["mapped_page"]),
                f"{c['opportunity_clicks_est']:,}",
                c["priority_band"],
            ]
        )
    md_lines.append(
        md_table(
            ["topic", "intent", "mapped page", "opp clicks", "band"],
            tc_rows,
            max_col_width=50
        )
    )
    md_lines.append("")
    md_lines.append("## Cannibalisation risks (top)")
    if brief["cannibalisation"]:
        cann_rows = []
        for c in brief["cannibalisation"][:10]:
            cann_rows.append(
                [
                    c["topic_label"],
                    slug_from_url(c["primary_page"]),
                    slug_from_url(c["runner_up_page"]),
                    c["recommended_action"],
                ]
            )
        md_lines.append(md_table(["topic", "primary", "runner-up", "action"], cann_rows, max_col_width=60))
    else:
        md_lines.append("_None detected in top set._")
    md_lines.append("")
    md_lines.append("## Intent breakdown (totals)")
    ib_rows = []
    for intent, d in brief["intent_breakdown"].items():
        totals = d.get("totals", {}) or {}
        ib_rows.append(
            [
                intent,
                f"{totals.get('mapped_clicks', 0):,}",
                f"{totals.get('mapped_impressions', 0):,}",
                f"{totals.get('opportunity_clicks_est', 0):,}",
            ]
        )
    md_lines.append(md_table(["intent", "clicks", "impr", "opp clicks"], ib_rows))
    md_lines.append("")
    md_lines.append("## Instructions for the Custom GPT")
    md_lines.append(
        "- Produce: executive summary (3 bullets), top 20 pages with actions, top clusters with actions, cannibalisation fixes, and a 30/60/90-day plan.\n"
        "- Use only data from this brief. If you make assumptions, list them."
    )

    return brief, "\n".join(md_lines)


# ----------------------------
# Core pipeline
# ----------------------------
@st.cache_data(show_spinner=False)
def run_pipeline(queries_df: pd.DataFrame, pages_df: pd.DataFrame, compiled_rules):
    # ---- ingest queries
    q = queries_df.rename(
        columns={"Top queries": "query", "Clicks": "clicks", "Impressions": "impressions", "CTR": "ctr_raw", "Position": "position"}
    ).copy()

    q["clicks"] = pd.to_numeric(q["clicks"], errors="coerce").fillna(0).astype(float)
    q["impressions"] = pd.to_numeric(q["impressions"], errors="coerce").fillna(0).astype(float)
    q["ctr"] = q["ctr_raw"].apply(parse_ctr)
    q.loc[q["ctr"].isna() & (q["impressions"] > 0), "ctr"] = q["clicks"] / q["impressions"]
    q["position"] = pd.to_numeric(q["position"], errors="coerce")
    q["query_norm"] = q["query"].apply(normalize_query)

    q[["intent", "intent_confidence", "intent_hits"]] = q["query_norm"].apply(
        lambda s: pd.Series(classify_intent(s, compiled_rules))
    )

    # ---- semantic clustering
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_df=0.95)
    X = vec.fit_transform(q["query_norm"].tolist())
    simdist = 1 - cosine_similarity(X)

    cl = AgglomerativeClustering(metric="precomputed", linkage="average", distance_threshold=0.90, n_clusters=None)
    labels = cl.fit_predict(simdist)
    q["cluster_id"] = labels

    # ---- topic label via common n-grams
    def tokens(qn):
        return [t for t in qn.split() if t]

    def ngrams(toks, n):
        return [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]

    q["tokens"] = q["query_norm"].apply(tokens)

    topic = {}
    for cid, sub in q.groupby("cluster_id"):
        cand = Counter()
        toks_list = sub["tokens"].tolist()
        for toks in toks_list:
            for n in (3, 2):
                for g in ngrams(toks, n):
                    cand[g] += 1

        best = None
        for g, cnt in cand.most_common(100):
            if cnt >= max(2, int(0.3 * len(sub))):
                score = cnt * len(g.split())
                if best is None or score > best[0]:
                    best = (score, g)
        topic[cid] = best[1] if best else (toks_list[0][0] if toks_list and toks_list[0] else "misc")

    q["topic_label"] = q["cluster_id"].map(topic)

    # ---- expected ctr baseline + opportunity clicks + priority
    q["pos_bucket"] = q["position"].apply(pos_bucket)
    bucket_median = q.groupby("pos_bucket")["ctr"].median().to_dict()
    overall_median = q["ctr"].median()

    q["expected_ctr"] = q["position"].apply(lambda p: expected_ctr_with_floor(p, bucket_median, overall_median))
    q["opportunity_clicks"] = (q["impressions"] * np.maximum(0, q["expected_ctr"] - q["ctr"])).fillna(0)

    q["priority_score"] = (
        np.log1p(q["clicks"]) * 1.0
        + np.log1p(q["opportunity_clicks"]) * 1.2
        + np.log1p(q["impressions"]) * 0.4
    )
    q70, q90 = np.quantile(q["priority_score"], [0.70, 0.90])
    q["priority_band"] = q["priority_score"].apply(lambda s: "P1" if s >= q90 else ("P2" if s >= q70 else "P3"))

    # ---- cluster summary
    cluster = (
        q.groupby(["cluster_id", "topic_label"])
        .agg(
            queries=("query", "count"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            avg_position=("position", "mean"),
            opportunity_clicks=("opportunity_clicks", "sum"),
            priority_score=("priority_score", "mean"),
        )
        .reset_index()
    )

    intent_mix = q.pivot_table(index="cluster_id", columns="intent", values="impressions", aggfunc="sum", fill_value=0)
    dominant_intent = intent_mix.idxmax(axis=1).to_dict()
    cluster["dominant_intent"] = cluster["cluster_id"].map(dominant_intent)

    q70c, q90c = np.quantile(cluster["priority_score"], [0.70, 0.90])
    cluster["priority_band"] = cluster["priority_score"].apply(lambda s: "P1" if s >= q90c else ("P2" if s >= q70c else "P3"))

    # ---- ingest pages
    p = pages_df.rename(
        columns={"Top pages": "page", "Clicks": "page_clicks", "Impressions": "page_impressions", "CTR": "page_ctr_raw", "Position": "page_position"}
    ).copy()

    p["page_clicks"] = pd.to_numeric(p["page_clicks"], errors="coerce").fillna(0).astype(float)
    p["page_impressions"] = pd.to_numeric(p["page_impressions"], errors="coerce").fillna(0).astype(float)
    p["page_ctr"] = p["page_ctr_raw"].apply(parse_ctr)
    p.loc[p["page_ctr"].isna() & (p["page_impressions"] > 0), "page_ctr"] = p["page_clicks"] / p["page_impressions"]
    p["page_position"] = pd.to_numeric(p["page_position"], errors="coerce")
    p["slug"] = p["page"].apply(slug_from_url)
    p["page_tokens"] = p["page"].apply(url_to_tokens)
    p["page_text"] = p["page_tokens"].apply(lambda t: " ".join(t))

    # ---- cluster -> page matching (heuristic)
    rep = (
        q.sort_values(["cluster_id", "impressions"], ascending=[True, False])
        .groupby("cluster_id")
        .head(5)
        .groupby("cluster_id")["query_norm"]
        .apply(lambda s: " ".join(s.tolist()))
        .to_dict()
    )
    cluster["cluster_text"] = (cluster["topic_label"] + " " + cluster["dominant_intent"] + " " + cluster["cluster_id"].map(rep).fillna("")).str.lower()

    all_text = pd.concat(
        [
            cluster[["cluster_id", "cluster_text"]].rename(columns={"cluster_text": "text"}).assign(kind="cluster"),
            p[["page", "page_text"]].rename(columns={"page_text": "text"}).assign(kind="page"),
        ],
        ignore_index=True,
    )

    v2 = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
    V = v2.fit_transform(all_text["text"].tolist())

    cluster_idx = all_text.index[all_text["kind"] == "cluster"].to_numpy()
    page_idx = all_text.index[all_text["kind"] == "page"].to_numpy()

    Vc = V[cluster_idx]
    Vp = V[page_idx]
    sim = cosine_similarity(Vc, Vp)

    best = sim.argmax(axis=1)
    cluster["primary_page"] = [p.iloc[j]["page"] for j in best]
    cluster["match_score"] = sim.max(axis=1)

    top2 = np.argsort(-sim, axis=1)[:, :2]
    cluster["runner_up_page"] = [p.iloc[j]["page"] for j in top2[:, 1]]
    cluster["runner_up_score"] = np.take_along_axis(sim, top2, axis=1)[:, 1]
    cluster["cannibalisation_risk"] = (cluster["match_score"] - cluster["runner_up_score"] < 0.03) & (cluster["match_score"] > 0.12)

    def action(row):
        if row["cannibalisation_risk"]:
            return "Fix cannibalisation: pick 1 winner page + align internal links"
        if row["dominant_intent"] == "transactional" and row["avg_position"] > 10:
            return "Build/upgrade landing page + strengthen internal links"
        if row["avg_position"] <= 10:
            return "CTR/snippet + internal links (quick win)"
        return "Content refresh + topical authority (hub/spokes)"

    cluster["recommended_action"] = cluster.apply(action, axis=1)

    # ---- map each query to its primary page (via cluster)
    cluster_to_page = cluster.set_index("cluster_id")["primary_page"].to_dict()
    q["primary_page"] = q["cluster_id"].map(cluster_to_page)

    # ---- ALL intents: page opportunity (based on mapped clusters)
    page_opportunity_all = (
        cluster.groupby("primary_page")
        .agg(
            clusters=("cluster_id", "count"),
            impressions=("impressions", "sum"),
            actual_clicks=("clicks", "sum"),
            opportunity_clicks=("opportunity_clicks", "sum"),
            avg_position=("avg_position", "mean"),
        )
        .reset_index()
        .rename(columns={"primary_page": "page"})
    )

    page_opportunity_all = page_opportunity_all.merge(
        p[["page", "slug", "page_impressions", "page_clicks", "page_ctr", "page_position"]],
        on="page",
        how="left",
    )
    page_opportunity_all["ctr"] = page_opportunity_all["actual_clicks"] / page_opportunity_all["impressions"].replace(0, np.nan)

    # ---- PER INTENT: page opportunity (query-level mapped)
    page_opportunity_intent = (
        q.groupby(["primary_page", "intent"])
        .agg(
            impressions=("impressions", "sum"),
            actual_clicks=("clicks", "sum"),
            opportunity_clicks=("opportunity_clicks", "sum"),
            avg_position=("position", "mean"),
        )
        .reset_index()
        .rename(columns={"primary_page": "page"})
    )
    page_opportunity_intent["ctr"] = page_opportunity_intent["actual_clicks"] / page_opportunity_intent["impressions"].replace(0, np.nan)

    page_opportunity_intent = page_opportunity_intent.merge(
        p[["page", "slug", "page_clicks", "page_impressions", "page_ctr", "page_position"]],
        on="page",
        how="left",
    )

    return q, cluster, p, page_opportunity_all, page_opportunity_intent


# ----------------------------
# UI
# ----------------------------
st.title("GSC Opportunity Mapper (Exec-friendly)")

st.sidebar.header("Client settings")
client_name = st.sidebar.text_input("Client name (optional)", value="")

brand_terms_input = st.sidebar.text_input(
    "Brand terms for navigational intent (comma-separated)",
    value="",
    help="Example: acme, acme ltd, acme widgets (leave empty if unknown)",
)
brand_terms = [t.strip() for t in brand_terms_input.split(",") if t.strip()]

# Keep advanced custom rules optional
use_custom_rules = st.sidebar.checkbox("Use custom intent regex rules (advanced)", value=False)
custom_rules = None
if use_custom_rules:
    st.sidebar.caption("One regex per line. These REPLACE defaults for that intent.")
    nav_text = st.sidebar.text_area("Navigational regex", value="")
    txn_text = st.sidebar.text_area("Transactional regex", value="")
    com_text = st.sidebar.text_area("Commercial regex", value="")
    inf_text = st.sidebar.text_area("Informational regex", value="")

    def lines_to_list(s):
        return [ln.strip() for ln in s.splitlines() if ln.strip()]

    custom_rules = {}
    if nav_text.strip():
        custom_rules["navigational"] = lines_to_list(nav_text)
    if txn_text.strip():
        custom_rules["transactional"] = lines_to_list(txn_text)
    if com_text.strip():
        custom_rules["commercial"] = lines_to_list(com_text)
    if inf_text.strip():
        custom_rules["informational"] = lines_to_list(inf_text)

intent_rules = build_intent_rules(brand_terms=brand_terms, custom_rules=custom_rules)
compiled_rules = compile_rules(intent_rules)

c1, c2 = st.columns(2)
with c1:
    q_file = st.file_uploader("Upload GSC Queries CSV", type=["csv"])
with c2:
    p_file = st.file_uploader("Upload GSC Pages CSV", type=["csv"])

if not (q_file and p_file):
    st.info("Upload both CSVs to generate the opportunity map and charts.")
    st.stop()

queries_df = pd.read_csv(q_file)
pages_df = pd.read_csv(p_file)

with st.spinner("Running clustering + intent + mapping…"):
    q_out, cluster_out, pages_out, page_opps_all, page_opps_intent = run_pipeline(
        queries_df, pages_df, compiled_rules
    )

st.success("Done.")

INTENTS = ["All", "informational", "commercial", "transactional", "navigational"]

t1, t2, t3 = st.tabs(["Executive visuals", "Tables", "Download pack"])

with t1:
    st.subheader("Executive controls")

    sel_intent = st.selectbox("View by intent", INTENTS, index=0)
    min_impr = st.number_input("Min impressions (filter)", value=100, step=50, min_value=0)

    topn_bar = st.selectbox("Top N pages (stacked bar)", [15, 20, 25, 50], index=1)
    topn_scatter = st.selectbox("Top N pages (scatter)", [25, 50, 75, 100], index=1)
    label_top_scatter = st.selectbox("Scatter labels (top pages)", [0, 5, 10, 15], index=2)

    topn_ctr = st.selectbox("CTR vs Position: plot top N pages by impressions", [100, 200, 300, 500], index=1)
    label_top_ctr = st.selectbox("CTR vs Position labels (top by opportunity)", [0, 6, 12, 20], index=2)

    if sel_intent == "All":
        df = page_opps_all.copy()
        title_suffix = ""
    else:
        df = page_opps_intent[page_opps_intent["intent"] == sel_intent].copy()
        title_suffix = f" ({sel_intent})"

    df = df[df["impressions"].fillna(0) >= float(min_impr)].copy()
    df["slug"] = df["slug"].fillna(df["page"].apply(slug_from_url))
    df["ctr"] = df["ctr"].fillna(df.get("page_ctr"))
    df["avg_position"] = df["avg_position"].fillna(df.get("page_position"))

    # Executive KPIs
    total_clicks = df["actual_clicks"].fillna(0).sum()
    total_opp = df["opportunity_clicks"].fillna(0).sum()
    uplift_pct = (total_opp / total_clicks * 100) if total_clicks > 0 else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Actual clicks (mapped)", f"{total_clicks:,.0f}")
    k2.metric("Opportunity clicks (est.)", f"{total_opp:,.0f}")
    k3.metric("Estimated upside", f"{uplift_pct:,.1f}%")

    st.pyplot(chart_leaderboard_stacked(df, topn=int(topn_bar), title_suffix=title_suffix, label_col="slug"), clear_figure=True)
    st.pyplot(chart_scatter_labeled(df, topn=int(topn_scatter), label_top=int(label_top_scatter), title_suffix=title_suffix, label_col="slug"), clear_figure=True)
    st.pyplot(chart_ctr_vs_position_labeled(df, topn=int(topn_ctr), label_top=int(label_top_ctr), title_suffix=title_suffix, label_col="slug"), clear_figure=True)

    st.subheader("Cannibalisation risks (clusters)")
    cann = cluster_out[cluster_out["cannibalisation_risk"] == True].sort_values("priority_score", ascending=False)
    st.dataframe(cann[["topic_label", "dominant_intent", "primary_page", "runner_up_page", "match_score", "runner_up_score", "recommended_action"]])

with t2:
    st.subheader("Page opportunity (All)")
    st.dataframe(page_opps_all.sort_values("opportunity_clicks", ascending=False))

    st.subheader("Page opportunity by intent")
    i1, i2, i3, i4 = st.tabs(["informational", "commercial", "transactional", "navigational"])
    with i1:
        st.dataframe(page_opps_intent[page_opps_intent["intent"] == "informational"].sort_values("opportunity_clicks", ascending=False))
    with i2:
        st.dataframe(page_opps_intent[page_opps_intent["intent"] == "commercial"].sort_values("opportunity_clicks", ascending=False))
    with i3:
        st.dataframe(page_opps_intent[page_opps_intent["intent"] == "transactional"].sort_values("opportunity_clicks", ascending=False))
    with i4:
        st.dataframe(page_opps_intent[page_opps_intent["intent"] == "navigational"].sort_values("opportunity_clicks", ascending=False))

    st.subheader("Cluster → page map (sorted by priority)")
    st.dataframe(cluster_out.sort_values(["priority_band", "priority_score"], ascending=[True, False]))

    st.subheader("Query-level output (top 200 by priority)")
    st.dataframe(q_out.sort_values(["priority_band", "priority_score"], ascending=[True, False]).head(200))

with t3:
    st.write("Download a ZIP with outputs + charts + a Custom-GPT briefing pack (`gpt_brief.json` + `gpt_brief.md`).")

    # Defaults for the GPT pack
    gpt_min_impr = st.number_input("GPT brief: min impressions filter", value=100, step=50, min_value=0)
    gpt_top_pages = st.selectbox("GPT brief: top pages", [10, 20, 30, 50], index=1)
    gpt_top_clusters = st.selectbox("GPT brief: top clusters", [10, 20, 30, 50], index=1)
    gpt_q_per_cluster = st.selectbox("GPT brief: sample queries per cluster", [3, 5, 8, 10], index=1)

    brief_json, brief_md = build_gpt_brief(
        q_out=q_out,
        cluster_out=cluster_out,
        page_opps_all=page_opps_all,
        page_opps_intent=page_opps_intent,
        brand_terms=brand_terms,
        client_name=client_name,
        min_impressions=int(gpt_min_impr),
        top_pages=int(gpt_top_pages),
        top_clusters=int(gpt_top_clusters),
        queries_per_cluster=int(gpt_q_per_cluster),
    )

    # Show previews
    with st.expander("Preview: GPT brief (Markdown)"):
        st.markdown(brief_md)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
        # Data outputs
        z.writestr("gsc_query_clusters_hybrid.csv", q_out.to_csv(index=False))
        z.writestr("keyword_to_page_map_clusters.csv", cluster_out.to_csv(index=False))
        z.writestr("gsc_pages_clean.csv", pages_out.to_csv(index=False))
        z.writestr("page_opportunity_all.csv", page_opps_all.to_csv(index=False))
        z.writestr("page_opportunity_by_intent.csv", page_opps_intent.to_csv(index=False))

        # GPT pack
        z.writestr("gpt_brief.json", json.dumps(brief_json, indent=2))
        z.writestr("gpt_brief.md", brief_md)

        # Charts (All intent default, filtered for readability)
        df_default = page_opps_all.copy()
        df_default = df_default[df_default["impressions"].fillna(0) >= 100].copy()
        df_default["slug"] = df_default["slug"].fillna(df_default["page"].apply(slug_from_url))
        df_default["ctr"] = df_default["ctr"].fillna(df_default["page_ctr"])
        df_default["avg_position"] = df_default["avg_position"].fillna(df_default["page_position"])

        charts_to_save = [
            ("charts/top_pages_actual_vs_opportunity.png", chart_leaderboard_stacked(df_default, topn=20, title_suffix=" (All)")),
            ("charts/impact_vs_effort_scatter.png", chart_scatter_labeled(df_default, topn=50, label_top=10, title_suffix=" (All)")),
            ("charts/ctr_vs_position_labeled.png", chart_ctr_vs_position_labeled(df_default, topn=200, label_top=12, title_suffix=" (All)")),
        ]

        for intent in ["informational", "commercial", "transactional", "navigational"]:
            dfi = page_opps_intent[page_opps_intent["intent"] == intent].copy()
            if len(dfi) == 0:
                continue
            dfi = dfi[dfi["impressions"].fillna(0) >= 100].copy()
            dfi["slug"] = dfi["slug"].fillna(dfi["page"].apply(slug_from_url))
            dfi["ctr"] = dfi["ctr"].fillna(dfi["page_ctr"])
            dfi["avg_position"] = dfi["avg_position"].fillna(dfi["page_position"])
            charts_to_save.append(
                (f"charts/top_pages_actual_vs_opportunity_{intent}.png", chart_leaderboard_stacked(dfi, topn=20, title_suffix=f" ({intent})"))
            )

        for name, fig in charts_to_save:
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=160)
            plt.close(fig)
            z.writestr(name, img_buf.getvalue())

    st.download_button(
        "Download report pack (.zip)",
        data=zip_buf.getvalue(),
        file_name="gsc_opportunity_report_pack.zip",
        mime="application/zip",
    )

    # Optional: standalone downloads for GPT pack
    st.download_button(
        "Download gpt_brief.json",
        data=json.dumps(brief_json, indent=2).encode("utf-8"),
        file_name="gpt_brief.json",
        mime="application/json",
    )
    st.download_button(
        "Download gpt_brief.md",
        data=brief_md.encode("utf-8"),
        file_name="gpt_brief.md",
        mime="text/markdown",
    )
