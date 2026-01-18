import io
import json
import zipfile
import re
from collections import Counter
from urllib.parse import urlparse
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

# ----------------------------
# Configuration & Constants
# ----------------------------
st.set_page_config(page_title="GSC Opportunity Mapper (Stratified)", layout="wide")

CTR_FLOOR = {
    "1-3": 0.15,
    "4-6": 0.08,
    "7-10": 0.04,
    "11-20": 0.015,
    "21+": 0.005,
    "unknown": 0.01
}

PRIORITY_QUANTILES = (0.70, 0.90)
CLUSTERING_THRESHOLD_N = 10000  # Switch to MiniBatchKMeans if queries > this

# Action effort ordering (1 = low effort, 4 = high effort)
ACTION_EFFORT_ORDER = {
    "CTR/snippet + internal links": 1,  # Low effort
    "Content refresh + topical authority": 2,  # Medium effort
    "Fix cannibalisation": 3,  # Medium-High effort
    "Build/upgrade landing page": 4  # High effort
}

# Required field mappings for multi-language support
REQUIRED_QUERY_FIELDS = {
    "query": ["query", "zoekopdracht", "requête", "consulta", "anfrage", "top queries", "queries", "zoekterm"],
    "clicks": ["clicks", "klikken", "clics", "clic", "klicks", "click"],
    "impressions": ["impressions", "weergaven", "impressions", "impresiones", "impressionen", "weergave"],
    "ctr": ["ctr", "click-through rate", "taux de clic", "tasa de clics", "clickrate"],
    "position": ["position", "pos", "posizione", "posición", "position", "rang", "rank"]
}

REQUIRED_PAGE_FIELDS = {
    "page": ["page", "url", "pagina", "página", "seite", "top pages", "pages"],
    "clicks": ["clicks", "klikken", "clics", "clic", "klicks", "click"],
    "impressions": ["impressions", "weergaven", "impressions", "impresiones", "impressionen", "weergave"],
    "ctr": ["ctr", "click-through rate", "taux de clic", "tasa de clics", "clickrate"],
    "position": ["position", "pos", "posizione", "posición", "position", "rang", "rank"]
}

# ----------------------------
# Helpers
# ----------------------------
def parse_ctr(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

def normalize_query_vectorized(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"[’']", "", regex=True)
        .str.replace(r"[^a-z0-9\s\-]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def url_to_tokens(url: str) -> List[str]:
    try:
        p = urlparse(str(url))
        path = p.path.strip("/")
    except Exception:
        path = str(url)
    if not path:
        return ["home"]
    toks = re.split(r"[\/\-\_]+", path)
    clean_toks = []
    for t in toks:
        if not t:
            continue
        t_clean = re.sub(r"[^a-z0-9]", "", t.lower())
        if t_clean and t_clean not in {"amp", "utm", "http", "https", "www", "com"}:
            clean_toks.append(t_clean)
    return clean_toks

def slug_from_url(u: str) -> str:
    try:
        p = urlparse(str(u))
        s = p.path.strip("/")
        return s if s else "home"
    except Exception:
        return str(u)[:60]

def pos_bucket(p: float) -> str:
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

def expected_ctr_with_floor(p: float, bucket_median: Dict[str, float], overall_median: float) -> float:
    b = pos_bucket(p)
    med = bucket_median.get(b, overall_median)
    if pd.isna(med):
        med = overall_median if not pd.isna(overall_median) else 0.01
    return max(med, CTR_FLOOR.get(b, 0.01))

def safe_int(x):
    try:
        if pd.isna(x): return 0
        return int(round(float(x)))
    except: return 0

def safe_float(x):
    try:
        if pd.isna(x): return None
        return float(x)
    except: return None

def md_table(headers, rows, max_col_width=80):
    def trunc(s):
        s = str(s)
        return s if len(s) <= max_col_width else (s[: max_col_width - 1] + "…")
    hdr = "| " + " | ".join([trunc(h) for h in headers]) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join(["| " + " | ".join([trunc(c) for c in r]) + " |" for r in rows])
    return "\n".join([hdr, sep, body])

def detect_column_mapping(df: pd.DataFrame, required_fields: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    """
    Auto-detect column mappings based on common patterns and translations.
    
    Args:
        df: DataFrame with CSV columns
        required_fields: Dict mapping internal field name to list of possible column name patterns
        
    Returns:
        Dict mapping internal field name to detected column name (or None if not found)
    """
    mapping = {}
    df_columns_lower = {col.lower().strip(): col for col in df.columns}
    
    for field_name, patterns in required_fields.items():
        detected = None
        for pattern in patterns:
            pattern_lower = pattern.lower().strip()
            # Exact match
            if pattern_lower in df_columns_lower:
                detected = df_columns_lower[pattern_lower]
                break
            # Partial match (contains pattern)
            for col_lower, col_original in df_columns_lower.items():
                if pattern_lower in col_lower or col_lower in pattern_lower:
                    detected = col_original
                    break
            if detected:
                break
        mapping[field_name] = detected
    
    return mapping

def validate_column_mapping(mapping: Dict[str, Optional[str]], required_fields: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that all required fields have mappings.
    
    Args:
        mapping: Dict mapping internal field name to column name (or None)
        required_fields: List of required field names
        
    Returns:
        Tuple of (is_valid, missing_fields)
    """
    missing = [field for field in required_fields if not mapping.get(field)]
    return (len(missing) == 0, missing)

# ----------------------------
# Intent & Brand Logic
# ----------------------------
def build_intent_rules(brand_terms: List[str], custom_rules: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    bt = [re.escape(t.strip().lower()) for t in brand_terms if t and t.strip()]
    brand_regexes = [rf"\b{term}\b" for term in bt]

    default_rules = {
        "navigational": brand_regexes + [
            r"\blogin\b", r"\bsign[\s\-]?in\b", r"\bsign[\s\-]?up\b",
            r"\bcontact\b", r"\bcustomer service\b", r"\bphone number\b",
        ],
        "transactional": [
            r"\bbuy\b", r"\border\b", r"\bpricing\b", r"\bprice\b", r"\bcost\b",
            r"\bquote\b", r"\bbook(ing)?\b", r"\bappointment\b", r"\brequest\b",
            r"\bdemo\b", r"\bhire\b", r"\bagency\b", r"\bservice(s)?\b",
            r"\bconsultant\b", r"\bcompany\b", r"\bnear me\b",
            r"\b(in|near)\s+[a-z]{3,}\b",
        ],
        "commercial": [
            r"\bbest\b", r"\btop\b", r"\breview(s)?\b", r"\bvs\b",
            r"\bcompare\b", r"\bcomparison\b", r"\balternative(s)?\b",
            r"\brating(s)?\b",
        ],
        "informational": [
            r"\bhow\b", r"\bwhat\b", r"\bwhy\b", r"\bguide\b",
            r"\btutorial\b", r"\bdefinition\b", r"\bmeaning\b",
            r"\bexample(s)?\b", r"\btips\b",
        ],
    }
    if custom_rules:
        for k, v in custom_rules.items():
            if k in default_rules and isinstance(v, list) and v:
                default_rules[k] = v
    return default_rules

def compile_rules(intent_rules: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    compiled = {}
    for intent, patterns in intent_rules.items():
        compiled[intent] = [re.compile(p, flags=re.IGNORECASE) for p in patterns if p and p.strip()]
    return compiled

def classify_intent_row(q_norm: str, compiled_rules: Dict[str, List[re.Pattern]]) -> Tuple[str, float, List[str]]:
    matched = []
    for intent, regs in compiled_rules.items():
        for rgx in regs:
            if rgx.search(q_norm):
                matched.append(intent)
                break
    
    if "navigational" in matched: intent = "navigational"
    elif "transactional" in matched: intent = "transactional"
    elif "commercial" in matched: intent = "commercial"
    elif "informational" in matched: intent = "informational"
    else: intent = "informational"

    conf = min(0.55 + 0.15 * len(matched), 0.95)
    return intent, conf, matched

def check_branded(q_norm: str, brand_regexes: List[re.Pattern]) -> bool:
    for rgx in brand_regexes:
        if rgx.search(q_norm):
            return True
    return False

# ----------------------------
# Pipeline Steps
# ----------------------------

def preprocess_queries(df: pd.DataFrame, compiled_rules: Dict[str, List[re.Pattern]], brand_regexes: List[re.Pattern], column_mapping: Dict[str, str]) -> pd.DataFrame:
    # Validate that all required columns are mapped
    required_fields = ["query", "clicks", "impressions", "ctr", "position"]
    missing = [field for field in required_fields if field not in column_mapping or not column_mapping[field]]
    if missing:
        raise ValueError(f"Missing column mappings in Queries CSV: {missing}")
    
    # Validate that mapped columns exist in dataframe
    missing_cols = [col for field, col in column_mapping.items() if col and col not in df.columns]
    if missing_cols:
        raise ValueError(f"Mapped columns not found in Queries CSV: {missing_cols}")

    q = df.rename(
        columns={
            column_mapping["query"]: "query",
            column_mapping["clicks"]: "clicks",
            column_mapping["impressions"]: "impressions",
            column_mapping["ctr"]: "ctr_raw",
            column_mapping["position"]: "position"
        }
    ).copy()

    q["clicks"] = pd.to_numeric(q["clicks"], errors="coerce").fillna(0).astype(float)
    q["impressions"] = pd.to_numeric(q["impressions"], errors="coerce").fillna(0).astype(float)
    q["ctr"] = q["ctr_raw"].apply(parse_ctr)
    mask_ctr_nan = q["ctr"].isna() & (q["impressions"] > 0)
    q.loc[mask_ctr_nan, "ctr"] = q.loc[mask_ctr_nan, "clicks"] / q.loc[mask_ctr_nan, "impressions"]
    q["position"] = pd.to_numeric(q["position"], errors="coerce")
    
    q["query_norm"] = normalize_query_vectorized(q["query"])

    # Intent
    intents = q["query_norm"].apply(lambda s: classify_intent_row(s, compiled_rules))
    q["intent"] = [x[0] for x in intents]
    q["intent_confidence"] = [x[1] for x in intents]
    
    # Brand Check
    q["is_branded"] = q["query_norm"].apply(lambda s: check_branded(s, brand_regexes))
    q["brand_label"] = q["is_branded"].map({True: "Branded", False: "Non-Branded"})
    
    # Segment for Stratified Clustering
    q["segment"] = q["brand_label"] + " - " + q["intent"]
    
    return q

def cluster_queries_stratified(q: pd.DataFrame) -> pd.DataFrame:
    """Clusters queries within each segment independently."""
    q["cluster_id"] = -1
    
    unique_segments = q["segment"].unique()
    next_start_id = 0
    
    for seg in unique_segments:
        subset = q[q["segment"] == seg].copy()
        if subset.empty:
            continue
            
        indices = subset.index
        n_queries = len(subset)
        
        vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_df=0.95)
        try:
            X = vec.fit_transform(subset["query_norm"].tolist())
        except ValueError:
            q.loc[indices, "cluster_id"] = next_start_id
            next_start_id += 1
            continue

        if n_queries > CLUSTERING_THRESHOLD_N:
            n_clusters = max(5, int(n_queries / 10))
            cl = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42, n_init="auto")
            labels = cl.fit_predict(X)
        elif n_queries < 2:
            labels = np.zeros(n_queries, dtype=int)
        else:
            simdist = 1 - cosine_similarity(X)
            simdist[simdist < 0] = 0
            cl = AgglomerativeClustering(metric="precomputed", linkage="average", 
                                         distance_threshold=0.85, n_clusters=None)
            labels = cl.fit_predict(simdist)
            
        q.loc[indices, "cluster_id"] = labels + next_start_id
        next_start_id += (labels.max() + 1) if len(labels) > 0 else 1
        
    return q

def label_clusters(q: pd.DataFrame) -> pd.DataFrame:
    def tokens(qn):
        return [t for t in qn.split() if t]
    def ngrams(toks, n):
        return [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]

    q["tokens"] = q["query_norm"].apply(tokens)
    topic_map = {}
    
    for cid, sub in q.groupby("cluster_id"):
        cand = Counter()
        toks_list = sub["tokens"].tolist()
        for toks in toks_list:
            for n in (3, 2):
                for g in ngrams(toks, n):
                    cand[g] += 1
        
        best = None
        threshold = max(2, int(0.3 * len(sub)))
        for g, cnt in cand.most_common(50):
            if cnt >= threshold:
                score = cnt * len(g.split())
                if best is None or score > best[0]:
                    best = (score, g)
        
        if best:
            topic_map[cid] = best[1]
        else:
            fallback = sub["query_norm"].iloc[0] if not sub.empty else "misc"
            topic_map[cid] = fallback

    q["topic_label"] = q["cluster_id"].map(topic_map)
    return q

def calculate_opportunity(q: pd.DataFrame) -> pd.DataFrame:
    q["pos_bucket"] = q["position"].apply(pos_bucket)
    bucket_median = q.groupby("pos_bucket")["ctr"].median().to_dict()
    overall_median = q["ctr"].median()
    
    bucket_expectations = {}
    for b in CTR_FLOOR.keys():
        med = bucket_median.get(b, overall_median)
        if pd.isna(med): med = overall_median if not pd.isna(overall_median) else 0.01
        bucket_expectations[b] = max(med, CTR_FLOOR.get(b, 0.01))
    
    q["expected_ctr"] = q["pos_bucket"].map(bucket_expectations).fillna(0.01)
    q["opportunity_clicks"] = (q["impressions"] * np.maximum(0, q["expected_ctr"] - q["ctr"])).fillna(0)
    
    q["priority_score"] = (
        np.log1p(q["clicks"]) * 1.0
        + np.log1p(q["opportunity_clicks"]) * 1.2
        + np.log1p(q["impressions"]) * 0.4
    )
    
    q70, q90 = np.quantile(q["priority_score"], PRIORITY_QUANTILES)
    q["priority_band"] = q["priority_score"].apply(lambda s: "P1" if s >= q90 else ("P2" if s >= q70 else "P3"))
    return q

def aggregate_clusters(q: pd.DataFrame) -> pd.DataFrame:
    # 1. Get top queries per cluster
    top_queries = (
        q.sort_values(["cluster_id", "impressions"], ascending=[True, False])
        .groupby("cluster_id")
        .head(5)
        .groupby("cluster_id")["query"]
        .apply(lambda x: " | ".join(x.astype(str)))
        .reset_index(name="example_queries")
    )

    # 2. Aggregate metrics
    cluster = (
        q.groupby(["cluster_id", "topic_label", "segment", "brand_label", "intent"])
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
    
    # 3. Merge top queries
    cluster = cluster.merge(top_queries, on="cluster_id", how="left")
    
    q70c, q90c = np.quantile(cluster["priority_score"], PRIORITY_QUANTILES)
    cluster["priority_band"] = cluster["priority_score"].apply(
        lambda s: "P1" if s >= q90c else ("P2" if s >= q70c else "P3")
    )
    
    cluster["dominant_intent"] = cluster["intent"] 
    
    return cluster

def preprocess_pages(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    # Validate that all required columns are mapped
    required_fields = ["page", "clicks", "impressions", "ctr", "position"]
    missing = [field for field in required_fields if field not in column_mapping or not column_mapping[field]]
    if missing:
        raise ValueError(f"Missing column mappings in Pages CSV: {missing}")
    
    # Validate that mapped columns exist in dataframe
    missing_cols = [col for field, col in column_mapping.items() if col and col not in df.columns]
    if missing_cols:
        raise ValueError(f"Mapped columns not found in Pages CSV: {missing_cols}")

    p = df.rename(
        columns={
            column_mapping["page"]: "page",
            column_mapping["clicks"]: "page_clicks",
            column_mapping["impressions"]: "page_impressions",
            column_mapping["ctr"]: "page_ctr_raw",
            column_mapping["position"]: "page_position"
        }
    ).copy()

    p["page_clicks"] = pd.to_numeric(p["page_clicks"], errors="coerce").fillna(0).astype(float)
    p["page_impressions"] = pd.to_numeric(p["page_impressions"], errors="coerce").fillna(0).astype(float)
    p["page_ctr"] = p["page_ctr_raw"].apply(parse_ctr)
    mask_ctr_nan = p["page_ctr"].isna() & (p["page_impressions"] > 0)
    p.loc[mask_ctr_nan, "page_ctr"] = p.loc[mask_ctr_nan, "page_clicks"] / p.loc[mask_ctr_nan, "page_impressions"]
    p["page_position"] = pd.to_numeric(p["page_position"], errors="coerce")
    p["slug"] = p["page"].apply(slug_from_url)
    p["page_tokens"] = p["page"].apply(url_to_tokens)
    p["page_text"] = p["page_tokens"].apply(lambda t: " ".join(t))
    return p

def map_clusters_to_pages(q: pd.DataFrame, cluster: pd.DataFrame, p: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rep = (
        q.sort_values(["cluster_id", "impressions"], ascending=[True, False])
        .groupby("cluster_id")
        .head(5)
        .groupby("cluster_id")["query_norm"]
        .apply(lambda s: " ".join(s.tolist()))
        .to_dict()
    )
    
    cluster["cluster_text"] = (
        cluster["topic_label"] + " " + 
        cluster["dominant_intent"] + " " + 
        cluster["cluster_id"].map(rep).fillna("")
    ).str.lower()

    cluster_docs = cluster[["cluster_id", "cluster_text"]].rename(columns={"cluster_text": "text"})
    page_docs = p[["page", "page_text"]].rename(columns={"page_text": "text"})
    all_text = pd.concat([cluster_docs, page_docs], ignore_index=True)
    
    v2 = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
    V = v2.fit_transform(all_text["text"].tolist())
    
    n_clusters = len(cluster)
    Vc = V[:n_clusters]
    Vp = V[n_clusters:]
    sim = cosine_similarity(Vc, Vp)
    
    best_idx = sim.argmax(axis=1)
    cluster["primary_page"] = [p.iloc[j]["page"] for j in best_idx]
    cluster["match_score"] = sim.max(axis=1)
    
    if sim.shape[1] >= 2:
        top2_sorted = np.argsort(-sim, axis=1)[:, :2]
        cluster["runner_up_page"] = [p.iloc[j]["page"] for j in top2_sorted[:, 1]]
        cluster["runner_up_score"] = np.take_along_axis(sim, top2_sorted, axis=1)[:, 1]
    else:
        cluster["runner_up_page"] = None
        cluster["runner_up_score"] = 0.0

    cluster["cannibalisation_risk"] = (
        (cluster["match_score"] - cluster["runner_up_score"] < 0.03) & 
        (cluster["match_score"] > 0.12)
    )

    def action(row):
        if row["cannibalisation_risk"]: return "Fix cannibalisation"
        if row["dominant_intent"] == "transactional" and row["avg_position"] > 10: return "Build/upgrade landing page"
        if row["avg_position"] <= 10: return "CTR/snippet + internal links"
        return "Content refresh + topical authority"

    cluster["recommended_action"] = cluster.apply(action, axis=1)
    
    cluster_to_page = cluster.set_index("cluster_id")["primary_page"].to_dict()
    q["primary_page"] = q["cluster_id"].map(cluster_to_page)
    
    return q, cluster

def calculate_page_opportunities(q: pd.DataFrame, p: pd.DataFrame, cluster: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ALL intents
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
        on="page", how="left"
    )
    page_opportunity_all["ctr"] = page_opportunity_all["actual_clicks"] / page_opportunity_all["impressions"].replace(0, np.nan)

    # PER SEGMENT (Brand/Intent)
    page_opportunity_segment = (
        q.groupby(["primary_page", "segment", "brand_label", "intent"])
        .agg(
            impressions=("impressions", "sum"),
            actual_clicks=("clicks", "sum"),
            opportunity_clicks=("opportunity_clicks", "sum"),
            avg_position=("position", "mean"),
        )
        .reset_index()
        .rename(columns={"primary_page": "page"})
    )
    page_opportunity_segment["ctr"] = page_opportunity_segment["actual_clicks"] / page_opportunity_segment["impressions"].replace(0, np.nan)
    page_opportunity_segment = page_opportunity_segment.merge(
        p[["page", "slug", "page_clicks", "page_impressions", "page_ctr", "page_position"]],
        on="page", how="left"
    )
    
    return page_opportunity_all, page_opportunity_segment

def analyze_cannibalization_risks(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze cannibalization risks by finding clusters that compete for the same pages.
    
    Args:
        cluster_df: DataFrame with cluster data including cannibalisation_risk, primary_page, etc.
        
    Returns:
        DataFrame with cannibalization risk analysis
    """
    # Filter clusters with cannibalization risk
    risk_clusters = cluster_df[cluster_df["cannibalisation_risk"] == True].copy()
    
    if risk_clusters.empty:
        return pd.DataFrame(columns=[
            "primary_cluster_id", "primary_topic", "primary_intent",
            "competing_cluster_id", "competing_topic", "competing_intent",
            "shared_page", "primary_match_score", "runner_up_match_score",
            "score_difference", "risk_level"
        ])
    
    # Group by primary_page to find clusters competing for same page
    risk_rows = []
    
    for page in risk_clusters["primary_page"].unique():
        page_clusters = risk_clusters[risk_clusters["primary_page"] == page].copy()
        
        if len(page_clusters) > 1:
            # Multiple clusters targeting same page - create pairs
            for i, row1 in page_clusters.iterrows():
                for j, row2 in page_clusters.iterrows():
                    if i < j:  # Avoid duplicates
                        score_diff = abs(row1["match_score"] - row2["match_score"])
                        
                        # Determine risk level
                        if score_diff < 0.01:
                            risk_level = "High"
                        elif score_diff < 0.02:
                            risk_level = "Medium"
                        else:
                            risk_level = "Low"
                        
                        risk_rows.append({
                            "primary_cluster_id": row1["cluster_id"],
                            "primary_topic": row1["topic_label"],
                            "primary_intent": row1["intent"],
                            "competing_cluster_id": row2["cluster_id"],
                            "competing_topic": row2["topic_label"],
                            "competing_intent": row2["intent"],
                            "shared_page": page,
                            "primary_match_score": row1["match_score"],
                            "runner_up_match_score": row2["match_score"],
                            "score_difference": score_diff,
                            "risk_level": risk_level
                        })
        
        # Also check runner_up_page scenarios
        for _, row in page_clusters.iterrows():
            if pd.notna(row.get("runner_up_page")) and row["runner_up_page"] != page:
                # Check if runner_up_page is also a primary_page for another cluster
                competing = risk_clusters[risk_clusters["primary_page"] == row["runner_up_page"]]
                if not competing.empty:
                    for _, comp_row in competing.iterrows():
                        score_diff = abs(row["match_score"] - row["runner_up_score"])
                        
                        if score_diff < 0.01:
                            risk_level = "High"
                        elif score_diff < 0.02:
                            risk_level = "Medium"
                        else:
                            risk_level = "Low"
                        
                        risk_rows.append({
                            "primary_cluster_id": row["cluster_id"],
                            "primary_topic": row["topic_label"],
                            "primary_intent": row["intent"],
                            "competing_cluster_id": comp_row["cluster_id"],
                            "competing_topic": comp_row["topic_label"],
                            "competing_intent": comp_row["intent"],
                            "shared_page": row["runner_up_page"],
                            "primary_match_score": row["match_score"],
                            "runner_up_match_score": row["runner_up_score"],
                            "score_difference": score_diff,
                            "risk_level": risk_level
                        })
    
    if not risk_rows:
        return pd.DataFrame(columns=[
            "primary_cluster_id", "primary_topic", "primary_intent",
            "competing_cluster_id", "competing_topic", "competing_intent",
            "shared_page", "primary_match_score", "runner_up_match_score",
            "score_difference", "risk_level"
        ])
    
    risk_df = pd.DataFrame(risk_rows)
    # Sort by score difference (ascending - smallest difference = highest risk)
    risk_df = risk_df.sort_values("score_difference", ascending=True)
    
    return risk_df

# ----------------------------
# GPT Brief & Downloads
# ----------------------------
def build_gpt_brief(
    q_out: pd.DataFrame,
    cluster_out: pd.DataFrame,
    page_opps_all: pd.DataFrame,
    page_opps_segment: pd.DataFrame,
    brand_terms: list,
    client_name: str = "",
    min_impressions: int = 100,
):
    top_pages = page_opps_all.sort_values("opportunity_clicks", ascending=False).head(20)
    top_clusters = cluster_out.sort_values("opportunity_clicks", ascending=False).head(20)
    
    brief = {
        "meta": {
            "client": client_name,
            "generated": datetime.now(timezone.utc).isoformat(),
            "brand_terms": brand_terms
        },
        "top_pages": top_pages[["slug", "opportunity_clicks", "avg_position"]].to_dict(orient="records"),
        "top_clusters": top_clusters[["topic_label", "segment", "opportunity_clicks", "recommended_action", "example_queries"]].to_dict(orient="records")
    }
    
    md = f"# SEO Brief: {client_name}\n\n"
    md += "## Top Opportunities\n"
    md += md_table(["Page", "Opp Clicks", "Pos"], [[r["slug"], f"{r['opportunity_clicks']:.0f}", f"{r['avg_position']:.1f}"] for r in brief["top_pages"]])
    
    return brief, md

def chart_leaderboard_static(df, topn=20):
    d = df.sort_values("opportunity_clicks", ascending=False).head(topn)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(d))
    ax.barh(y, d["actual_clicks"], label="Actual")
    ax.barh(y, d["opportunity_clicks"], left=d["actual_clicks"], label="Opportunity")
    ax.set_yticks(y)
    ax.set_yticklabels(d["slug"])
    ax.invert_yaxis()
    ax.legend()
    ax.set_title(f"Top {topn} Pages")
    plt.tight_layout()
    return fig

# ----------------------------
# Main App
# ----------------------------
def main():
    st.title("GSC Opportunity Mapper (Stratified)")
    st.markdown("""
    **Stratified Logic:** Queries are split into **Branded/Non-Branded**, then by **Intent**, and clustered *within* those segments.
    """)

    # --- Sidebar ---
    st.sidebar.header("Client settings")
    client_name = st.sidebar.text_input("Client name", value="Client")
    brand_terms_input = st.sidebar.text_input("Brand terms (comma-separated)", value="")
    brand_terms = [t.strip() for t in brand_terms_input.split(",") if t.strip()]

    # --- Upload ---
    c1, c2 = st.columns(2)
    q_file = c1.file_uploader("Upload GSC Queries CSV", type=["csv"])
    p_file = c2.file_uploader("Upload GSC Pages CSV", type=["csv"])

    if not (q_file and p_file):
        st.info("Please upload CSVs.")
        st.stop()

    # --- Processing ---
    try:
        queries_df = pd.read_csv(q_file)
        pages_df = pd.read_csv(p_file)
    except Exception as e:
        st.error(f"Error reading CSVs: {e}")
        st.stop()

    # --- Column Mapping ---
    # Initialize session state for column mappings
    if "queries_column_mapping" not in st.session_state:
        st.session_state.queries_column_mapping = {}
    if "pages_column_mapping" not in st.session_state:
        st.session_state.pages_column_mapping = {}
    
    # Auto-detect column mappings
    queries_auto_mapping = detect_column_mapping(queries_df, REQUIRED_QUERY_FIELDS)
    pages_auto_mapping = detect_column_mapping(pages_df, REQUIRED_PAGE_FIELDS)
    
    # Initialize session state with auto-detected mappings if empty
    if not st.session_state.queries_column_mapping:
        st.session_state.queries_column_mapping = queries_auto_mapping.copy()
    if not st.session_state.pages_column_mapping:
        st.session_state.pages_column_mapping = pages_auto_mapping.copy()
    
    # Column Mapping UI
    with st.expander("🔧 Column Mapping - Queries CSV", expanded=False):
        st.markdown("**Detected columns:** " + ", ".join(queries_df.columns.tolist()))
        st.markdown("---")
        
        field_labels = {
            "query": "Query/Page Column",
            "clicks": "Clicks Column",
            "impressions": "Impressions Column",
            "ctr": "CTR Column",
            "position": "Position Column"
        }
        
        for field in ["query", "clicks", "impressions", "ctr", "position"]:
            auto_detected = queries_auto_mapping.get(field)
            current_value = st.session_state.queries_column_mapping.get(field, auto_detected)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected = st.selectbox(
                    f"{field_labels[field]} → `{field}`",
                    options=[None] + list(queries_df.columns),
                    index=0 if current_value is None else (list(queries_df.columns).index(current_value) + 1 if current_value in queries_df.columns else 0),
                    key=f"queries_mapping_{field}"
                )
                st.session_state.queries_column_mapping[field] = selected
            with col2:
                if selected == auto_detected and auto_detected:
                    st.markdown("<br>✅ Auto-detected", unsafe_allow_html=True)
                elif selected:
                    st.markdown("<br>✓ Selected", unsafe_allow_html=True)
    
    with st.expander("🔧 Column Mapping - Pages CSV", expanded=False):
        st.markdown("**Detected columns:** " + ", ".join(pages_df.columns.tolist()))
        st.markdown("---")
        
        field_labels = {
            "page": "Page/URL Column",
            "clicks": "Clicks Column",
            "impressions": "Impressions Column",
            "ctr": "CTR Column",
            "position": "Position Column"
        }
        
        for field in ["page", "clicks", "impressions", "ctr", "position"]:
            auto_detected = pages_auto_mapping.get(field)
            current_value = st.session_state.pages_column_mapping.get(field, auto_detected)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected = st.selectbox(
                    f"{field_labels[field]} → `{field}`",
                    options=[None] + list(pages_df.columns),
                    index=0 if current_value is None else (list(pages_df.columns).index(current_value) + 1 if current_value in pages_df.columns else 0),
                    key=f"pages_mapping_{field}"
                )
                st.session_state.pages_column_mapping[field] = selected
            with col2:
                if selected == auto_detected and auto_detected:
                    st.markdown("<br>✅ Auto-detected", unsafe_allow_html=True)
                elif selected:
                    st.markdown("<br>✓ Selected", unsafe_allow_html=True)
    
    # Validate column mappings
    queries_valid, queries_missing = validate_column_mapping(
        st.session_state.queries_column_mapping,
        ["query", "clicks", "impressions", "ctr", "position"]
    )
    pages_valid, pages_missing = validate_column_mapping(
        st.session_state.pages_column_mapping,
        ["page", "clicks", "impressions", "ctr", "position"]
    )
    
    if not queries_valid or not pages_valid:
        error_msg = "**Please map all required columns before processing:**\n\n"
        if not queries_valid:
            error_msg += f"- **Queries CSV**: Missing mappings for {', '.join(queries_missing)}\n"
        if not pages_valid:
            error_msg += f"- **Pages CSV**: Missing mappings for {', '.join(pages_missing)}\n"
        st.error(error_msg)
        st.stop()

    with st.spinner("Running stratified pipeline..."):
        intent_rules = build_intent_rules(brand_terms)
        compiled_rules = compile_rules(intent_rules)
        brand_regexes = [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in brand_terms]

        # Pipeline
        q_clean = preprocess_queries(queries_df, compiled_rules, brand_regexes, st.session_state.queries_column_mapping)
        p_clean = preprocess_pages(pages_df, st.session_state.pages_column_mapping)
        
        q_clustered = cluster_queries_stratified(q_clean)
        q_labeled = label_clusters(q_clustered)
        
        q_opp = calculate_opportunity(q_labeled)
        cluster_agg = aggregate_clusters(q_opp)
        
        q_final, cluster_final = map_clusters_to_pages(q_opp, cluster_agg, p_clean)
        page_opps_all, page_opps_segment = calculate_page_opportunities(q_final, p_clean, cluster_final)
        
        st.success("Done!")

    # --- Results ---
    tabs = st.tabs(["Visuals", "Tables", "Download"])

    with tabs[0]:
        st.subheader("Top Opportunity Clusters")
        
        # Filters
        f1, f2, f3 = st.columns(3)
        sel_brand = f1.selectbox("Brand Segment", ["All", "Branded", "Non-Branded"])
        sel_intent = f2.selectbox("Intent", ["All", "informational", "commercial", "transactional", "navigational"])
        
        # Filter Logic (Use cluster_final now)
        df_viz = cluster_final.copy()
        
        if sel_brand != "All":
            df_viz = df_viz[df_viz["brand_label"] == sel_brand]
        if sel_intent != "All":
            df_viz = df_viz[df_viz["intent"] == sel_intent]
        
        # Show opportunity clicks per intent
        st.markdown("### Opportunity Clicks by Intent")
        intent_opps = cluster_final.groupby("intent")["opportunity_clicks"].sum().sort_values(ascending=False)
        intent_cols = st.columns(len(intent_opps))
        for idx, (intent_name, opp_clicks) in enumerate(intent_opps.items()):
            with intent_cols[idx]:
                # Highlight if this intent is selected
                is_selected = (sel_intent != "All" and sel_intent == intent_name)
                if is_selected:
                    # Use markdown to create a highlighted container
                    st.markdown(
                        f'<div style="background-color: #e8f4f8; padding: 15px; border-radius: 8px; border: 3px solid #1f77b4; margin-bottom: 10px;">',
                        unsafe_allow_html=True
                    )
                    st.metric(intent_name.title(), f"{opp_clicks:,.0f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.metric(intent_name.title(), f"{opp_clicks:,.0f}")
        
        # Also show filtered total if filters are applied
        if sel_brand != "All" or sel_intent != "All":
            st.metric("Opportunity Clicks (Filtered)", f"{df_viz['opportunity_clicks'].sum():,.0f}")
        
        # Top 15 Clusters for Bubble Chart
        top_15 = df_viz.sort_values("opportunity_clicks", ascending=False).head(15).copy()
        
        # Calculate sizes
        top_15["total_potential"] = top_15["clicks"] + top_15["opportunity_clicks"]
        
        fig = go.Figure()
        
        colors = {"P1": "red", "P2": "orange", "P3": "blue"}
        
        for band in ["P1", "P2", "P3"]:
            subset = top_15[top_15["priority_band"] == band]
            if subset.empty:
                continue
                
            # Outer Bubble (Potential)
            fig.add_trace(go.Scatter(
                x=subset["avg_position"],
                y=subset["opportunity_clicks"],
                mode='markers',
                marker=dict(
                    size=subset["total_potential"],
                    sizemode='area',
                    sizeref=2.0 * max(top_15["total_potential"]) / (80**2),
                    color=colors[band],
                    opacity=0.3,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name=f"{band} (Potential)",
                legendgroup=band,
                text=subset["topic_label"],
                hovertemplate="<b>%{text}</b><br>Potential: %{marker.size:.0f}<br>Opp: %{y:.0f}<extra></extra>"
            ))

            # Inner Bubble (Actual)
            fig.add_trace(go.Scatter(
                x=subset["avg_position"],
                y=subset["opportunity_clicks"],
                mode='markers',
                marker=dict(
                    size=subset["clicks"],
                    sizemode='area',
                    sizeref=2.0 * max(top_15["total_potential"]) / (80**2),
                    color=colors[band],
                    opacity=1.0,
                    line=dict(width=0)
                ),
                name=f"{band} (Actual)",
                legendgroup=band,
                showlegend=False,
                text=subset["topic_label"],
                hovertemplate="<b>%{text}</b><br>Actual: %{marker.size:.0f}<extra></extra>"
            ))

        fig.update_layout(
            title="Top 15 Opportunity Clusters (Inner=Actual, Outer=Potential)",
            xaxis=dict(title="Average Position", autorange="reversed"),
            yaxis=dict(title="Opportunity Clicks"),
            showlegend=True,
            height=600,
            legend=dict(title="Priority Band")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanations
        st.info("""
        **How to read this chart:**
        *   **Outer Bubble (Faded)**: Total Potential Clicks (Actual + Opportunity). The larger the outer bubble, the bigger the total pie.
        *   **Inner Bubble (Solid)**: Actual Clicks. The solid core shows what you are already capturing.
        *   **The Gap**: The visible faded ring represents the *Opportunity*—clicks you are missing out on.
        """)
        
        with st.expander("ℹ️ How are metrics calculated?"):
            st.markdown("""
            **1. Opportunity Clicks**
            We estimate how many *more* clicks you could get if you improved your CTR to the "expected" level for your position.
            $$
            \\text{Opp Clicks} = \\text{Impressions} \\times (\\text{Expected CTR} - \\text{Actual CTR})
            $$
            *Expected CTR is based on the median CTR for that position bucket across your dataset (floored at industry baselines).*

            **2. Priority Score**
            A composite score to rank opportunities, balancing high volume with high potential.
            $$
            \\text{Score} = 1.0 \\times \\ln(\\text{Clicks}) + 1.2 \\times \\ln(\\text{Opp Clicks}) + 0.4 \\times \\ln(\\text{Impressions})
            $$

            **3. Priority Bands**
            *   🔴 **P1 (High)**: Top 10% of opportunities.
            *   🟠 **P2 (Medium)**: Next 20%.
            *   🔵 **P3 (Low)**: Bottom 70%.
            """)

    with tabs[1]:
        st.subheader("Analysis Tables")
        
        # Create sub-tabs for different views
        table_tabs = st.tabs([
            "📊 All Keywords", 
            "⚠️ Cannibalization Risks", 
            "🎯 Action Priority",
            "📋 Raw Data"
        ])
        
        # Tab 1: Keyword-level table with filters
        with table_tabs[0]:
            st.markdown("### Filter and View All Keywords")
            
            # Merge q_final with cluster_final to get topic_label, intent, and brand_label for each query
            # Always get these from cluster_final to ensure consistency
            # Check which columns are available in cluster_final
            available_cols = ["cluster_id"]
            if "topic_label" in cluster_final.columns:
                available_cols.append("topic_label")
            if "intent" in cluster_final.columns:
                available_cols.append("intent")
            if "brand_label" in cluster_final.columns:
                available_cols.append("brand_label")
            
            cluster_cols = cluster_final[available_cols].copy()
            if "intent" in cluster_cols.columns:
                cluster_cols = cluster_cols.rename(columns={"intent": "intent_cluster"})
            if "brand_label" in cluster_cols.columns:
                cluster_cols = cluster_cols.rename(columns={"brand_label": "brand_label_cluster"})
            
            q_with_cluster = q_final.merge(
                cluster_cols,
                on="cluster_id",
                how="left"
            )
            
            # Use cluster values, falling back to query values if cluster values are missing
            if "intent_cluster" in q_with_cluster.columns:
                q_with_cluster["intent"] = q_with_cluster["intent_cluster"].fillna(
                    q_with_cluster.get("intent", "")
                )
            elif "intent" not in q_with_cluster.columns:
                q_with_cluster["intent"] = ""
                
            if "brand_label_cluster" in q_with_cluster.columns:
                q_with_cluster["brand_label"] = q_with_cluster["brand_label_cluster"].fillna(
                    q_with_cluster.get("brand_label", "")
                )
            elif "brand_label" not in q_with_cluster.columns:
                q_with_cluster["brand_label"] = ""
            
            # Ensure topic_label exists
            if "topic_label" not in q_with_cluster.columns:
                # Fallback: create topic_label from cluster_id mapping
                topic_map = cluster_final.set_index("cluster_id")["topic_label"].to_dict()
                q_with_cluster["topic_label"] = q_with_cluster["cluster_id"].map(topic_map)
            
            # Get unique values for filters (handle missing columns gracefully)
            unique_topics = sorted(q_with_cluster["topic_label"].dropna().unique().tolist()) if "topic_label" in q_with_cluster.columns and len(q_with_cluster["topic_label"].dropna()) > 0 else []
            unique_intents = sorted(q_with_cluster["intent"].dropna().unique().tolist()) if "intent" in q_with_cluster.columns and len(q_with_cluster["intent"].dropna()) > 0 else []
            unique_brands = sorted(q_with_cluster["brand_label"].dropna().unique().tolist()) if "brand_label" in q_with_cluster.columns and len(q_with_cluster["brand_label"].dropna()) > 0 else []
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_topics = st.multiselect(
                    "Filter by Topic",
                    options=unique_topics,
                    default=[],
                    key="keyword_filter_topics"
                )
            with col2:
                selected_intents = st.multiselect(
                    "Filter by Intent",
                    options=unique_intents,
                    default=[],
                    key="keyword_filter_intents"
                )
            with col3:
                selected_brands = st.multiselect(
                    "Filter by Brand",
                    options=unique_brands,
                    default=[],
                    key="keyword_filter_brands"
                )
            
            # Apply filters
            filtered_q = q_with_cluster.copy()
            if selected_topics and "topic_label" in filtered_q.columns:
                filtered_q = filtered_q[filtered_q["topic_label"].isin(selected_topics)]
            if selected_intents and "intent" in filtered_q.columns:
                filtered_q = filtered_q[filtered_q["intent"].isin(selected_intents)]
            if selected_brands and "brand_label" in filtered_q.columns:
                filtered_q = filtered_q[filtered_q["brand_label"].isin(selected_brands)]
            
            # Display count
            st.metric("Filtered Keywords", len(filtered_q))
            
            # Select columns to display (only include columns that exist)
            all_display_cols = [
                "query", "topic_label", "intent", "brand_label",
                "clicks", "impressions", "ctr", "position",
                "opportunity_clicks", "priority_score", "priority_band", "primary_page"
            ]
            display_cols = [col for col in all_display_cols if col in filtered_q.columns]
            
            # Sort options (only show columns that exist)
            available_sort_cols = [col for col in ["clicks", "impressions", "opportunity_clicks", "position", "priority_score"] if col in filtered_q.columns]
            if not available_sort_cols:
                available_sort_cols = ["query"]  # Fallback to query if no metrics available
            
            sort_col = st.selectbox(
                "Sort by",
                options=available_sort_cols,
                index=min(2, len(available_sort_cols) - 1) if "opportunity_clicks" in available_sort_cols else 0,
                key="keyword_sort"
            )
            sort_asc = st.checkbox("Ascending", value=False, key="keyword_sort_asc")
            
            if sort_col in filtered_q.columns:
                filtered_q_sorted = filtered_q.sort_values(sort_col, ascending=sort_asc)
            else:
                filtered_q_sorted = filtered_q
            
            # Display table
            st.dataframe(
                filtered_q_sorted[display_cols],
                use_container_width=True,
                height=400
            )
        
        # Tab 2: Cannibalization risks
        with table_tabs[1]:
            st.markdown("### Cannibalization Risk Analysis")
            
            risk_df = analyze_cannibalization_risks(cluster_final)
            
            if risk_df.empty:
                st.info("No cannibalization risks detected.")
            else:
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Risks", len(risk_df))
                with col2:
                    high_risk = len(risk_df[risk_df["risk_level"] == "High"])
                    st.metric("High Risk", high_risk)
                with col3:
                    st.metric("Medium Risk", len(risk_df[risk_df["risk_level"] == "Medium"]))
                
                # Color coding function for risk levels
                def color_risk(val):
                    if val == "High":
                        return "background-color: #ffcccc"
                    elif val == "Medium":
                        return "background-color: #fff4cc"
                    else:
                        return "background-color: #ccffcc"
                
                # Display table with styling
                display_risk_cols = [
                    "primary_topic", "primary_intent",
                    "competing_topic", "competing_intent",
                    "shared_page", "primary_match_score", "runner_up_match_score",
                    "score_difference", "risk_level"
                ]
                
                styled_df = risk_df[display_risk_cols].style.applymap(
                    color_risk,
                    subset=["risk_level"]
                )
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=400
                )
        
        # Tab 3: Action priority sorted by effort
        with table_tabs[2]:
            st.markdown("### Action Priority by Effort Level")
            
            # Add effort_level column
            cluster_with_effort = cluster_final.copy()
            cluster_with_effort["effort_level"] = cluster_with_effort["recommended_action"].map(
                ACTION_EFFORT_ORDER
            ).fillna(99)  # Unknown actions go to end
            
            # Sort by effort_level (ascending - low effort first), then by opportunity_clicks (descending)
            cluster_with_effort = cluster_with_effort.sort_values(
                ["effort_level", "opportunity_clicks"],
                ascending=[True, False]
            )
            
            # Add effort level labels
            effort_labels = {
                1: "Low Effort",
                2: "Medium Effort",
                3: "Medium-High Effort",
                4: "High Effort"
            }
            cluster_with_effort["effort_label"] = cluster_with_effort["effort_level"].map(effort_labels).fillna("Unknown")
            
            # Show counts per effort level (sorted by effort_level, not label)
            effort_summary = cluster_with_effort.groupby("effort_level")["effort_label"].first().to_dict()
            effort_counts = cluster_with_effort["effort_level"].value_counts().sort_index()
            cols = st.columns(len(effort_counts))
            for idx, (effort_level, count) in enumerate(effort_counts.items()):
                if effort_level != 99:  # Skip unknown
                    with cols[idx]:
                        st.metric(effort_summary.get(effort_level, f"Level {effort_level}"), int(count))
            
            # Group by effort level with expandable sections
            for effort_level in sorted(cluster_with_effort["effort_level"].unique()):
                if effort_level == 99:
                    continue
                effort_label = effort_labels.get(effort_level, f"Level {effort_level}")
                effort_clusters = cluster_with_effort[cluster_with_effort["effort_level"] == effort_level]
                
                # Convert numpy types to Python bool for Streamlit
                is_expanded = bool(effort_level == 1)
                with st.expander(f"{effort_label} ({len(effort_clusters)} clusters)", expanded=is_expanded):
                    display_cols = [
                        "effort_label", "recommended_action", "topic_label", "intent",
                        "opportunity_clicks", "priority_band", "primary_page", "avg_position"
                    ]
                    st.dataframe(
                        effort_clusters[display_cols],
                        use_container_width=True
                    )
        
        # Tab 4: Original raw data tables
        with table_tabs[3]:
            st.subheader("Page Opportunities by Segment")
            st.dataframe(page_opps_segment, use_container_width=True)
            
            st.subheader("Clusters (with Top Queries)")
            st.dataframe(cluster_final, use_container_width=True)

    with tabs[2]:
        st.subheader("Download Pack")
        
        brief_json, brief_md = build_gpt_brief(q_final, cluster_final, page_opps_all, page_opps_segment, brand_terms, client_name)
        
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("gpt_brief.md", brief_md)
            z.writestr("gpt_brief.json", json.dumps(brief_json, indent=2))
            z.writestr("clusters.csv", cluster_final.to_csv(index=False))
            z.writestr("pages_segment.csv", page_opps_segment.to_csv(index=False))
            
            # Static chart
            img_buf = io.BytesIO()
            fig = chart_leaderboard_static(page_opps_all)
            fig.savefig(img_buf, format="png")
            z.writestr("top_pages_chart.png", img_buf.getvalue())
            
        st.download_button(
            "Download ZIP Report",
            data=zip_buf.getvalue(),
            file_name="gsc_report.zip",
            mime="application/zip"
        )

if __name__ == "__main__":
    main()
