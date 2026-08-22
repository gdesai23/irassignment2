import hashlib
import html
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import networkx as nx
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


APP_TITLE = "Group 44 HealthIR Workbench"
DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)

THEME_CSS = """
<style>
    :root {
        --ink: #17212b;
        --muted: #5b6674;
        --line: #dce3e9;
        --paper: #f6f8fb;
        --panel: #ffffff;
        --teal: #0f766e;
        --blue: #2563eb;
        --coral: #c2410c;
        --amber: #b45309;
    }
    .stApp {
        background:
            linear-gradient(180deg, #edf7f5 0%, #f6f8fb 340px, #ffffff 100%);
        color: var(--ink);
    }
    [data-testid="stSidebar"] {
        background: #102331;
        color: #f8fafc;
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #dbe7ef;
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 16px 14px;
        box-shadow: 0 10px 30px rgba(15, 35, 49, 0.07);
    }
    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }
    div[data-testid="stMetricValue"] {
        color: var(--ink);
    }
    .hero {
        border: 1px solid rgba(15, 118, 110, 0.18);
        border-radius: 8px;
        padding: 26px 28px;
        margin-bottom: 18px;
        background:
            linear-gradient(135deg, rgba(15,118,110,0.14), rgba(37,99,235,0.08)),
            #ffffff;
        box-shadow: 0 18px 45px rgba(15, 35, 49, 0.10);
    }
    .hero-title {
        font-size: 2.25rem;
        line-height: 1.05;
        font-weight: 800;
        margin: 0 0 8px;
        color: var(--ink);
    }
    .hero-copy {
        max-width: 920px;
        color: #3a4654;
        font-size: 1rem;
        margin: 0;
    }
    .hero-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 18px;
    }
    .pill {
        border: 1px solid rgba(23, 33, 43, 0.12);
        background: rgba(255, 255, 255, 0.74);
        color: #22313f;
        border-radius: 999px;
        padding: 7px 11px;
        font-size: 0.82rem;
        font-weight: 650;
    }
    .section-note {
        border-left: 4px solid var(--teal);
        background: #f8fbfb;
        padding: 11px 14px;
        border-radius: 6px;
        color: #3a4654;
        margin: 4px 0 16px;
    }
    .result-card {
        border: 1px solid var(--line);
        border-left: 5px solid var(--teal);
        background: #ffffff;
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: 0 10px 24px rgba(15, 35, 49, 0.07);
    }
    .result-title {
        color: var(--ink);
        font-size: 1.02rem;
        font-weight: 780;
        margin-bottom: 8px;
    }
    .result-snippet {
        color: #334155;
        margin-bottom: 10px;
    }
    .result-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
    }
    .tag {
        border-radius: 999px;
        background: #eef6f5;
        color: #115e59;
        border: 1px solid #cde7e3;
        padding: 4px 8px;
        font-size: 0.76rem;
        font-weight: 650;
    }
    .tag.blue {
        background: #eef4ff;
        color: #1d4ed8;
        border-color: #cfddff;
    }
    .tag.amber {
        background: #fff7ed;
        color: #9a3412;
        border-color: #fed7aa;
    }
    div[data-testid="stTabs"] button {
        border-radius: 8px 8px 0 0;
        font-weight: 650;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }
</style>
"""

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "has",
    "have", "in", "into", "is", "it", "of", "on", "or", "such", "that", "the",
    "their", "this", "to", "uses", "with", "when", "while", "without", "over",
    "above", "below", "both", "each", "many", "must", "should", "through",
}

SYNONYMS = {
    "heart attack": ["myocardial infarction", "cardiac event"],
    "doctor": ["clinician", "physician"],
    "ranking": ["bm25", "pagerank", "hits", "authority"],
    "recommend": ["recommendation", "similar", "hybrid"],
    "crawl": ["crawling", "crawler", "seed"],
    "privacy": ["sensitive", "identifiable", "access control"],
    "evaluation": ["precision", "recall", "map", "ndcg"],
}


@dataclass
class SearchResult:
    doc_id: str
    score: float
    text_score: float
    authority_score: float


def normalize_url(url: str) -> str:
    url, _ = urldefrag(url.strip())
    parsed = urlparse(url)
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"[^A-Za-z0-9\s-]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def stem_token(token: str) -> str:
    for suffix in ("ization", "ational", "fulness", "iveness", "ation", "ing", "ers", "ies", "ed", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: str, remove_stopwords: bool = True, use_stemming: bool = True) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z-]{1,}", clean_text(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if use_stemming:
        tokens = [stem_token(t) for t in tokens]
    return tokens


def feature_tokens(text: str, remove_stopwords: bool = True, use_stemming: bool = True, ngram_max: int = 1) -> list[str]:
    tokens = tokenize(text, remove_stopwords, use_stemming)
    if ngram_max >= 2:
        tokens = tokens + [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return tokens


def expand_query(query: str) -> str:
    expanded = [query]
    q = query.lower()
    for key, values in SYNONYMS.items():
        if key in q or any(v in q for v in values):
            expanded.extend(values)
    return " ".join(expanded)


@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    docs = pd.read_csv(DATA_DIR / "documents.csv")
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    return meta.merge(docs, on="doc_id", how="inner")


def load_crawled_data() -> pd.DataFrame:
    path = STORAGE_DIR / "crawled_documents.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["doc_id", "title", "source", "url", "category", "date", "author", "content"])


def active_corpus(use_crawled: bool) -> pd.DataFrame:
    base = load_demo_data()
    if use_crawled:
        crawled = load_crawled_data()
        if not crawled.empty:
            base = pd.concat([base, crawled], ignore_index=True)
    base["content_hash"] = base["content"].map(lambda x: hashlib.sha256(clean_text(x).encode()).hexdigest())
    base = base.drop_duplicates("url").drop_duplicates("content_hash").reset_index(drop=True)
    return base


def save_crawled(rows: list[dict]) -> None:
    if not rows:
        return
    existing = load_crawled_data()
    df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    df["url"] = df["url"].map(normalize_url)
    df["content_hash"] = df["content"].map(lambda x: hashlib.sha256(clean_text(x).encode()).hexdigest())
    df = df.drop_duplicates("url").drop_duplicates("content_hash").drop(columns=["content_hash"])
    df.to_csv(STORAGE_DIR / "crawled_documents.csv", index=False)


def crawl(seeds: list[str], max_depth: int, max_pages: int, same_domain: bool) -> tuple[pd.DataFrame, dict]:
    seen_urls, seen_hashes, rows = set(), set(), []
    queue = [(normalize_url(seed), 0, urlparse(seed).netloc.lower()) for seed in seeds if seed.strip()]
    started = time.perf_counter()
    failures = []
    headers = {"User-Agent": "Group44-IR-Assignment-Crawler/1.0"}

    while queue and len(rows) < max_pages:
        url, depth, root_domain = queue.pop(0)
        if url in seen_urls or depth > max_depth:
            continue
        seen_urls.add(url)
        try:
            response = requests.get(url, timeout=5, headers=headers)
            if "text/html" not in response.headers.get("content-type", ""):
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            title = soup.title.string.strip() if soup.title and soup.title.string else urlparse(url).netloc
            text = clean_text(soup.get_text(" "))
            if len(text.split()) < 60:
                continue
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                rows.append(
                    {
                        "doc_id": f"C{len(rows) + 1:03d}",
                        "title": title[:120],
                        "source": urlparse(url).netloc,
                        "url": url,
                        "category": "Crawled",
                        "date": pd.Timestamp.today().date().isoformat(),
                        "author": "Web crawl",
                        "content": text[:12000],
                    }
                )
            if depth < max_depth:
                for link in soup.find_all("a", href=True)[:80]:
                    child = normalize_url(urljoin(url, link["href"]))
                    child_domain = urlparse(child).netloc.lower()
                    if child.startswith("http") and (not same_domain or child_domain == root_domain):
                        queue.append((child, depth + 1, root_domain))
        except Exception as exc:
            failures.append(f"{url}: {exc}")

    metrics = {
        "visited_urls": len(seen_urls),
        "unique_documents": len(rows),
        "duplicate_urls_skipped": max(0, len(seen_urls) - len(rows)),
        "failures": len(failures),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    return pd.DataFrame(rows), metrics


def build_index(df: pd.DataFrame, remove_stopwords: bool, stemming: bool, ngram_max: int) -> dict:
    started = time.perf_counter()
    analyzer = lambda text: feature_tokens(text, remove_stopwords, stemming, ngram_max)
    tfidf = TfidfVectorizer(analyzer=analyzer)
    tfidf_matrix = tfidf.fit_transform(df["content"])
    count_vectorizer = CountVectorizer(analyzer=analyzer)
    counts = count_vectorizer.fit_transform(df["content"])
    tokens_by_doc = [analyzer(text) for text in df["content"]]
    avgdl = float(np.mean([len(t) for t in tokens_by_doc])) if len(df) else 0.0
    vocabulary = count_vectorizer.get_feature_names_out()

    sim = cosine_similarity(tfidf_matrix)
    graph = nx.Graph()
    for doc_id in df["doc_id"]:
        graph.add_node(doc_id)
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if sim[i, j] >= 0.15:
                graph.add_edge(df.loc[i, "doc_id"], df.loc[j, "doc_id"], weight=float(sim[i, j]))
    pr = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {d: 1 / len(df) for d in df["doc_id"]}

    classifier = None
    if df["category"].nunique() > 1:
        classifier = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer=analyzer)),
                ("clf", MultinomialNB()),
            ]
        )
        classifier.fit(df["content"], df["category"])

    return {
        "tfidf": tfidf,
        "tfidf_matrix": tfidf_matrix,
        "counts": counts,
        "count_vectorizer": count_vectorizer,
        "tokens_by_doc": tokens_by_doc,
        "avgdl": avgdl,
        "vocabulary": vocabulary,
        "pagerank": pr,
        "similarity": sim,
        "graph": graph,
        "classifier": classifier,
        "build_seconds": round(time.perf_counter() - started, 3),
    }


def bm25_scores(query: str, index: dict, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    terms = tokenize(query)
    docs_tokens = index["tokens_by_doc"]
    n_docs = len(docs_tokens)
    df_counts = Counter()
    for tokens in docs_tokens:
        df_counts.update(set(tokens))
    scores = np.zeros(n_docs)
    avgdl = index["avgdl"] or 1.0
    for i, tokens in enumerate(docs_tokens):
        tf = Counter(tokens)
        dl = len(tokens) or 1
        for term in terms:
            if tf[term] == 0:
                continue
            idf = math.log(1 + (n_docs - df_counts[term] + 0.5) / (df_counts[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / avgdl)
            scores[i] += idf * (tf[term] * (k1 + 1) / denom)
    return scores


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or values.max() == values.min():
        return np.zeros_like(values)
    return (values - values.min()) / (values.max() - values.min())


def search(df: pd.DataFrame, index: dict, query: str, method: str, authority_weight: float, expand: bool) -> tuple[list[SearchResult], float]:
    started = time.perf_counter()
    q = expand_query(query) if expand else query
    if method == "TF-IDF":
        q_vec = index["tfidf"].transform([q])
        text_scores = cosine_similarity(q_vec, index["tfidf_matrix"]).ravel()
    elif method == "BM25":
        text_scores = bm25_scores(q, index)
    else:
        tfidf_scores = cosine_similarity(index["tfidf"].transform([q]), index["tfidf_matrix"]).ravel()
        text_scores = 0.55 * minmax(tfidf_scores) + 0.45 * minmax(bm25_scores(q, index))

    authority = np.array([index["pagerank"].get(doc_id, 0.0) for doc_id in df["doc_id"]])
    final_scores = (1 - authority_weight) * minmax(text_scores) + authority_weight * minmax(authority)
    results = [
        SearchResult(row.doc_id, float(final_scores[i]), float(text_scores[i]), float(authority[i]))
        for i, row in df.iterrows()
        if final_scores[i] > 0 or text_scores[i] > 0
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return results, round((time.perf_counter() - started) * 1000, 2)


def highlight_snippet(text: str, query: str, words: int = 38) -> str:
    query_terms = set(tokenize(query, use_stemming=False))
    tokens = str(text).split()
    if not tokens:
        return ""
    hit = next((i for i, t in enumerate(tokens) if clean_text(t) in query_terms), 0)
    start = max(0, hit - 8)
    snippet = " ".join(tokens[start : start + words])
    return snippet + ("..." if start + words < len(tokens) else "")


def corpus_profile(df: pd.DataFrame, index: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in df.iterrows():
        tokens = tokenize(row["content"])
        rows.append(
            {
                "doc_id": row["doc_id"],
                "title": row["title"],
                "category": row["category"],
                "tokens": len(tokens),
                "unique_terms": len(set(tokens)),
                "top_terms": ", ".join([t for t, _ in Counter(tokens).most_common(5)]),
            }
        )
    all_terms = Counter(term for tokens in index["tokens_by_doc"] for term in tokens)
    term_df = pd.DataFrame(all_terms.most_common(25), columns=["term", "frequency"])
    return pd.DataFrame(rows), term_df


def compare_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    configs = [
        ("raw tokens", False, False),
        ("stopword removal", True, False),
        ("stopword + stemming", True, True),
    ]
    rows = []
    for name, stops, stem in configs:
        tokens = [tokenize(text, stops, stem) for text in df["content"]]
        vocab = sorted({t for doc in tokens for t in doc})
        rows.append(
            {
                "strategy": name,
                "vocabulary_size": len(vocab),
                "avg_doc_length": round(np.mean([len(t) for t in tokens]), 2),
                "feature_density": round(np.mean([len(set(t)) for t in tokens]) / max(1, len(vocab)), 4),
            }
        )
    return pd.DataFrame(rows)


def recommendations(df: pd.DataFrame, index: dict, doc_id: str, mode: str, k: int) -> pd.DataFrame:
    idx = int(df.index[df["doc_id"] == doc_id][0])
    content_scores = index["similarity"][idx]
    popularity = np.array([index["pagerank"].get(d, 0) for d in df["doc_id"]])
    same_category = (df["category"].values == df.loc[idx, "category"]).astype(float)
    collaborative = 0.65 * minmax(popularity) + 0.35 * same_category
    if mode == "Content-based":
        scores = content_scores
    elif mode == "Collaborative-style":
        scores = collaborative
    else:
        scores = 0.7 * minmax(content_scores) + 0.3 * minmax(collaborative)
    order = np.argsort(scores)[::-1]
    rows = []
    for pos in order:
        if pos == idx:
            continue
        rows.append(
            {
                "doc_id": df.loc[pos, "doc_id"],
                "title": df.loc[pos, "title"],
                "category": df.loc[pos, "category"],
                "similarity_score": round(float(scores[pos]), 4),
            }
        )
        if len(rows) >= k:
            break
    return pd.DataFrame(rows)


def load_judgements() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "relevance_judgments.csv")


def average_precision(ranked: list[str], relevant: set[str], k: int) -> float:
    hits, total = 0, 0.0
    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / i
    return total / max(1, len(relevant))


def ndcg(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum((1 / math.log2(i + 2)) for i, doc_id in enumerate(ranked[:k]) if doc_id in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def evaluate(df: pd.DataFrame, index: dict, k: int, authority_weight: float) -> pd.DataFrame:
    rows = []
    for _, row in load_judgements().iterrows():
        relevant = set(str(row["relevant_doc_ids"]).split())
        for method in ["TF-IDF", "BM25", "Hybrid"]:
            ranked = [r.doc_id for r in search(df, index, row["query"], method, authority_weight, True)[0]]
            retrieved = set(ranked[:k])
            tp = len(retrieved & relevant)
            precision = tp / max(1, len(retrieved))
            recall = tp / max(1, len(relevant))
            f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
            rr = next((1 / (i + 1) for i, doc_id in enumerate(ranked) if doc_id in relevant), 0.0)
            rows.append(
                {
                    "query": row["query"],
                    "method": method,
                    "Precision": round(precision, 3),
                    "Recall": round(recall, 3),
                    "F1": round(f1, 3),
                    f"Precision@{k}": round(tp / k, 3),
                    f"Recall@{k}": round(recall, 3),
                    "AP": round(average_precision(ranked, relevant, k), 3),
                    "MRR": round(rr, 3),
                    "NDCG": round(ndcg(ranked, relevant, k), 3),
                }
            )
    return pd.DataFrame(rows)


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_hero(df: pd.DataFrame, index: dict) -> None:
    crawled_count = len(load_crawled_data())
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">Group 44 HealthIR Workbench</div>
            <p class="hero-copy">
                A complete Streamlit information retrieval command center for crawling, mining,
                indexing, ranked search, recommendations, evaluation, and performance analytics.
            </p>
            <div class="hero-strip">
                <span class="pill">{len(df)} active documents</span>
                <span class="pill">{df["category"].nunique()} knowledge areas</span>
                <span class="pill">{len(index["vocabulary"])} indexed features</span>
                <span class="pill">{index["graph"].number_of_edges()} similarity links</span>
                <span class="pill">{crawled_count} stored crawl items</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_note(text: str) -> None:
    st.markdown(f'<div class="section-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def render_rubric_map() -> None:
    st.markdown(
        """
        <div class="hero-strip">
            <span class="pill">Dashboard</span>
            <span class="pill">Search interface</span>
            <span class="pill">Crawling interface</span>
            <span class="pill">Index management</span>
            <span class="pill">Ranking visualization</span>
            <span class="pill">Recommendation panel</span>
            <span class="pill">Evaluation dashboard</span>
            <span class="pill">Performance analytics</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(df: pd.DataFrame, index: dict) -> None:
    duplicate_url_count = len(df) - df["url"].nunique()
    duplicate_content_count = len(df) - df["content_hash"].nunique() if "content_hash" in df else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(df))
    c2.metric("Categories", df["category"].nunique())
    c3.metric("Vocabulary", len(index["vocabulary"]))
    c4.metric("Duplicate risk", duplicate_url_count + duplicate_content_count)


def render_dashboard(df: pd.DataFrame, index: dict) -> None:
    render_metric_cards(df, index)
    st.subheader("Corpus Overview")
    section_note("The demo corpus is bundled for reliable Virtual Lab execution, while the crawler can add live public web documents.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.bar_chart(df["category"].value_counts())
    with c2:
        profile, terms = corpus_profile(df, index)
        st.bar_chart(terms.set_index("term"))
    st.dataframe(df[["doc_id", "title", "source", "category", "date", "url"]], width="stretch", hide_index=True)


def render_crawler() -> None:
    st.subheader("Configurable Web Crawling")
    section_note("Multiple seed URLs, depth control, URL normalization, and content hashing are used to manage acquisition and deduplication.")
    seeds_text = st.text_area(
        "Seed URLs",
        value="https://www.who.int/news-room\nhttps://www.cdc.gov",
        height=90,
    )
    c1, c2, c3 = st.columns(3)
    depth = c1.slider("Crawling depth", 0, 2, 1)
    max_pages = c2.slider("Maximum pages", 1, 20, 5)
    same_domain = c3.toggle("Stay on seed domain", value=True)
    if st.button("Run crawler", type="primary"):
        seeds = [line.strip() for line in seeds_text.splitlines() if line.strip()]
        with st.spinner("Crawling pages and deduplicating URLs/documents..."):
            rows, metrics = crawl(seeds, depth, max_pages, same_domain)
            save_crawled(rows.to_dict("records"))
        st.success(f"Crawled {metrics['unique_documents']} unique documents from {metrics['visited_urls']} visited URLs.")
        st.json(metrics)
        if not rows.empty:
            st.dataframe(rows[["doc_id", "title", "source", "url"]], width="stretch", hide_index=True)
    crawled = load_crawled_data()
    st.caption("Stored crawled documents")
    st.dataframe(crawled[["doc_id", "title", "source", "url", "date"]] if not crawled.empty else crawled, width="stretch", hide_index=True)


def render_text_mining(df: pd.DataFrame, index: dict) -> None:
    st.subheader("Text Preprocessing and Mining")
    section_note("This screen turns raw text into structured features and exposes keyword extraction, document profiling, classification, and strategy comparison.")
    profile, terms = corpus_profile(df, index)
    st.dataframe(profile, width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Top corpus terms")
        st.bar_chart(terms.set_index("term"))
    with c2:
        st.caption("Preprocessing strategy comparison")
        st.dataframe(compare_preprocessing(df), width="stretch", hide_index=True)
    if index["classifier"] is not None:
        text = st.text_area("Classify a new document", "BM25 ranking and query expansion improve search quality for clinical retrieval.")
        pred = index["classifier"].predict([text])[0]
        st.info(f"Predicted category: {pred}")


def render_index_management(df: pd.DataFrame, index: dict) -> None:
    st.subheader("Index Management")
    section_note("Metadata is stored in data/metadata.csv; document contents are stored separately in data/documents.csv.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Index build time", f"{index['build_seconds']} s")
    c2.metric("Average document length", round(index["avgdl"], 2))
    c3.metric("Similarity graph edges", index["graph"].number_of_edges())
    st.dataframe(pd.DataFrame({"term": index["vocabulary"]}).head(200), width="stretch", hide_index=True)
    if st.button("Export active index metadata"):
        export = df[["doc_id", "title", "source", "url", "category", "date", "author"]]
        export.to_csv(STORAGE_DIR / "active_metadata_export.csv", index=False)
        st.success("Exported storage/active_metadata_export.csv")


def render_search(df: pd.DataFrame, index: dict) -> None:
    st.subheader("Search Interface")
    section_note("Run TF-IDF, BM25, or Hybrid search with optional query expansion and PageRank authority blending.")
    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
    query = c1.text_input("Query", "clinical search ranking")
    method = c2.selectbox("Retrieval model", ["Hybrid", "BM25", "TF-IDF"])
    top_k = c3.slider("Top K", 3, 15, 8)
    expand = c4.toggle("Query expansion", value=True)
    authority_weight = st.slider("PageRank authority weight", 0.0, 0.5, 0.15, 0.05)
    results, latency = search(df, index, query, method, authority_weight, expand)
    st.caption(f"{len(results)} matching documents in {latency} ms")
    for rank, result in enumerate(results[:top_k], start=1):
        row = df[df["doc_id"] == result.doc_id].iloc[0]
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">{rank}. {html.escape(str(row["title"]))}</div>
                <div class="result-snippet">{html.escape(highlight_snippet(row["content"], query))}</div>
                <div class="result-meta">
                    <span class="tag">{html.escape(result.doc_id)}</span>
                    <span class="tag blue">{html.escape(str(row["category"]))}</span>
                    <span class="tag amber">final {result.score:.4f}</span>
                    <span class="tag">text {result.text_score:.4f}</span>
                    <span class="tag">PageRank {result.authority_score:.4f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_ranking(df: pd.DataFrame, index: dict) -> None:
    st.subheader("Ranking Visualization")
    section_note("PageRank is calculated on a document similarity graph to show how authority changes final ordering.")
    pr_df = pd.DataFrame(
        [{"doc_id": k, "PageRank": v, "title": df.loc[df["doc_id"] == k, "title"].iloc[0]} for k, v in index["pagerank"].items()]
    ).sort_values("PageRank", ascending=False)
    st.bar_chart(pr_df.set_index("doc_id")["PageRank"])
    st.dataframe(pr_df, width="stretch", hide_index=True)
    st.caption("Authority is calculated over a similarity graph. Documents linked to many related documents receive higher PageRank.")


def render_recommendations(df: pd.DataFrame, index: dict) -> None:
    st.subheader("Recommendation Panel")
    section_note("Top-K recommendations show similarity scores for content-based, collaborative-style, and hybrid modes.")
    c1, c2, c3 = st.columns([2, 1, 1])
    selected = c1.selectbox("Current document", df["doc_id"] + " - " + df["title"])
    doc_id = selected.split(" - ")[0]
    mode = c2.selectbox("Recommendation type", ["Hybrid", "Content-based", "Collaborative-style"])
    k = c3.slider("Top K", 3, 10, 5, key="rec_k")
    st.dataframe(recommendations(df, index, doc_id, mode, k), width="stretch", hide_index=True)


def render_evaluation(df: pd.DataFrame, index: dict) -> None:
    st.subheader("Evaluation Dashboard")
    section_note("Relevance judgements drive Precision, Recall, F1, Precision@K, Recall@K, AP/MAP, MRR, and NDCG comparisons.")
    k = st.slider("Evaluation K", 3, 10, 5)
    authority_weight = st.slider("Evaluation authority weight", 0.0, 0.5, 0.15, 0.05, key="eval_authority")
    eval_df = evaluate(df, index, k, authority_weight)
    st.dataframe(eval_df, width="stretch", hide_index=True)
    summary = eval_df.groupby("method").mean(numeric_only=True).round(3).reset_index()
    st.caption("Mean metric comparison")
    st.dataframe(summary, width="stretch", hide_index=True)
    st.bar_chart(summary.set_index("method")[["Precision", "Recall", "F1", "MRR", "NDCG"]])


def render_performance(df: pd.DataFrame, index: dict) -> None:
    st.subheader("Performance Analytics")
    section_note("Latency and index statistics make the system operationally inspectable, not just algorithmically correct.")
    queries = ["clinical search ranking", "duplicate documents", "recommend similar medical documents", "web crawling hospital advisories"]
    rows = []
    for query in queries:
        for method in ["TF-IDF", "BM25", "Hybrid"]:
            _, latency = search(df, index, query, method, 0.15, True)
            rows.append({"query": query, "method": method, "latency_ms": latency})
    perf = pd.DataFrame(rows)
    st.dataframe(perf, width="stretch", hide_index=True)
    st.bar_chart(perf.groupby("method")["latency_ms"].mean())
    st.write(
        "Operational inference: this corpus builds quickly, but the same dashboard exposes latency and index growth trends "
        "that would decide when to recrawl, shard the index, or cache frequent queries."
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_theme()

    with st.sidebar:
        st.header("Run Configuration")
        use_crawled = st.toggle("Include crawled documents", value=True)
        remove_stopwords = st.toggle("Remove stopwords", value=True)
        stemming = st.toggle("Apply stemming", value=True)
        ngram_max = st.select_slider("N-gram range", options=[1, 2], value=1)
        st.markdown("---")
        st.write("Rubric coverage")
        st.caption("Workflow, crawling, mining, search, PageRank, recommendations, evaluation, analytics.")

    df = active_corpus(use_crawled)
    index = build_index(df, remove_stopwords, stemming, ngram_max)
    render_hero(df, index)
    render_rubric_map()

    tabs = st.tabs(
        [
            "Dashboard",
            "Crawling",
            "Text Mining",
            "Index",
            "Search",
            "Ranking",
            "Recommendations",
            "Evaluation",
            "Analytics",
        ]
    )
    with tabs[0]:
        render_dashboard(df, index)
    with tabs[1]:
        render_crawler()
    with tabs[2]:
        render_text_mining(df, index)
    with tabs[3]:
        render_index_management(df, index)
    with tabs[4]:
        render_search(df, index)
    with tabs[5]:
        render_ranking(df, index)
    with tabs[6]:
        render_recommendations(df, index)
    with tabs[7]:
        render_evaluation(df, index)
    with tabs[8]:
        render_performance(df, index)


if __name__ == "__main__":
    main()
