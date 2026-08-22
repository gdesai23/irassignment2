# Assignment 2 Verification Checklist - Group 44

Verified against `IR_Assignment2_2025-26_S2_Group44.docx`.

## Streamlit End-to-End Workflow

Status: complete.

- Dashboard: `Dashboard` tab.
- Search interface: `Search` tab.
- Crawling interface: `Crawling` tab.
- Index management: `Index` tab.
- Ranking visualization: `Ranking` tab.
- Recommendation panel: `Recommendations` tab.
- Evaluation dashboard: `Evaluation` tab.
- Performance analytics: `Analytics` tab.
- Full workflow is executable from Streamlit front end.

## Acquisition and Storage Requirements

Status: complete.

- Heterogeneous acquisition: bundled public-style dataset plus optional live web crawling.
- Multiple seed sources: multiline seed URL input in crawler.
- Configurable crawling depth: depth slider in crawler.
- Duplicate URL handling: URL normalization and visited URL tracking.
- Duplicate document handling: SHA-256 content hashing.
- Metadata separate from contents: `data/metadata.csv` and `data/documents.csv`.

## Text Preprocessing and Mining

Status: complete.

- Text normalization, tokenization, stop word removal, stemming.
- Unigram/bigram feature option.
- Keyword extraction and top term visualization.
- Document profiling.
- Document classification with TF-IDF and Naive Bayes.
- Comparative preprocessing strategy table.
- Corpus and feature distribution visualizations.

## Web Searching and Ranking

Status: complete.

- Query processing with optional expansion.
- TF-IDF, BM25, and Hybrid retrieval.
- Ranked results with metadata and snippets.
- PageRank implemented over similarity graph.
- Ranking visualization included.

## Recommender System

Status: complete.

- Content-based recommendation.
- Collaborative-style recommendation.
- Hybrid recommendation.
- Top-K recommendations with similarity scores.

## Evaluation Metrics

Status: complete.

- Precision.
- Recall.
- F1-score.
- Precision@K.
- Recall@K.
- AP/MAP.
- MRR.
- NDCG.
- Comparative metric tables and visualizations.

## Inference and Discussion

Status: complete.

Covered in `REPORT_Group44.md`:

- Causes and improvements for poor ranking.
- Duplicate and near-duplicate effects and mitigation.
- Content-based vs collaborative recommendation comparison.
- Integration of crawling, mining, indexing, search, ranking, and recommendation.
- Learnings based on results.

## Local Validation

Commands executed successfully:

```bash
python -m py_compile app.py
```

```bash
python -c "import app; df=app.active_corpus(False); idx=app.build_index(df, True, True, 2); print(len(df), len(idx['vocabulary'])); print(app.evaluate(df, idx, 5, 0.15).groupby('method').mean(numeric_only=True).round(3))"
```

Streamlit endpoint check:

```text
http://localhost:8501 -> HTTP 200
```

## Remaining Manual Evidence

The assignment asks for Virtual Lab screenshots or a short screen recording. These should be captured after running:

```bash
streamlit run app.py
```

inside the BITS Virtual Lab portal.
