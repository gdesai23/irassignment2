# Group 44 Information Retrieval Assignment 2

## Project

**Group 44 HealthIR Workbench** is an end-to-end Streamlit Information Retrieval system for a healthcare knowledge corpus. It covers crawling, duplicate handling, metadata separation, preprocessing, text mining, indexing, ranked search, PageRank authority ranking, recommendations, evaluation metrics, and performance analytics.

## Files

- `app.py` - Streamlit application.
- `data/documents.csv` - document contents only.
- `data/metadata.csv` - document metadata stored separately from contents.
- `data/relevance_judgments.csv` - relevance labels for evaluation.
- `requirements.txt` - dependencies.
- `REPORT_Group44.md` - implementation explanation, results, and inferences.
- `LOCAL_SCREENSHOT_REPORT_Group44.pdf` - local execution evidence with screenshots of all required tabs.
- `LOCAL_SCREENSHOT_REPORT_Group44.html` - browser-viewable screenshot report.
- `evidence_screenshots/` - individual screenshots for each Streamlit tab.
- `storage/` - generated at runtime for crawled documents and exports.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Suggested Virtual Lab Demo Flow

1. Open the **Dashboard** tab and show corpus distribution and metadata table.
2. Open **Crawling**, set seed URLs, depth, and maximum pages, then run the crawler.
3. Open **Text Mining** and show keyword extraction, document profiling, preprocessing comparison, and classification.
4. Open **Index** and show index build time, vocabulary, and metadata/content separation.
5. Open **Search**, run queries such as `clinical search ranking` and `duplicate documents in corpus`.
6. Open **Ranking** and show PageRank scores over the similarity graph.
7. Open **Recommendations** and display Top-K recommendations with similarity scores.
8. Open **Evaluation** and show Precision, Recall, F1, Precision@K, Recall@K, MAP/AP, MRR, and NDCG.
9. Open **Analytics** and show query latency and performance comparison.

## Notes

The bundled corpus makes the application executable even if the lab environment has no internet access. Live web crawling is optional and stores crawled documents in `storage/crawled_documents.csv`.

Local proof of execution has been captured in `LOCAL_SCREENSHOT_REPORT_Group44.pdf` and `evidence_screenshots/`.
