# Information Retrieval Assignment 2 Report

**Course:** Information Retrieval  
**Group:** 44  
**Application:** HealthIR Workbench - end-to-end healthcare document retrieval system  
**Implementation:** Streamlit application in `app.py`

## 1. Objective and Use Case

The implemented system is a professional Streamlit-based Information Retrieval workflow for healthcare knowledge management. The chosen use case is retrieval over hospital and public-health knowledge documents, including clinical triage, telemedicine, monitoring, privacy, ranking, recommendation, crawling, and evaluation material.

The application is designed so that the full workflow is executable through the Streamlit front end. It does not depend on notebooks or static backend-only output.

## 2. Dataset and Acquisition

The system uses a bundled dataset of 24 healthcare IR documents so the assignment can run reliably in the Virtual Lab. It also includes a configurable web crawler for acquiring additional documents from heterogeneous public web sources.

Dataset files:

- `data/documents.csv`: stores document contents.
- `data/metadata.csv`: stores title, source, URL, category, date, and author separately.
- `data/relevance_judgments.csv`: stores test queries and relevant document IDs for evaluation.

Crawling support:

- Multiple seed URLs.
- Configurable crawling depth.
- Maximum page limit.
- Same-domain toggle.
- Duplicate URL handling by URL normalization.
- Duplicate document handling by content hashing.
- Runtime storage in `storage/crawled_documents.csv`.

## 3. Streamlit Workflow Coverage

The application contains the following tabs:

- **Dashboard:** corpus statistics, category distribution, top terms, and document metadata.
- **Crawling:** seed configuration, depth control, duplicate handling, and crawl metrics.
- **Text Mining:** document profiling, keyword extraction, preprocessing comparison, and classification.
- **Index:** vocabulary view, index build time, average document length, and metadata export.
- **Search:** query processing, query expansion, TF-IDF/BM25/Hybrid retrieval, ranked results, snippets, and metadata.
- **Ranking:** PageRank authority visualization over a document similarity graph.
- **Recommendations:** content-based, collaborative-style, and hybrid Top-K recommendations.
- **Evaluation:** Precision, Recall, F1, Precision@K, Recall@K, AP/MAP, MRR, and NDCG.
- **Analytics:** query latency and performance comparison.

## 4. Text Preprocessing and Mining

The preprocessing pipeline includes:

- HTML removal and text normalization.
- Tokenization.
- Stop word removal.
- Lightweight stemming.
- Configurable unigram/bigram feature extraction.

Mining functions include:

- Corpus vocabulary analysis.
- Top keyword extraction using term frequency.
- Document profiling with token counts, unique terms, and top terms.
- Comparative preprocessing analysis for raw tokens, stop word removal, and stemming.
- Naive Bayes document classification using TF-IDF features.

Inference: stop word removal and stemming reduce vocabulary size and noise, making the feature space more compact. This improves ranking robustness for short clinical queries, but aggressive stemming should be monitored because healthcare terms can carry precise meaning.

## 5. Search and Ranking

The search module supports:

- TF-IDF cosine retrieval.
- BM25 retrieval.
- Hybrid retrieval combining normalized TF-IDF and BM25 scores.
- Query expansion using domain synonyms.
- Snippets with result metadata.
- PageRank authority weighting.

PageRank is calculated over a document similarity graph. Documents become connected when their TF-IDF similarity crosses a threshold. This approximates document authority even when the corpus has no explicit hyperlinks.

Inference: BM25 is strong for concise keyword queries because it controls document length bias. TF-IDF is useful for semantic similarity and recommendation. Hybrid ranking gives a balanced retrieval model by combining term matching and vector similarity, while PageRank adds an authority signal.

## 6. Recommender System

The recommendation panel supports:

- **Content-based recommendation:** cosine similarity between TF-IDF document vectors.
- **Collaborative-style recommendation:** category and authority/popularity signals.
- **Hybrid recommendation:** combines content similarity with collaborative-style signals.

Top-K recommendations are displayed with similarity scores.

Inference: content-based recommendation is reliable for a small corpus because it only needs document text. Collaborative recommendation becomes more powerful when real user interaction logs exist. The hybrid method is most practical here because it combines document similarity with corpus-level authority.

## 7. Evaluation Metrics

The evaluation dashboard uses relevance judgements from `data/relevance_judgments.csv`. Metrics implemented:

- Precision.
- Recall.
- F1-score.
- Precision@K.
- Recall@K.
- AP, used as per-query Average Precision.
- MAP, obtained by averaging AP across queries.
- MRR.
- NDCG.

Verified sample result at K=5 with authority weight 0.15:

| Method | Precision | Recall | F1 | Precision@5 | Recall@5 | MAP/AP mean | MRR | NDCG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.475 | 0.710 | 0.565 | 0.475 | 0.710 | 0.575 | 0.875 | 0.698 |
| Hybrid | 0.475 | 0.710 | 0.565 | 0.475 | 0.710 | 0.559 | 0.854 | 0.686 |
| TF-IDF | 0.450 | 0.669 | 0.533 | 0.450 | 0.669 | 0.534 | 0.854 | 0.663 |

Inference: BM25 performs best on the bundled relevance set because the queries are short and term-focused. Hybrid is close and may become stronger when authority signals and crawled documents are added. TF-IDF remains useful for similarity and recommendation.

## 8. Compulsory Inference and Discussion

### 8.1 Highly Relevant Documents Retrieved but Ranked Poorly

Possible causes:

- Query terms are matched, but important field information such as title or category is not boosted.
- Synonyms are missing, for example "heart attack" versus "myocardial infarction".
- Document length normalization is weak, allowing long documents to dominate.
- PageRank or authority weight is too high and suppresses topical relevance.
- Preprocessing removes meaningful clinical terms.

Improvements:

- Add title/category field boosting.
- Tune BM25 parameters and authority weight.
- Improve query expansion using domain thesauri.
- Use pseudo-relevance feedback.
- Add learning-to-rank when click or judgement data is available.

### 8.2 Effect of Duplicate or Near-Duplicate Documents

Duplicates affect the system in several ways:

- **Indexing:** vocabulary and posting lists become inflated.
- **Ranking:** repeated documents occupy multiple top positions and reduce diversity.
- **Recommendation:** similar duplicates are recommended instead of genuinely useful related documents.
- **Evaluation:** metrics become misleading because retrieving duplicates may look like multiple successful hits.

Mitigation:

- URL canonicalization.
- Exact content hashing.
- Shingling or cosine similarity for near-duplicate detection.
- Cluster duplicates and keep one canonical version.
- Apply result diversification during ranking.

### 8.3 Content-Based vs Collaborative Recommendation

Content-based recommendation is preferable when:

- Rich document text and metadata are available.
- User interaction history is limited.
- New documents are frequently added.
- Explanations are required.

Collaborative recommendation is preferable when:

- There are many users and reliable interaction logs.
- User behavior reveals hidden relationships between documents.
- Discovery beyond text similarity is important.

In this assignment, content-based and hybrid recommendation are most suitable because the corpus has strong document text but no real user logs.

### 8.4 Integration of IR Components

The complete IR lifecycle improves effectiveness because each module strengthens the next:

- Crawling acquires fresh and heterogeneous content.
- Metadata management supports filtering, display, and provenance.
- Preprocessing cleans text and creates meaningful features.
- Indexing makes retrieval efficient.
- Ranking orders documents by relevance and authority.
- Recommendation helps users continue exploration after search.
- Evaluation quantifies effectiveness.
- Analytics reveals performance bottlenecks.

The integrated Streamlit interface makes these steps observable and executable in one workflow.

### 8.5 Learnings

Key learnings:

- Good retrieval depends on both matching and ranking.
- Duplicate handling is essential before indexing and evaluation.
- BM25 remains a strong baseline for keyword search.
- TF-IDF vectors are useful for similarity, classification, and recommendations.
- PageRank can be adapted to non-web corpora using a similarity graph.
- IR evaluation must include ranking-sensitive metrics such as MRR and NDCG, not only precision and recall.

## 9. Experimental Evidence to Capture in Virtual Lab

Recommended screenshots:

- Dashboard tab showing corpus statistics.
- Crawling tab after a crawl run or showing crawl configuration.
- Text Mining tab showing keyword and preprocessing comparison.
- Search tab showing ranked results for `clinical search ranking`.
- Ranking tab showing PageRank visualization.
- Recommendations tab showing Top-K recommendations.
- Evaluation tab showing metric table and comparison chart.
- Analytics tab showing latency comparison.

Local screenshots have also been captured for all required tabs. They are available in:

- `LOCAL_SCREENSHOT_REPORT_Group44.pdf`
- `LOCAL_SCREENSHOT_REPORT_Group44.html`
- `evidence_screenshots/`

## 10. Conclusion

The submitted system satisfies the required Assignment 2 components: Streamlit workflow, heterogeneous acquisition through crawling and dataset ingestion, duplicate handling, separate metadata/content storage, preprocessing and text mining, ranked web search, PageRank visualization, recommender system, IR evaluation metrics, performance analytics, and compulsory inference discussion.
