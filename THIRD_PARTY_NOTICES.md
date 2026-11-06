# Third-party data and software notice

> **Status:** license inventory for the classroom release. This notice is not legal advice.

## SMS Spam Collection dataset

The notebook downloads the SMS Spam Collection at runtime; no copy of that dataset is committed to this repository.

The one committed data file, `src/data/hard_eval_50.csv` (50 hard test messages), was written by hand for this lab. It is original and not taken from the SMS Spam Collection; it ships under the same terms as the rest of this repository.

- Primary source: Kaggle mirror `uciml/sms-spam-collection-dataset` — <https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset>
- Fallback source: UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), as listed on the UCI page.
- Citation: Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). *Contributions to the Study of SMS Spam
  Filtering: New Collection and Results.* Proceedings of the 2011 ACM Symposium on Document Engineering.

## Models and libraries

| Component | Use in the lab | License |
| --- | --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | sentence embeddings (Part 3), downloaded at runtime | Apache-2.0 |
| `sentence-transformers` | loading the embedding model | Apache-2.0 |
| PyTorch | MLP training (Part 2) | BSD-3-Clause |
| NumPy, pandas, Matplotlib | numerics and plots | BSD-3-Clause / PSF-compatible |
| `kagglehub` | dataset download | Apache-2.0 |
| `openai` Python SDK | Chat Completions calls (Part 4) | Apache-2.0 |

Calls to the OpenAI API are subject to OpenAI's terms of use and are billed to the learner's own API key.
