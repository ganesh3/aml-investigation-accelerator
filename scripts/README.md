# 📊 Enhanced AML Data Pipeline - README

This project delivers a **modular, intelligent AML (Anti-Money Laundering) data pipeline**. It supports raw data ingestion, schema unification, feature enrichment, quality assessment, and agent-specific dataset generation.

---

## 🚀 Pipeline Overview

### 1. `download_datasets.py`

Handles downloading and caching of public AML datasets (e.g., Kaggle), CSV to Parquet conversion, datatype optimization, and synthetic dataset generation.

### 2. `preprocess_data.py`

Cleans the data, unifies schemas, engineers features, scores data quality, and outputs customized datasets for different investigative AI agents.

---

## 🔧 Feature Engineering Summary

| Feature Type      | Features                                                                |
| ----------------- | ----------------------------------------------------------------------- |
| **Temporal**      | `hour`, `day_of_week`, `is_weekend`, `is_business_hours`                |
| **Amount-Based**  | `log_amount`, `amount_rounded`, `is_round_amount`, `high_amount`        |
| **Risk Patterns** | `cross_border`, `unusual_hour`                                          |
| **Account Stats** | `orig_tx_count`, `orig_total_amount`, `orig_avg_amount`, `orig_ml_rate` |

---

## ✅ Data Quality Scoring

Quality is assessed based on:

* **Completeness**: % of non-null cells
* **Consistency**: schema sanity (risk scores, timestamps, unique IDs, ML balance)
* **Business Logic**: laundering patterns align with risk scores, alert flags, typologies

---

## 🤖 Agent-Specific Datasets

Each dataset targets a specific task and includes only the most relevant features.

| Agent                    | Columns Used                                                                                                                                                                                                |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Alert Triage**         | `transaction_id`, `timestamp`, `amount`, `currency`, `transaction_type`, `risk_score`, `typology`, `alert_flag`, `cross_border`, `unusual_hour`, `high_amount`, `originator_country`, `beneficiary_country` |
| **Evidence Collection**  | `transaction_id`, `originator_account`, `beneficiary_account`, `originator_bank`, `beneficiary_bank`, `amount`, `timestamp`, `orig_tx_count`, `orig_total_amount`, `orig_avg_amount`, `orig_ml_rate`        |
| **Pattern Analysis**     | All core and derived features: amount, time, originator stats, typologies, flags                                                                                                                            |
| **Narrative Generation** | `transaction_id`, `timestamp`, `amount`, `currency`, `typology`, `originator_account`, `beneficiary_account`, `originator_country`, `beneficiary_country`, `risk_score`, `alert_flag`, `is_laundering`      |

---

## 📂 Outputs

All cleaned and processed outputs are saved in:

* `/data/processed/agent_datasets/`
* `/data/processed/samples/`
* `/data/processed/data_statistics.json`
* `/data/aml_local.db`

---

## 🧪 Development Samples

* `dev_sample` — 500 rows
* `balanced_sample` — Class-balanced
* `high_risk_sample` — Risk score > 0.7
* `recent_sample` — Most recent 1,000 transactions

---

## 💾 Storage Format

* **Parquet**: optimized for speed and size
* **CSV**: provided for compatibility
* **SQLite**: embedded querying and dashboards

---

## 📌 Requirements

* Python 3.7+
* `pandas`, `numpy`, `pyarrow`, `kaggle`, `sqlite3`

---

## 📣 License

This AML pipeline is released for research and educational use only.

---

For diagrams, workflows, and implementation logic, see `/docs/` folder or the [flowchart below](#).

---
