# AML Investigation Accelerator - Data Directory

## Available Datasets (Parquet Optimized)

### Performance Optimized Storage
All datasets are available in both CSV and **Parquet** formats:
- **Parquet**: 60-80% smaller file size, 5-10x faster loading
- **CSV**: Human-readable, compatible with all tools

### Ready-to-Use Datasets

**SAML-D Style**: `data/raw/saml_d/`
- `saml_transactions.csv` (original format)
- `saml_transactions.parquet` (optimized)
- 5,000 transactions, 28 ML typologies

**IBM AML Style**: `data/raw/ibm_aml/`
- `ibm_transactions.csv` 
- `ibm_transactions.parquet` (recommended)
- 500,000 transactions, realistic patterns

**Maryam Style**: `data/raw/maryam/`
- `money_laundering_data.csv`
- `money_laundering_data.parquet` (recommended)
- 1,000 transactions, classic ML patterns

**Combined Dataset**: `data/synthetic/`
- `combined_sample.csv` (9MB)
- `combined_sample.parquet` (2MB) - **Use this for development**
- 9,000 total transactions

## Quick Start

### Load Data in Python (Recommended)
```python
import pandas as pd

# Fast loading with Parquet
df = pd.read_parquet('data/synthetic/combined_sample.parquet')
print(f"Loaded {len(df):,} transactions in seconds!")

# Load specific dataset
saml_df = pd.read_parquet('data/raw/saml_d/saml_transactions.parquet')
```

### Load Data in Python (CSV fallback)
```python
# Slower but compatible
df = pd.read_csv('data/synthetic/combined_sample.csv')
```

## File Size Comparison

| Dataset | CSV Size | Parquet Size | Space Saved |
|---------|----------|--------------|-------------|
| Combined | ~9 MB | ~2 MB | 78% |
| SAML-D | ~4 MB | ~1 MB | 75% |
| IBM AML | ~3 MB | ~0.8 MB | 73% |
| Maryam | ~0.8 MB | ~0.2 MB | 75% |

## Smart Caching

The download script automatically:
- Skips re-downloading existing datasets
- Converts CSV to Parquet for performance
- Validates data integrity
- Creates cache manifest

## Usage Recommendations

1. **Use Parquet files for development** (much faster)
2. **Start with combined_sample.parquet** (has everything)
3. **CSV files available for compatibility**
4. **Run download script safely** (won't re-download)

## Dataset Schema

```
transaction_id      - Unique identifier
timestamp          - Transaction datetime  
amount             - Transaction amount (optimized float32)
currency           - Currency code (category)
[... other fields with optimized data types]
```

Ready for blazing-fast agent development!
