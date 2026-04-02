# Zero-Shot Stock Sentiment Analysis with Explainable AI

A zero-shot sentiment analysis pipeline for predicting short-term (1 to 31-day) stock market movements from financial news, with built-in multi-layer explainability. The system uses RoBERTa-Large-MNLI for sentiment classification and DeBERTa-Large-MNLI for noise reduction and impact-horizon estimation, operating entirely without financial fine-tuning.

## Features

- **News retrieval** from the Finnhub API with chunked weekly fetching and market-date alignment
- **Sentence-level noise reduction** using zero-shot NLI relevance classification
- **Impact horizon classification** across nine literature-grounded event families
- **Gaussian horizon weighting** based on expected time-to-impact alignment
- **EWMA recency weighting** with horizon-adaptive decay
- **Five sentiment models**: RoBERTa zero-shot (primary), FinBERT, FinGPT, Llama 3.1 8B, Mistral 7B
- **Decision thresholds** with sigmoid scoring for neutral abstention
- **Five-layer XAI framework**: LIME token attribution, article-level contribution analysis, pipeline-level diagnostics, narrative clustering, and LLM-generated summaries
- **Evidence quality diagnostics** with six reliability flags
- **Interactive Streamlit dashboard** with nine analysis tabs

## Project Structure

```
src/
    config.py                  # Central configuration
    main.py                    # CLI entry point
    news_retriever/            # Finnhub API, ticker resolution, recency weighting
    preprocessing/             # Deduplication, filtering, noise reduction
    weighting/                 # Event classification, Gaussian horizon weighting
    predictors/                # Sentiment models (RoBERTa, FinBERT, FinGPT, Ollama)
    xai/                       # Explainability (LIME, article analysis, narratives)
    ui/                        # Streamlit web application
    testing/
        evaluation/            # Test runner, FPB benchmark, threshold tuning, XAI evaluation
        smoke/                 # Quick backtesting sanity checks
data/
    temp/                      # Transient pipeline outputs (articles.json, predictions.json)
    articles/                  # Cached preprocessed article sets
    evaluation/                # Test sets, evaluation results, threshold calibration, FPB
    financial_phrasebank_datasets/  # FPB CSV files (50agree, 75agree, allagree)
    xai_explanations/          # XAI output files and charts
```

## Requirements

- Python 3.11+
- CUDA-compatible GPU (recommended for inference speed)
- Ollama (for narrative synthesis and LLM-based model evaluation)
- Finnhub API key (free tier)

## Installation

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure the Finnhub API key in `src/config.py`:

```python
FINNHUB_API_KEY = "your_api_key_here"
```

3. Install Ollama and pull the required models:

```bash
ollama pull llama3.2:3b      # narrative synthesis
ollama pull llama3.1:8b      # baseline evaluation (optional)
ollama pull mistral:7b       # baseline evaluation (optional)
```

## Usage

### Terminal (CLI)

```bash
python src/main.py
```

The system will prompt for a company name (or ticker), start date, and end date.

### Web Interface (Streamlit)

```bash
streamlit run src/ui/app.py
```

### Evaluation

```bash
# Run holdout evaluation with the configured model
python -m testing.evaluation.test_runner --mode evaluate --test-set ../data/evaluation/pipeline_evaluation_dataset/holdout_set.json

# FPB benchmark
python -m testing.evaluation.phrasebank_benchmark --model <model_name>

# FPB label tuning (112 configs, K-fold CV)
python -m testing.evaluation.phrasebank_benchmark --mode tune --model <model_name>

# Threshold tuning on tune set scores
python -m testing.evaluation.tune_thresholds

# XAI evaluation (quality flags + flip-set analysis)
python -m testing.evaluation.xai_evaluation

# Narrative consistency experiment
python -m testing.evaluation.narrative_consistency --cases 30 --runs 5

# Generate report figures
python -m testing.evaluation.report_figures
```

## Configuration

Key settings in `src/config.py`:

| Setting | Description | Default |
|---------|-------------|---------|
| `SENTIMENT_MODEL` | Active sentiment model | `"zero-shot"` |
| `MODEL_NAME` | NLI model for sentiment | `"roberta-large-mnli"` |
| `NOISE_REDUCTION_MODEL` | NLI model for noise filtering | `"microsoft/deberta-large-mnli"` |
| `IMPACT_HORIZON_MODEL` | NLI model for event classification | `"microsoft/deberta-large-mnli"` |
| `DECISION_THRESHOLD_ENABLED` | Enable sigmoid threshold abstention | `True` |
| `DECISION_THRESHOLD_POS` | Positive class threshold | `0.56` |
| `DECISION_THRESHOLD_NEG` | Negative class threshold | `0.26` |
| `XAI_ENABLED` | Enable XAI report generation | `True` |
| `XAI_LIME_TOP_N` | Articles to explain with LIME | `5` |
| `COVERAGE_COUNT_BOOST` | Boost multi-source articles | `True` |
| `HEADLINE_ONLY_WEIGHT` | Discount for headline-only articles | `0.5` |

## Evaluation Dataset

The evaluation dataset comprises 480 test cases generated from 20 US-listed companies across five sectors (Technology, Finance, Healthcare, Energy, Consumer), evaluated over six prediction windows (1, 3, 5, 7, 14, and 31 days). Ground-truth labels are derived from volatility-scaled stock returns via the yfinance API.

The dataset is split into:
- **Tune set**: 288 cases (12 companies) for threshold calibration and model selection
- **Holdout set**: 192 cases (8 companies) for final evaluation

The split is performed at the ticker level to prevent information leakage.

## Disclaimer

This system is an academic research project and is not intended as financial advice. Predictions are generated automatically and should not be used as the sole basis for investment decisions. The author accepts no responsibility for any financial losses incurred from using this system.

## Acknowledgements

This project was developed as a final-year undergraduate dissertation at the University of Nottingham, School of Computer Science. I would like to thank my supervisor, Shreyank Narayana Gowda, for his guidance throughout the project.

## License

This project was developed for academic purposes as part of BSc Computer Science with Artificial Intelligence at the University of Nottingham. All pre-trained models used (RoBERTa, DeBERTa, LLaMA) are subject to their respective open-source licences.
