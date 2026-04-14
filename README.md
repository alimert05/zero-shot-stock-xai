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
    articles/                  # Cached preprocessed article sets (DeBERTa and RoBERTa pipelines)
    evaluation/                # Test sets, per-model results, tune-set configurations, FPB benchmarks
    financial_phrasebank_datasets/  # FPB CSV files (50agree, 75agree, allagree)
    temp/                      # Transient pipeline outputs (created at runtime)
    xai_explanations/          # XAI output files and charts (created at runtime)
logs/                          # Runtime logs (fetch.logs)
requirements.txt               # Python dependencies
```

## Requirements

- Python 3.11+
- CUDA-compatible GPU with at least 6 GB VRAM recommended (8 GB for comfortable operation). The pipeline will fall back to CPU automatically if no GPU is available, but inference will be significantly slower.
- Ollama (for narrative synthesis and LLM-based model evaluation)
- Finnhub API key (free tier, obtain from [finnhub.io](https://finnhub.io/))

## Installation

1. Obtain the source code:

**Option A — From GitHub:**

```bash
git clone <repository-url>
cd zero-shot-stock-xai
```

**Option B — From the submitted ZIP file:**

Extract the ZIP archive and navigate to the extracted folder:

```bash
cd zero-shot-stock-xai
```

2. (Recommended) Create a virtual environment to isolate dependencies:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the required NLTK data:

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

5. Configure the Finnhub API key in `src/config.py`:

```python
FINNHUB_API_KEY = "your_api_key_here"
```

6. Install Ollama from [ollama.com](https://ollama.com/). Open a separate terminal and keep it running throughout your session:

```bash
ollama serve
```

Then in your main terminal, pull the required models:

```bash
ollama pull llama3.2:3b       # required for narrative synthesis (falls back to a deterministic template if unavailable)
ollama pull llama3.1:8b       # optional, only needed for baseline model evaluation
ollama pull mistral:7b        # optional, only needed for baseline model evaluation
```

**Note:** The first run will automatically download the HuggingFace models (RoBERTa-Large-MNLI, DeBERTa-Large-MNLI), which requires approximately 1.5 GB of disk space.

## Usage

### Web Interface (Streamlit)

From the project root:

```bash
streamlit run src/ui/app.py
```

The dashboard will open in your browser. Enter a company name (e.g., "Apple" or "AAPL"), a start date, and an end date, then click Analyse. Results are presented across nine tabs covering the prediction, evidence quality, storylines, event types, article analysis, robustness, weighting, LIME tokens, and interactive charts. A single prediction typically takes 1-3 minutes depending on the number of articles retrieved and the available hardware.

### Terminal (CLI)

```bash
python src/main.py
```

The system will prompt for a company name (or ticker), start date, and end date.

### Evaluation

All evaluation commands should be run from the `src/` directory:

```bash
cd src

# Run holdout evaluation with the configured model
python -m testing.evaluation.test_runner --mode evaluate --test-set ../data/evaluation/pipeline_evaluation_dataset/holdout_set.json

# FPB benchmark (model options: deberta, roberta, finbert, fingpt, ollama-llama3, ollama-mistral)
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

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA not available` warning | The pipeline will use CPU automatically. For GPU inference, ensure CUDA-compatible drivers and PyTorch with CUDA support are installed. |
| `ConnectionError` from Ollama | Ensure the Ollama server is running (`ollama serve`) before starting the application. |
| Finnhub API rate limit errors | The free tier allows 60 calls per minute. The pipeline includes a built-in rate limiter (55/min), but if multiple runs overlap, wait briefly before retrying. |
| `nltk.download` errors | Run `python -c "import nltk; nltk.download('punkt_tab')"` manually. |
| HuggingFace model download fails | Ensure a stable internet connection. Models are cached after the first download in `~/.cache/huggingface/`. |

## Disclaimer

This system is an academic research project and is not intended as financial advice. Predictions are generated automatically and should not be used as the sole basis for investment decisions. The author accepts no responsibility for any financial losses incurred from using this system.

## Acknowledgements

This project was developed as a final-year undergraduate dissertation at the University of Nottingham, School of Computer Science. I would like to thank my supervisor, Shreyank Narayana Gowda, for his guidance throughout the project.

## License

This project was developed for academic purposes as part of BSc Computer Science with Artificial Intelligence at the University of Nottingham. All pre-trained models used (RoBERTa, DeBERTa, LLaMA) are subject to their respective open-source licences.
