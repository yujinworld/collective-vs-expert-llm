# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project titled "1 LLM vs. 100 persona given sLLMs" that compares the decision-making capabilities of a single high-performance LLM against 100 smaller LLMs for stock investment decisions based on news events.

## Development Commands

### Environment Setup
```bash
# Install dependencies using Poetry
poetry install

# Run the main experiment framework (recommended)
python baseline.py

# Run the v2 framework with parallel event processing
python baseline_v2.py

# Execute the complete data collection pipeline
python yujin/00_run_pipeline.py

# Execute scoring and evaluation on prediction results
python yujin/08_score_predictions.py

# Run statistical tests on prediction results
python yujin/09_statistical_tests.py
```

### Environment Variables
Create a `.env` file with:
```
OPENROUTER_API_KEY=your_api_key_here
```

## Core Architecture

### Experiment Framework Architecture

The project follows a modular agent-based architecture for comparing expert vs crowd intelligence:

1. **Agent Abstraction Layer** (`baseline.py`):
   - `BaseModelAgent` abstract class defines the prediction interface
   - `OpenAIAgent` implements OpenRouter API calls with error handling
   - `AgentPrediction` TypedDict enforces multi-period response format with 5 prediction periods: day_1, day_3, day_7, day_15, day_30
   - Each period includes: `{"decision": "buy"|"sell", "confidence": int, "reason": str}`

2. **Prediction Orchestration** (`StockPredictor` class):
   - Manages expert agent (single high-performance model)
   - Coordinates crowd agents (100 smaller models with persona profiles)
   - Aggregates crowd decisions using confidence-weighted voting per prediction period
   - Supports parallel execution using ThreadPoolExecutor

3. **Experimental Templates**:
   - **baseline.py**: Full-featured framework with modular agent design, persona support, and sequential event processing with checkpoint/resume capability
   - **baseline_v2.py**: Enhanced version with parallel event processing using ThreadPoolExecutor (max_workers=10) for faster execution
   - Both versions support checkpoint recovery via JSON files for resuming interrupted experiments

### Data Processing Pipeline

Sequential data processing architecture in `yujin/` directory:

**Naming Convention:**
- **00-09**: Pipeline scripts in execution order
  - 00: Orchestration scripts
  - 01-06: Data collection pipeline
  - 07: Data formatting utilities
  - 08-09: Evaluation and analysis
- **_archived/**: Deprecated or replaced scripts

**Core Pipeline (yujin/ directory):**
1. **01_generate_event_data.py**: Initial S&P 100 event data generation from raw data
2. **02_crawl_news_headlines.py**: News headline collection using GNews API (OOP implementation)
3. **03_attach_market_data.py**: Market capitalization and sector data enrichment using yfinance
4. **04_resolve_article_urls.py**: URL resolution via Selenium to handle redirects
5. **05_extract_article_body.py**: Article body extraction using newspaper3k
6. **06_retry_failed_articles.py**: Retry logic for failed article fetches (newspaper + Selenium)
7. **07_format_agent_data.py**: Data formatting utilities (`convert_df_to_agent_format()`) - integrated into baseline.py
8. **08_score_predictions.py**: Scoring and evaluation with trading day adjustments and actual price matching
9. **09_statistical_tests.py**: Statistical significance testing (T-test, Mann-Whitney U-test, classification metrics)
10. **00_run_pipeline.py**: End-to-end pipeline orchestration (steps 1-6)

**Archived Scripts (_archived/ directory):**
- **crawl_news_basic.py**: Basic news crawler (replaced by 02_crawl_news_headlines.py)
- **resolve_urls_v1.py**: Original URL resolution script (replaced by 04_resolve_article_urls.py)
- **call_openrouter.py**: Direct OpenRouter API example (deprecated - use baseline.py)
- **check_crawling.py**: Unfinished validation script
- **result_refinement.py**: Legacy result processing

### OpenRouter API Integration

All LLM interactions use OpenRouter as a proxy for accessing multiple models:
- **Expert Model**: `gpt-5` for high-performance reasoning (uses "medium" effort level)
- **Crowd Models**: `gpt-5-nano` for cost-effective bulk predictions (uses "medium" effort level)
- **Response Format**: JSON objects with `decision` (up/down), `confidence` (0-100), and optional `reason` fields for each prediction period
- **Error Handling**: Returns None on failure, which is excluded from aggregation. Failures are logged to CSV with model_name, error_type, and error_message
- **Persona Integration**: Crowd agents receive persona profiles from ESS11 survey data (nationality, age, education, political views, economic outlook, news consumption habits) to simulate diverse investor perspectives
- **Parallel Execution**: Crowd agents execute in parallel with ThreadPoolExecutor (max_workers=10 in baseline.py, unlimited in baseline_v2.py)

## Data Structure and Flow

### Input Data Format
```python
agent_data = {
    'symbol': 'AAPL',
    'search_date': '2024-12-11',
    'titles': 'News title 1 / News title 2 / ...',
    'descriptions': 'Description 1 / Description 2 / ...',
    'article_body': 'Full article text 1 / Full article text 2 / ...',
    'sector': 'Technology'
}
```

### File Organization
- **Raw Data**: `data/articles/json/` and `data/articles/csv/`
- **Processed Data**: `data/event_data.csv`, `data/stock_data.csv`
- **Market Data**: Market cap enriched datasets with timestamps (e.g., `news_with_market_cap_YYYYMMDD_HHMMSS.csv`)
- **Prediction Results**:
  - `prediction_results_YYYYMMDD_HHMMSS.csv` (main output with expert, crowd_N, and crowd_average rows)
  - `prediction_failures_YYYYMMDD_HHMMSS.csv` (failure logs with error details)
  - `checkpoint_YYYYMMDD_HHMMSS.json` (progress tracking for resume capability)
- **Scoring Results**: `data/answer/scoring_results_detailed_*.csv`, `scoring_results_crowd_*.csv`, `scoring_comparison_*.csv`, `scoring_summary_*.csv`
- **Statistical Tests**: `data/answer/classification_metrics_*.csv`, `confidence_performance_*.csv`, `stats_test_results_*.csv`
- **Persona Data**: `data/ESS11/persona.json` (contains ESS11 survey-based demographic and behavioral profiles)
- **Configuration**: `.env` for API keys, `pyproject.toml` for dependencies

### Experiment Flow
1. Load and filter event data by symbol and date from CSV
2. Format data for LLM consumption using `convert_df_to_agent_format()` (integrated in baseline.py)
3. Load persona profiles and randomly select N personas for crowd agents
4. Execute expert prediction (single model, multi-period forecasts)
5. Execute crowd predictions (N models in parallel with persona profiles, multi-period forecasts)
6. Aggregate crowd results using confidence-weighted voting per prediction period
7. Save results to timestamped CSV with expert, individual crowd, and crowd_average rows
8. Evaluate results using `G_scoring_answer.py`:
   - Fetch actual stock prices using yfinance
   - Adjust prediction dates to trading days (accounting for weekends/holidays)
   - Calculate accuracy and weighted scores
   - Compare expert vs crowd aggregate performance

## Key Implementation Details

### Model Configuration
- All models use JSON response format enforcement via OpenRouter
- Both expert and crowd models use "medium" effort level
- Max tokens not explicitly limited (relies on OpenRouter defaults)
- Error handling returns None instead of fallback predictions, allowing for proper failure tracking
- Failed predictions are logged with detailed error information (JSON decode errors, key errors, unexpected errors)

### Decision Aggregation
Crowd decisions use confidence-weighted voting per prediction period:
```python
# For each prediction period (day_1, day_3, day_7, day_15, day_30)
up_sum = sum(res["confidence"] for res in crowd_results if res["decision"] == "up")
down_sum = sum(res["confidence"] for res in crowd_results if res["decision"] == "down")
final_decision = "up" if up_sum > down_sum else "down"
avg_confidence = (up_sum if up_sum > down_sum else down_sum) / count
```

The crowd_average row in output CSV represents this aggregated decision.

### Scoring and Evaluation
G_scoring_answer.py implements comprehensive evaluation:
1. **Trading Day Adjustment**: Uses US Federal Holiday Calendar to adjust prediction dates to actual trading days
2. **Price Matching**: Fetches actual stock prices via yfinance for search_date and prediction_date
3. **Accuracy Metrics**: Simple accuracy (sign matching) and weighted score (confidence * sign, where sign is +1 for correct, -1 for incorrect)
4. **Crowd Aggregation for Scoring**: Re-aggregates individual crowd predictions using confidence-weighted voting
5. **Period-wise Comparison**: Expert vs Crowd accuracy and weighted scores for each prediction horizon (1, 3, 7, 15, 30 days)
6. **Output Files**:
   - `scoring_results_detailed_*.csv`: All individual predictions with actual prices and correctness
   - `scoring_results_crowd_*.csv`: Aggregated crowd predictions
   - `scoring_comparison_*.csv`: Period-by-period expert vs crowd comparison
   - `scoring_summary_*.csv`: Overall accuracy and weighted score summary

I_stats_test.py adds statistical significance testing:
- Mann-Whitney U-test and T-test for crowd vs expert performance comparison
- Classification metrics (precision, recall, F1-score) for up/down predictions
- Confidence calibration analysis

## External Dependencies

### Required Python Packages
- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical operations for scoring
- **openai**: OpenAI/OpenRouter API client
- **python-dotenv**: Environment variable management (imported as `dotenv`)
- **yfinance**: Yahoo Finance API for stock prices and market cap data
- **scipy**: Statistical tests (Mann-Whitney U-test, T-test) in I_stats_test.py
- **scikit-learn**: Classification metrics in I_stats_test.py

### API Services
- **OpenRouter**: Proxy service for accessing multiple LLM providers (requires `OPENROUTER_API_KEY`)
- **Yahoo Finance**: Historical stock prices, market data, and market capitalization

## Important Notes

### Checkpoint and Resume Functionality
Both baseline.py and baseline_v2.py support checkpoint/resume:
- When interrupted (Ctrl+C), progress is saved to `checkpoint_*.json`
- On restart, the script detects existing checkpoints and offers to resume
- Completed events are tracked and skipped on resume
- Checkpoint files are automatically deleted upon successful completion

### Failure Handling
- Failed predictions return None and are excluded from aggregation (not counted as predictions)
- All failures are logged to `prediction_failures_*.csv` with detailed error information
- Data loading errors (empty news data) are also logged
- Each log entry includes: symbol, search_date, model_name, model_type, persona_id, agent_index, error_type, error_message, timestamp

### Parallel Processing Differences
- **baseline.py**: Sequential event processing, parallel crowd agent execution (max_workers=10)
- **baseline_v2.py**: Parallel event processing (max_workers=10), parallel crowd agent execution (unlimited workers)
- Thread-safe file operations using locks in baseline_v2.py