# 📊 Peer Stock Analysis Dashboard - README

Welcome to the **Peer Stock Analysis Dashboard**, an interactive Streamlit app that helps investors compare a company to its peers based on key financial, valuation, risk, and growth metrics. It also integrates recent news sentiment to give a qualitative perspective on a stock.

---

## 🚀 Features

### 🧠 News Summary & Sentiment
- Retrieves recent news headlines using the [FMP API](https://financialmodelingprep.com/developer/docs/news-api/).
- Summarizes the articles using a lightweight BART model (`sshleifer/distilbart-cnn-12-6`).
- Analyzes sentiment using FinBERT (`ProsusAI/finbert`).

### 📈 Fundamental & Financial Comparison
- Pulls data for the input ticker and its **peer universe**.
- Shows:
  - **EPS & Revenue Growth** (1-year and 3-year CAGR)
  - **Valuation Ratios** (Fwd P/E, P/S, 5-year averages)
  - **Cash Flow Growth**
  - **Price Risk Metrics** (VAR, CVAR, Volatility, Drops)

### 💡 Recommendation Engine
- Scores stocks using a weighted formula based on:
  - Revenue & EPS growth
  - Free cash flow CAGR
  - Operating margin
  - Valuation (Fwd P/S)
- Highlights the **Top Stock Pick** based on the score.

### 📉 Additional News Sentiment
- You can analyze a new ticker or see sentiment for the recommended one.

---

## 🤖 How LLMs Are Used

This app leverages pre-trained large language models (LLMs) from Hugging Face Transformers to enhance financial insights:

- **Summarization (BART)**:
  - Model: `sshleifer/distilbart-cnn-12-6`
  - Task: Takes long news article content and breaks it down into short, digestible summaries.
  - Usage: Applied to the most recent 5 headlines pulled from Financial Modeling Prep's news API.

- **Sentiment Analysis (FinBERT)**:
  - Model: `ProsusAI/finbert`
  - Task: Classifies financial text as Positive, Negative, or Neutral with confidence scores.
  - Usage: Runs on the summarized news content to derive an overall sentiment tone per stock.

These models help provide a quick snapshot of how the media is reacting to a stock without reading through each article.

---

## 🧪 How It Works

### 1. Input Ticker
You start by typing a stock ticker (e.g. `MSFT`).

### 2. Peer Building
The app fetches peers from FMP's `/stock-peers` endpoint and retrieves financial data for:
- Ratios (TTM)
- Valuation history (5 years)
- Income & cash flow statements
- Analyst estimates
- Price target consensus
- Price and risk metrics (via `yfinance`)

### 3. Score Calculation
Metrics are normalized and scored based on importance:
```python
weights = {
    'Three Year Revenue CAGR': 0.20,
    'Revenue Growth Next Year': 0.15,
    'EPS % Change': 0.15,
    'Fwd PS': 0.20,
    'Operating Margin TTM': 0.15,
    '3 Year Free Cashflow CAGR': 0.15
}
```

### 4. Sentiment Summary
- News is pulled and summarized.
- Sentiment is displayed with a confidence score.

---

## 📊 Explanation of Metrics

### EPS & Revenue Growth
- **EPS Last/This/Next Year**: Earnings per Share over time.
- **EPS % Change**: Percentage change from current to next year's estimated EPS.
- **Revenue Growth This/Next Year**: Projected growth rates from analyst estimates.
- **Three Year Revenue CAGR**: Compound Annual Growth Rate of revenue over the past 3 years.

### Valuation Metrics
- **Fwd PE**: Forward Price to Earnings ratio (current price / estimated EPS).
- **Avg PE 5 Yr**: Historical average P/E ratio over the last 5 years.
- **Fwd PS**: Forward Price to Sales ratio (market cap / estimated revenue).
- **Avg PS 5 Yr**: 5-year historical average P/S ratio.
- **Operating Margin TTM**: Operating profit as a percentage of revenue (Trailing 12 Months).
- **Operating Margin 3 Yr Ago**: Historical comparison of margins.
- **Debt To Equity TTM**: Financial leverage metric comparing total debt to shareholders' equity.

### Financial Growth (CAGR)
- **3 Year Net Income CAGR**: Compound annual growth of net income.
- **3 Year Operating Cashflow CAGR**: CAGR of cash flow from operations.
- **3 Year Free Cashflow CAGR**: Growth in cash remaining after capex.

### Price & Risk Metrics
- **Total 5 Year Return**: Cumulative price return over 5 years.
- **Total 1 Year Return**: Price return over 1 year.
- **Annualized Return**: 5-year return converted into annual rate.
- **Annualized Volatility**: Price fluctuation (standard deviation) annualized.
- **VAR (Value at Risk)**: Estimated worst 1% return in a single day.
- **CVAR (Conditional VAR)**: Average loss beyond VAR threshold.
- **Largest 1/5 Day Drop**: Max daily and 5-day loss in past 5 years.

---

## 🧩 Technologies Used

- `streamlit` — for the dashboard
- `pandas`, `numpy` — data wrangling
- `yfinance` — price & volatility data
- `transformers` — for BART and FinBERT models
- `FinancialModelingPrep API` — fundamental & estimate data

---

## ⚙️ Setup & Deployment

### Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deployed on Hugging Face Spaces:
> ✅ Live App: [https://huggingface.co/spaces/andrewnap211/Finance-Dashboard](https://huggingface.co/spaces/andrewnap211/Finance-Dashboard)

To deploy your own:
1. Upload `app.py`, `requirements.txt`, and `README.md`
2. Make sure `transformers` version >= 4.38.1
3. Add `sentencepiece` if using summarization
4. Set Space SDK to `Streamlit`

---

## 📬 Contact
Feel free to reach out if you'd like help deploying or extending the app!
email: andnap@optonline.net

> Made with 💼 and 💻 by a finance + ML enthusiast.
