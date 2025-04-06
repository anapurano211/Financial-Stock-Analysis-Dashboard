import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import re

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# === CONFIG ===
API_KEY = '3c42fc6567276d6d31b84b10e283828e'

def fetch_peers(ticker):
    try:
        url = f'https://financialmodelingprep.com/stable/stock-peers?symbol={ticker}&apikey={API_KEY}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return [entry['symbol'] for entry in data if 'symbol' in entry]
    except Exception as e:
        print(f"Error fetching peers for {ticker}: {e}")
    return []

def fetch_profiles(tickers):
    df_list = []
    for stock in tickers:
        try:
            url = f'https://financialmodelingprep.com/stable/profile?symbol={stock}&apikey={API_KEY}'
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and data:
                for entry in data:
                    entry['ticker'] = stock
                    df_list.append(entry)
        except Exception as e:
            print(f"Error fetching profile for {stock}: {e}")
    return pd.DataFrame(df_list) if df_list else pd.DataFrame()

def build_us_peer_universe(start_tickers):
    all_tickers = set(start_tickers)
    for ticker in start_tickers:
        all_tickers.update(fetch_peers(ticker))
    profile_df = fetch_profiles(list(all_tickers))
    if profile_df.empty:
        return []
    return profile_df[(profile_df['country'] == 'US') & (profile_df['isActivelyTrading'] == True)]['symbol'].dropna().unique().tolist()

def fetch_ratios_ttm(tickers):
    df_list = []
    for stock in tickers:
        try:
            url = f'https://financialmodelingprep.com/stable/ratios-ttm?symbol={stock}&apikey={API_KEY}'
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and data:
                for entry in data:
                    entry['ticker'] = stock
                    df_list.append(entry)
        except Exception as e:
            print(f"Error fetching ratios for {stock}: {e}")
    if not df_list:
        return pd.DataFrame()
    df = pd.DataFrame(df_list)
    #df.rename(columns={'symbol': 'ticker'}, inplace=True)
    return df[['ticker', 'operatingProfitMarginTTM', 'debtToEquityRatioTTM']]

def fetch_5yr_pe_ps_averages(tickers):
    df_list = []
    for stock in tickers:
        try:
            url = f'https://financialmodelingprep.com/api/v3/ratios/{stock}?period=annual&apikey={API_KEY}'
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                for entry in data:
                    entry['ticker'] = stock
                    df_list.append(entry)
        except Exception as e:
            print(f"Error fetching 5yr P/E and P/S for {stock}: {e}")
    if not df_list:
        return pd.DataFrame()
    df = pd.DataFrame(df_list)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['ticker', 'date'], ascending=[True, True], inplace=True)
    for col in ['priceEarningsRatio', 'priceToSalesRatio']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['five_year_avg_PE'] = df.groupby('ticker')['priceEarningsRatio'].transform(lambda x: x.mask(x <= 0).rolling(5, min_periods=1).mean())
    df['five_year_avg_PS'] = df.groupby('ticker')['priceToSalesRatio'].transform(lambda x: x.mask(x <= 0).rolling(5, min_periods=1).mean())
    df = df[df['date'] == df.groupby('ticker')['date'].transform('max')]
    return df[['ticker', 'five_year_avg_PE', 'five_year_avg_PS']]

def calculate_signed_cagr(current, past, years=3):
    if pd.isna(current) or pd.isna(past) or past == 0:
        return np.nan
    try:
        growth = (abs(current) / abs(past)) ** (1 / years) - 1
        return growth if abs(current) >= abs(past) else -growth
    except Exception:
        return np.nan


def fetch_income_statement_and_cagr(tickers):
    df_list = []
    for stock in tickers:
        try:
            url = f'https://financialmodelingprep.com/api/v3/income-statement/{stock}?period=annual&apikey={API_KEY}'
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            for entry in data:
                entry['ticker'] = stock
                df_list.append(entry)
        except Exception as e:
            print(f"Error fetching income statement for {stock}: {e}")
    if not df_list:
        return pd.DataFrame()
    df = pd.DataFrame(df_list)
    df = df[["ticker", "date", "revenue", "operatingIncome", "eps", "netIncome"]]
    df[["revenue", "operatingIncome", "eps", "netIncome"]] = df[["revenue", "operatingIncome", "eps", "netIncome"]].apply(pd.to_numeric, errors='coerce')
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by=["ticker", "date"], ascending=[True, True], inplace=True)
    df['op_margin'] = np.where(df['revenue'] != 0, df['operatingIncome'] / df['revenue'], np.nan)
    df["revenue_3y_ago"] = df.groupby("ticker")["revenue"].shift(3)
    df["netincome_3y_ago"] = df.groupby("ticker")["netIncome"].shift(3)
    df["three_year_rev_cagr"] = df.apply(lambda row: calculate_signed_cagr(row["revenue"], row["revenue_3y_ago"]), axis=1)
    df["three_year_netincome_cagr"] = df.apply(lambda row: calculate_signed_cagr(row["netIncome"], row["netincome_3y_ago"]), axis=1)
    df["op_margin_3y_ago"] = df.groupby("ticker")["op_margin"].shift(3)
    df["year"] = df["date"].dt.year
    df["max_date"] = df.groupby("ticker")["date"].transform("max")
    df_latest_income = df[df["date"] == df["max_date"]].drop(columns=["max_date"])
    df_latest_income["this_year"] = df_latest_income["date"] + pd.DateOffset(years=1)
    df_latest_income["next_year"] = df_latest_income["date"] + pd.DateOffset(years=2)
    return df_latest_income

def fetch_cashflow_cagr(tickers):
    df_list = []
    for stock in tickers:
        try:
            url = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{stock}?period=annual&apikey={API_KEY}'
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            for entry in data:
                entry['ticker'] = stock
                df_list.append(entry)
        except Exception as e:
            print(f"Error fetching cashflow for {stock}: {e}")
    if not df_list:
        return pd.DataFrame()
    df = pd.DataFrame(df_list)
    for col in ['netIncome', 'operatingCashFlow', 'freeCashFlow']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['ticker', 'date'], inplace=True)
    df['ni_3y_ago'] = df.groupby('ticker')['netIncome'].shift(3)
    df['oc_3y_ago'] = df.groupby('ticker')['operatingCashFlow'].shift(3)
    df['fcf_3y_ago'] = df.groupby('ticker')['freeCashFlow'].shift(3)
    df['ni_cagr'] = df.apply(lambda row: calculate_signed_cagr(row['netIncome'], row['ni_3y_ago']), axis=1)
    df['oc_cagr'] = df.apply(lambda row: calculate_signed_cagr(row['operatingCashFlow'], row['oc_3y_ago']), axis=1)
    df['fcf_cagr'] = df.apply(lambda row: calculate_signed_cagr(row['freeCashFlow'], row['fcf_3y_ago']), axis=1)
    df = df[df['date'] == df.groupby('ticker')['date'].transform('max')]
    return df[['ticker', 'netIncome', 'operatingCashFlow', 'freeCashFlow', 'ni_cagr', 'oc_cagr', 'fcf_cagr']]

def fetch_analyst_estimates(tickers):
    df_list = []
    for stock in tickers:
        try:
            url = f'https://financialmodelingprep.com/api/v3/analyst-estimates/{stock}?apikey={API_KEY}'
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            for entry in data:
                entry['ticker'] = stock
                df_list.append(entry)
        except Exception as e:
            print(f"Error fetching analyst estimates for {stock}: {e}")
    if not df_list:
        return pd.DataFrame()
    df = pd.DataFrame(df_list)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['ticker', 'date'], ascending=[True, False], inplace=True)
    df['date_start'] = df['date']
    df['date_end'] = df.groupby('ticker')['date'].shift(1)
    df.rename(columns={"date": "earnings_date"}, inplace=True)
    return df[['ticker','earnings_date','estimatedRevenueAvg', 'estimatedEpsAvg', 'date_start', 'date_end']]

def fetch_price_target_consensus(tickers):
    df_list = []
    for stock in tickers:
        try:
            url = f'https://financialmodelingprep.com/api/v4/price-target-consensus?symbol={stock}&apikey={API_KEY}'
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            for entry in data:
                entry['ticker'] = stock
                df_list.append(entry)
        except Exception as e:
            print(f"Error fetching price target for {stock}: {e}")
    return pd.DataFrame(df_list)

def fetch_yfinance_metrics(tickers):
    try:
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

        infos = [yf.Ticker(t).info for t in tickers]
        df_infos = pd.DataFrame(infos).set_index('symbol')
        fundamentals = ['longName', 'currentPrice', 'marketCap']
        stock_list_info = df_infos[df_infos.columns[df_infos.columns.isin(fundamentals)]]

        stock_prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
        rets = stock_prices['Adj Close'].pct_change()
        cumulative_return = (1 + rets).cumprod() - 1
        total_five_year_return = cumulative_return.iloc[[-1]].T.rename(columns={cumulative_return.iloc[[-1]].index[0]: 'Total 5 Year Return'})

        one_year_start = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        one_year_prices = yf.download(tickers, start=one_year_start, end=end_date, auto_adjust=False)['Adj Close']
        one_year_return = one_year_prices.pct_change().add(1).cumprod().iloc[-1] - 1
        one_year_return = one_year_return.to_frame(name="Total 1 Year Return")

        price_change = stock_prices['Close']
        largest_1_day_drop = price_change.pct_change().agg(['min']).T.rename(columns={'min': 'Largest 1 Day Drop'})
        largest_5_day_drop = price_change.pct_change(-5).agg(['min']).T.rename(columns={'min': 'Largest 5 Day Drop'})

        def val_risk_return(returns_df):
            risk = pd.DataFrame({'VAR': returns_df.quantile(0.01)})
            risk['CVAR'] = returns_df[returns_df < returns_df.quantile(0.01)].mean()
            return risk

        def ann_risk_return(returns_df):
            summary = returns_df.agg(['std']).T
            summary.columns = ['Risk']
            summary['Risk'] = summary['Risk'] * np.sqrt(252)
            return summary

        risk = val_risk_return(rets)
        vol_sum = ann_risk_return(rets)

        stock_data = stock_list_info.join([total_five_year_return, one_year_return, largest_1_day_drop,
                                           largest_5_day_drop, risk, vol_sum], how='inner')

        stock_data = stock_data.rename(columns={
            'longName': 'Company/ETF Name',
            'currentPrice': 'Current Stock Price',
            'Risk': 'Annualized Volatility',
            'marketCap': 'Market Cap'
        })
        stock_data['Annualized Return'] = (1 + stock_data['Total 5 Year Return']) ** (1 / 5) - 1
        stock_data.reset_index(inplace=True)
        stock_data.rename(columns={"index": "ticker"}, inplace=True)
        return stock_data
    except Exception as e:
        print(f"Error in fetch_yfinance_metrics: {e}")
        return pd.DataFrame()

# === Streamlit UI ===
st.set_page_config(page_title="Peer Stock Analysis Dashboard", layout="wide")
st.title("📊 Peer Stock Analysis Dashboard")
st.markdown("""
Enter a stock ticker to fetch peer companies and display:
1. EPS and Revenue Growth
2. Valuation Metrics
3. Financial Growth (CAGR)
4. Price Performance & Risk
""")

# === News Sentiment Section ===
#from transformers import pipeline

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig




summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6")

#model_name = "andrewnap211/distilbart-cnn-12-6-local"

#model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#generation_config = GenerationConfig.from_pretrained(model_name)

#summarizer = pipeline(
   # "summarization",
    #model=model,
   # tokenizer=tokenizer,
   # generation_config=generation_config
#)


#summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
#summarizer = pipeline("summarization", model="andrewnap211/distilbart-cnn-12-6-local")
sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert")





FMP_API_KEY = API_KEY  # reuse same API key

def get_stock_news(ticker):
    url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit=5&apikey={FMP_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        news_json = response.json()

        if not news_json:
            return []

        news_data = []
        seen_titles = set()

        for article in news_json:
            title = article.get("title", "").strip()
            summary = article.get("text", "No summary available").strip()
            link = article.get("url", "#")
            published_date = article.get("publishedDate", "")

            if published_date:
                article_date = datetime.strptime(published_date, "%Y-%m-%d %H:%M:%S")
                formatted_date = article_date.strftime("%Y-%m-%d")
            else:
                formatted_date = "Unknown"

            if title and title not in seen_titles:
                news_data.append({"date": formatted_date, "title": title, "summary": summary, "link": link})
                seen_titles.add(title)

        return news_data

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news for {ticker}: {e}")
        return []

def split_and_summarize(text, max_tokens=1024):
    words = text.split()
    if len(words) <= max_tokens:
        return summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
    chunk_size = max_tokens
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    summaries = summarizer(chunks, max_length=100, min_length=40, do_sample=False)
    return " ".join(summary["summary_text"] for summary in summaries)

def summarize_news(news_data):
    if not news_data:
        return "No recent news available."
    summaries = [article["summary"] for article in news_data if article["summary"] != "No summary available"]
    if len(summaries) < 2:
        return "Not enough news data for a reliable summary."
    full_text = " ".join(summaries)
    return split_and_summarize(full_text)

def analyze_sentiment(summary_text):
    result = sentiment_analyzer(summary_text[:512])[0]
    sentiment_label = result["label"]
    score = result["score"]
    if sentiment_label.lower() == "positive":
        interpretation = "🟢 Positive"
    elif sentiment_label.lower() == "negative":
        interpretation = "🔴 Negative"
    else:
        interpretation = "⚪ Neutral"
    return score, interpretation

# === Streamlit Input ===
ticker_input = st.text_input("Enter a stock ticker (e.g., MSFT):")

if ticker_input:
    with st.spinner("Analyzing recent news and sentiment..."):
        news_data = get_stock_news(ticker_input)
        if news_data:
            st.subheader(f"📰 Recent News for {ticker_input.upper()}")
            for article in news_data:
                st.markdown(f"**{article['date']} - {article['title']}**")
                st.markdown(f"{article['summary']}")
                st.markdown(f"[Read more]({article['link']})")
                st.markdown("---")

            summary = summarize_news(news_data)
            sentiment_score, sentiment_label = analyze_sentiment(summary)

            st.markdown(f"### 🧠 Summary Across Headlines")
            st.info(summary)
            st.markdown(f"### 📉 Sentiment: {sentiment_label} (Confidence: {sentiment_score:.2f})")
        else:
            st.warning("No recent news found for this ticker.")

if st.button("Run Analysis"):
    if not ticker_input:
        st.error("Please enter a valid stock ticker to proceed.")
    else:
        START_TICKERS = [ticker_input]
        with st.spinner("Fetching and processing data. This may take a minute..."):
            try:
                tickers = build_us_peer_universe(START_TICKERS)

                df_curr_metrics = fetch_ratios_ttm(tickers)
                df_hist_metrics = fetch_5yr_pe_ps_averages(tickers)
                df_income_cagr = fetch_income_statement_and_cagr(tickers)
                df_cashflow_cagr = fetch_cashflow_cagr(tickers)
                df_estimates = fetch_analyst_estimates(tickers)
                df_price_target = fetch_price_target_consensus(tickers)
                df_yf_metrics = fetch_yfinance_metrics(tickers)

                # Aggregation steps
                df_income_this_years_earnings = df_income_cagr.merge(df_estimates, on="ticker", how="left")
                df_income_this_years_earnings = df_income_this_years_earnings[(df_income_this_years_earnings["this_year"] >= df_income_this_years_earnings["date_start"]) & 
                                                                               (df_income_this_years_earnings["this_year"] < df_income_this_years_earnings["date_end"])]
                df_income_this_years_earnings.reset_index(drop=True, inplace=True)
                df_income_this_years_earnings.rename(columns={"earnings_date": "this_years_earnings_date",
                                                            "estimatedRevenueAvg": "revenue_this_year",
                                                            "estimatedEpsAvg": "eps_this_year",
                                                            "date_start": "this_year_date_start",
                                                            "date_end": "this_year_date_end"}, inplace=True)

                df_income_this_years_earnings = df_income_this_years_earnings.merge(df_estimates, on="ticker", how="left")
                df_income_this_years_earnings = df_income_this_years_earnings[(df_income_this_years_earnings["next_year"] >= df_income_this_years_earnings["date_start"]) & 
                                                                               (df_income_this_years_earnings["next_year"] < df_income_this_years_earnings["date_end"])]
                df_income_this_years_earnings.reset_index(drop=True, inplace=True)
                df_income_this_years_earnings.rename(columns={"earnings_date": "next_years_earnings_date",
                                                            "estimatedRevenueAvg": "revenue_next_year",
                                                            "estimatedEpsAvg": "eps_next_year",
                                                            "date_start": "next_year_date_start",
                                                            "date_end": "next_year_date_end"}, inplace=True)

                stock_data = df_income_this_years_earnings.merge(df_yf_metrics, on="ticker", how="left")
                stock_data = stock_data.merge(df_curr_metrics, on="ticker", how="left")
                stock_data = stock_data.merge(df_hist_metrics, on="ticker", how="left")
                stock_data = stock_data.merge(df_cashflow_cagr, on="ticker", how="left")
                stock_data = stock_data.merge(df_income_cagr, on="ticker", how="left")
                stock_data = stock_data.merge(df_price_target, on="ticker", how="left")

                stock_data['this_year_rev_growth'] = (stock_data['revenue_this_year'] / stock_data['revenue_x']) - 1
                stock_data['next_year_rev_growth'] = (stock_data['revenue_next_year'] / stock_data['revenue_this_year']) - 1
                stock_data['pe_fwd'] = stock_data['Current Stock Price'] / stock_data['eps_this_year']
                stock_data['ps_fwd'] = stock_data['Market Cap'] / stock_data['revenue_this_year']

                eps_rev_df = stock_data.loc[:,['Company/ETF Name','ticker','eps_x','eps_this_year','eps_next_year','this_year_rev_growth','next_year_rev_growth','three_year_rev_cagr_y']]
                eps_rev_df.rename(columns={"eps_x": "EPS Last Year","eps_this_year": "EPS This Year","eps_next_year": "EPS Next Year",
                                           "this_year_rev_growth": "Revenue Growth This Year" ,"next_year_rev_growth": "Revenue Growth Next Year",
                                           "three_year_rev_cagr_y":"Three Year Revenue CAGR"}, inplace=True)

                valuation_df = stock_data.loc[:,['Company/ETF Name','ticker','pe_fwd','five_year_avg_PE','ps_fwd','five_year_avg_PS','operatingProfitMarginTTM','op_margin_3y_ago_y','debtToEquityRatioTTM']]
                valuation_df.rename(columns={"pe_fwd": "Fwd PE","five_year_avg_PE": "Avg PE 5 Yr","ps_fwd": "Fwd PS","five_year_avg_PS": "Avg PS 5 yr",
                                             "operatingProfitMarginTTM": "Operating Margin TTM","op_margin_3y_ago_y":"Operating Margin 3 Yr Ago",
                                             "debtToEquityRatioTTM":"Debt To Equity TTM"}, inplace=True)

                growth_df = stock_data.loc[:,['Company/ETF Name','ticker','ni_cagr','oc_cagr','fcf_cagr']]
                growth_df.rename(columns={"ni_cagr": "3 Year Net Income CAGR","oc_cagr": "3 Year Operating Cashflow CAGR",
                                          "fcf_cagr": "3 Year Free Cashflow CAGR"}, inplace=True)

                price_df = stock_data.loc[:,['Company/ETF Name','ticker','Total 5 Year Return','Total 1 Year Return','Annualized Return',
                                             'Annualized Volatility','VAR','CVAR','Largest 1 Day Drop','Largest 5 Day Drop']]

                st.success("✅ Analysis Complete!")

                # === EPS & Revenue Growth Section (Enhanced) ===
                st.subheader("📈 EPS & Revenue Growth")

# Add EPS % Change (as absolute percent growth)
                eps_rev_df['EPS % Change'] = (eps_rev_df['EPS Next Year'] - eps_rev_df['EPS This Year']) / abs(eps_rev_df['EPS This Year'])

# Reorder columns to put EPS % Change right after EPS Next Year
                cols = eps_rev_df.columns.tolist()
                if 'EPS % Change' in cols:
                    cols.insert(cols.index('EPS Next Year') + 1, cols.pop(cols.index('EPS % Change')))
                    eps_rev_df = eps_rev_df[cols]

                    # Columns for heatmapst
                percentage_cols = ['EPS % Change', 'Revenue Growth This Year', 'Revenue Growth Next Year', 'Three Year Revenue CAGR']
                
                eps_cols = ['EPS Last Year', 'EPS This Year', 'EPS Next Year']
                for col in eps_cols:
                    if col in eps_rev_df.columns:
                       eps_rev_df[col] = eps_rev_df[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")


                numeric_df = eps_rev_df.copy()
                for col in percentage_cols:
                     if numeric_df[col].dtype == 'object':
                         numeric_df[col] = numeric_df[col].str.replace('%', '').astype(float) / 100

                formatted_df = numeric_df.copy()
                for col in percentage_cols:
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")


                styled_df = formatted_df.style
                for col in percentage_cols:
                     numeric_col = numeric_df[col].copy()

                     positive_col = numeric_col[numeric_col > 0]
                     top_idxs = positive_col.nlargest(2).index
                   
                     def highlight_top_2_col(s, top=top_idxs):
                         return ['background-color: lightblue' if i in top else '' for i in s.index]

                     styled_df = styled_df.apply(highlight_top_2_col, subset=[col], axis=0)
                     bottom_idxs = numeric_col.nsmallest(2).index
                     def highlight_bottom_2_col(s, bottom=bottom_idxs):
                         return ['background-color: salmon' if i in bottom else '' for i in s.index]
                     styled_df = styled_df.apply(highlight_bottom_2_col, subset=[col], axis=0)






                st.dataframe(styled_df, use_container_width=True)
                
                
                # === Remaining Sections ===
                st.subheader("💸 Valuation Metrics")
                pe_ps_cols = ["Fwd PE", "Avg PE 5 Yr", "Fwd PS", "Avg PS 5 yr"]
                percentage_cols = ["Operating Margin TTM", "Operating Margin 3 Yr Ago", "Debt To Equity TTM"]
                val_cols = pe_ps_cols + percentage_cols

                valuation_numeric = valuation_df.copy()
                formatted_valuation = valuation_numeric.copy()

                for col in val_cols:
                    if col in percentage_cols:
                        formatted_valuation[col] = formatted_valuation[col].apply(
                            lambda x: f"{x * 100:.1f}%" if pd.notnull(x) else "N/A"
                        )
                    else:
                        formatted_valuation[col] = formatted_valuation[col].apply(
                            lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A"
                        )
                styled_valuation = formatted_valuation.style

                for col in pe_ps_cols:
                    numeric_col = valuation_numeric[col].copy()
                    numeric_col_nonnull = numeric_col.dropna()

                    pos_vals = numeric_col_nonnull[numeric_col_nonnull >= 0]
                    top_idxs = pos_vals.nsmallest(2).index if not pos_vals.empty else []
                    bottom_idxs = numeric_col_nonnull.index.difference(top_idxs)

                    def highlight_peps_col(s, top=top_idxs, bottom=bottom_idxs):
                        return [
                          'background-color: lightblue' if i in top
                          else 'background-color: salmon' if i in bottom and pd.notnull(s[i])
                          else ''
                          for i in s.index
                    ]

                    styled_valuation = styled_valuation.apply(highlight_peps_col, subset=[col], axis=0)

                for col in percentage_cols:
                    numeric_col = valuation_numeric[col].copy()
                    numeric_col_nonnull = numeric_col.dropna()

                    if col == "Debt To Equity TTM":

                         # For Debt to Equity, lower is better
                       top_idxs = numeric_col_nonnull.nsmallest(2).index  # lowest
                       bottom_idxs = numeric_col_nonnull.nlargest(2).index  # highest
                    else:
                        top_idxs = numeric_col_nonnull.nlargest(2).index
                        bottom_idxs = numeric_col_nonnull.nsmallest(2).index

                    def highlight_top_bottom(s, top=top_idxs, bottom=bottom_idxs):
                        return [
                           'background-color: lightblue' if i in top
                        else 'background-color: salmon' if i in bottom
                        else ''
                        for i in s.index
                          ]

                    styled_valuation = styled_valuation.apply(highlight_top_bottom, subset=[col], axis=0)

# Display the table
                st.dataframe(styled_valuation, use_container_width=True)



                st.subheader("📊 Financial Growth (CAGR)")
                growth_numeric = growth_df.copy()
                non_numeric_cols = ["Company/ETF Name", "ticker"]
                numeric_cols = [col for col in growth_df.columns if col not in non_numeric_cols]



                for col in numeric_cols:
                    growth_numeric[col] = pd.to_numeric(growth_numeric[col], errors='coerce')

                formatted_growth = growth_numeric.copy()

                for col in numeric_cols:
                    formatted_growth[col] = formatted_growth[col].apply(
                          lambda x: f"{x * 100:.1f}%" if pd.notnull(x) else "N/A"
                    )
                      
                styled_growth = formatted_growth.style
                for col in numeric_cols:
                    numeric_col = growth_numeric[col].copy()
                    numeric_col_nonnull = numeric_col.dropna()
                    top_idxs = numeric_col_nonnull.nlargest(2).index
                    bottom_idxs = numeric_col_nonnull.nsmallest(2).index
                    def highlight_top_bottom(s, top=top_idxs, bottom=bottom_idxs):
                        return [
                           'background-color: lightblue' if i in top
                          else 'background-color: salmon' if i in bottom
                          else '' for i in s.index
                        ]
                    styled_growth = styled_growth.apply(highlight_top_bottom, subset=[col], axis=0)





                st.dataframe(styled_growth, use_container_width=True)
              
                st.subheader("📉 Price & Risk Metrics")
                reverse_cols = ["Annualized Volatility"]
                percent_cols = [col for col in price_df.columns if col not in ["Company/ETF Name", "ticker"]]
                price_numeric = price_df.copy()
                for col in percent_cols:
                    price_numeric[col] = pd.to_numeric(price_numeric[col], errors='coerce')
                formatted_price = price_numeric.copy()
                for col in percent_cols:
                    formatted_price[col] = formatted_price[col].apply(lambda x: f"{x * 100:.1f}%" if pd.notnull(x) else "N/A")

                styled_price = formatted_price.style

                for col in percent_cols:
                    numeric_col = price_numeric[col].copy()
                    numeric_col_nonnull = numeric_col.dropna()

                    if col in reverse_cols:
                        top_idxs = numeric_col_nonnull.nsmallest(2).index
                        bottom_idxs = numeric_col_nonnull.nlargest(2).index
                    else:
                        top_idxs = numeric_col_nonnull.nlargest(2).index
                        bottom_idxs = numeric_col_nonnull.nsmallest(2).index


                    def highlight(s, top=top_idxs, bottom=bottom_idxs):
                         return [
                         'background-color: lightblue' if i in top
                         else 'background-color: salmon' if i in bottom
                         else ''
                         for i in s.index
                        ]
                    styled_price_df = styled_price.apply(highlight, subset=[col], axis=0)


                st.dataframe(styled_price, use_container_width=True)
                stock_data = stock_data.merge(
    eps_rev_df[['ticker', 'EPS % Change']],
    on='ticker',
    how='left'
)
                st.subheader("📌 Stock Recommendation")
                required_cols = ['ticker', 'three_year_rev_cagr_y', 'next_year_rev_growth', 'EPS % Change', 'ps_fwd', 'operatingProfitMarginTTM', 'fcf_cagr']
                recommend_df = stock_data.copy()
                recommend_df = recommend_df.rename(columns={
    'three_year_rev_cagr_y': 'Three Year Revenue CAGR',
    'next_year_rev_growth': 'Revenue Growth Next Year',
    'EPS % Change': 'EPS % Change',
    'ps_fwd': 'Fwd PS',
    'operatingProfitMarginTTM': 'Operating Margin TTM',
    'fcf_cagr': '3 Year Free Cashflow CAGR'})
                required_cols = [
    'ticker',
    'Three Year Revenue CAGR',
    'Revenue Growth Next Year',
    'EPS % Change',
    'Fwd PS',
    'Operating Margin TTM',
    '3 Year Free Cashflow CAGR']


                recommend_df = recommend_df[required_cols].dropna()
                recommend_df = recommend_df[recommend_df['Fwd PS'] > 0]  # Important: filter out invalid PS
                weights = {
    'Three Year Revenue CAGR': 0.20,
    'Revenue Growth Next Year': 0.15,
    'EPS % Change': 0.15,
    'Fwd PS': 0.20,
    'Operating Margin TTM': 0.15,
    '3 Year Free Cashflow CAGR': 0.15,
}
                norm_df = recommend_df.copy()
                for col in weights:
                    if col == 'Fwd PS':
                        norm_df[col] = recommend_df[col].apply(
            lambda x: 1 - ((x - recommend_df[col].min()) / (recommend_df[col].max() - recommend_df[col].min()))
        )
                    else:
                        norm_df[col] = (recommend_df[col] - recommend_df[col].min()) / (recommend_df[col].max() - recommend_df[col].min())

                norm_df['Score'] = sum(norm_df[col] * weights[col] for col in weights)
                final_scores = recommend_df[['ticker']].copy()
                final_scores['Score'] = norm_df['Score']
                final_scores = final_scores.sort_values('Score', ascending=False).reset_index(drop=True)
                st.dataframe(final_scores)
                top = final_scores.iloc[0]['ticker']
                st.markdown(f"""
                ✅ **Top Recommendation: `{top}`**

                   This stock ranks **#1** based on your custom investment scoring system:

                     - High **Revenue & EPS Growth**
                     - Efficient **Free Cashflow & Operating Margins**
                     - Attractive **Forward Price-to-Sales Ratio**

                     It has the strongest overall balance of growth, profitability, and valuation among its peers.
                            """)
                
                st.subheader("🧠 News Sentiment for Recommended or Custom Ticker")
                st.markdown("""Check recent news sentiment for the top recommended stock or enter a different ticker below:""")
                default_ticker = top if 'top' in locals() else ""
                bottom_ticker = st.text_input("Enter a stock ticker to check news sentiment again:", value=default_ticker)
                if bottom_ticker:
                     with st.spinner(f"Fetching and analyzing news sentiment for {bottom_ticker.upper()}..."):
                         bottom_news = get_stock_news(bottom_ticker)
                         if bottom_news:
                             st.subheader(f"📰 Recent News for {bottom_ticker.upper()}")
                             for article in bottom_news:
                                 st.markdown(f"**{article['date']} - {article['title']}**")
                                 st.markdown(f"{article['summary']}")
                                 st.markdown(f"[Read more]({article['link']})")
                                 st.markdown("---")

                             bottom_summary = summarize_news(bottom_news)
                             bottom_sentiment_score, bottom_sentiment_label = analyze_sentiment(bottom_summary)

                             st.markdown(f"### 🧠 Summary Across Headlines")
                             st.info(bottom_summary)
                             st.markdown(f"### 📉 Sentiment: {bottom_sentiment_label} (Confidence: {bottom_sentiment_score:.2f})")

                         else:
                             st.warning("No recent news found for this ticker.")

              


            except Exception as e:
                st.error(f"Something went wrong: {e}")


        

