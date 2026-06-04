"""
PaySprint AI Investment Agent - Complete Working Core Module
=========================================================
AAI-510 | Group 6 | University of San Diego

This file is the finished, working backbone of the project.
The three notebooks import from here - teammates do NOT need to edit this file
unless they want to improve something specific.

Notebook mapping:
  data_pipeline.ipynb   -> Reema Eid    (Data Engineer)   -> DATA LAYER section
  agent_definition.ipynb -> Hyunju Yu   (AI Engineer)      -> AGENT LAYER section
  traces_evaluation.ipynb -> Quang Tran (PM) -> EVALUATION section

Architecture: ReAct-style agent (Reason -> Act -> Observe -> Repeat)
LLMs used:
  - MODEL_REASONING (gpt-4o)      -> orchestrator + final report
  - MODEL_SUMMARY  (gpt-4o-mini)  -> per-stock quick summaries (lower cost)
"""

import os
import json
import sqlite3
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from openai import OpenAI

#  optional libs - gracefully skipped if not installed 
try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False
    print("[WARN] yfinance missing - run: pip install yfinance")

try:
    from gnews import GNews
    GNEWS_OK = True
except ImportError:
    GNEWS_OK = False
    print("[WARN] gnews missing - run: pip install gnews")

try:
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
    SIA = SentimentIntensityAnalyzer()
    VADER_OK = True
except Exception:
    VADER_OK = False
    SIA = None

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False
    nlp = None

try:
    from yahooquery import search as yq_search
    YQ_OK = True
except Exception:
    YQ_OK = False

try:
    import pandas_ta as ta
    TA_OK = True
except ImportError:
    TA_OK = False


# =============================================================================
# CONFIGURATION  (shared by all notebooks)
# =============================================================================

DB_PATH        = "paysprint.db"
TRACES_DIR     = "data/traces"
MODEL_REASONING = os.getenv("MODEL_REASONING", "gpt-4o")
MODEL_SUMMARY   = os.getenv("MODEL_SUMMARY",   "gpt-4o-mini")

# Scoring weights for stock ranking (sum must equal 1.0)
#  Hyunju can adjust these in agent_definition.ipynb 
SCORING_WEIGHTS = {
    "sentiment": 0.40,
    "momentum":  0.35,
    "mentions":  0.25,
}

# Stocks offered per strategy tier
#  Hyunju can add/remove tickers in agent_definition.ipynb 
SCREENER_STOCKS = {
    "conservative": ["JNJ", "PG", "KO", "VZ", "WMT", "MCD", "MMM", "ABT"],
    "moderate":     ["AAPL", "MSFT", "GOOGL", "V", "MA", "AMZN", "JPM", "HD"],
    "aggressive":   ["NVDA", "META", "TSLA", "AMD", "PLTR", "CRWD", "SNOW", "SMCI"],
}

# Trusted financial news publishers
#  Reema can add more sources in data_pipeline.ipynb 
TRUSTED_SOURCES = {
    "reuters", "bloomberg", "wsj", "cnbc", "financial times",
    "yahoo finance", "marketwatch", "barrons", "seeking alpha",
    "morningstar", "nasdaq", "forbes", "investopedia",
    "the wall street journal", "ft.com", "benzinga",
    "the motley fool", "fool.com", "thestreet", "zacks",
    "associated press", "apnews", "businesswire", "prnewswire",
}

# Trading-day horizons for price forecasts
H_3M  = 63   # ? 3 months
H_12M = 252  # ? 12 months

#  OpenAI client 
def _get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file or run:\n"
            "  import os; os.environ['OPENAI_API_KEY'] = 'sk-...'"
        )
    return OpenAI(api_key=key)


# =============================================================================
# DATA LAYER
# Used by: data_pipeline.ipynb  (Reema - Data Engineer)
# =============================================================================

#  Price history 
def fetch_price_history(ticker: str, lookback_days: int = 120) -> pd.Series:
    """
    Pull adjusted close prices from Yahoo Finance via yfinance.
    Returns a pd.Series indexed by date, sorted ascending.
    Returns an empty Series if the ticker is invalid or data is unavailable.
    """
    if not YF_OK:
        return pd.Series(dtype=float)
    try:
        df = yf.Ticker(ticker).history(period=f"{lookback_days}d")
        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float)
        s = df["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.sort_index()
    except Exception as e:
        print(f"[price] {ticker}: {e}")
        return pd.Series(dtype=float)


#  Fundamentals 
def fetch_fundamentals(ticker: str) -> dict:
    """
    Pull key fundamental metrics from Yahoo Finance.
    All missing values default to None so callers can check cleanly.
    Fields: pe_ratio, eps, revenue_growth, debt_to_equity,
            free_cash_flow, market_cap, sector, industry
    """
    if not YF_OK:
        return {}
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "ticker":         ticker,
            "pe_ratio":       info.get("trailingPE"),
            "eps":            info.get("trailingEps"),
            "revenue_growth": info.get("revenueGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cash_flow": info.get("freeCashflow"),
            "market_cap":     info.get("marketCap"),
            "sector":         info.get("sector"),
            "industry":       info.get("industry"),
        }
    except Exception as e:
        print(f"[fundamentals] {ticker}: {e}")
        return {"ticker": ticker}


#  News & sentiment 
def is_trusted(publisher: str) -> bool:
    """Return True if publisher is in TRUSTED_SOURCES (case-insensitive)."""
    pub = (publisher or "").lower().strip()
    return any(src in pub for src in TRUSTED_SOURCES)


def score_sentiment(text: str) -> float:
    """VADER compound sentiment score in [-1, 1]. Returns 0.0 if VADER unavailable."""
    if not VADER_OK or not SIA:
        return 0.0
    return float(SIA.polarity_scores(text or "")["compound"])


_TICKER_PATTERNS = [
    re.compile(r'\$([A-Z]{1,5})(?![A-Z])'),
    re.compile(r'\((?:NASDAQ|NYSE|AMEX):\s*([A-Z]{1,5})\)'),
]

def _extract_tickers(text: str) -> list:
    found = set()
    for pat in _TICKER_PATTERNS:
        found.update(pat.findall(text or ""))
    return list(found)


def fetch_news(query: str = "stock market", days: int = 30, max_results: int = 40) -> pd.DataFrame:
    """
    Fetch financial news articles via GNews.
    Returns a DataFrame with: title, summary, url, publisher, trusted, sentiment, tickers.
    Returns an empty DataFrame if GNews is unavailable or returns nothing.
    """
    empty = pd.DataFrame(columns=["title", "summary", "url", "publisher", "trusted", "sentiment", "tickers"])
    if not GNEWS_OK:
        print("[news] gnews not installed - returning empty DataFrame")
        return empty
    try:
        g    = GNews(language="en", country="US", period=f"{days}d", max_results=max_results)
        rows = g.get_news(query) or []
        records = []
        for r in rows:
            title = r.get("title", "")
            desc  = r.get("description", "")
            url   = r.get("url", "")
            pub   = (r.get("publisher") or {}).get("title", "")
            if not title or not url:
                continue
            combined = f"{title} {desc}"
            records.append({
                "title":     title,
                "summary":   desc,
                "url":       url,
                "publisher": pub,
                "trusted":   is_trusted(pub),
                "sentiment": score_sentiment(combined),
                "tickers":   _extract_tickers(combined),
            })
        return pd.DataFrame(records) if records else empty
    except Exception as e:
        print(f"[news] fetch failed: {e}")
        return empty


#  Technical indicators 
def compute_indicators(ticker: str, lookback_days: int = 120) -> dict:
    """
    Compute price-based indicators for a ticker:
      last_price, slope_per_day, forecast_3m, forecast_12m,
      volatility_30d, rsi (if pandas_ta installed), macd_signal.
    Returns empty dict if no price data is available.
    """
    closes = fetch_price_history(ticker, lookback_days)
    if closes.empty or len(closes) < 5:
        return {"ticker": ticker}

    arr = closes.values.astype(float)
    X   = np.arange(len(arr)).reshape(-1, 1)
    reg = LinearRegression().fit(X, arr)
    slope   = float(reg.coef_[0])
    last    = float(closes.iloc[-1])
    returns = closes.pct_change().dropna().tail(30)
    vol     = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0

    result = {
        "ticker":        ticker,
        "last_price":    round(last, 2),
        "slope_per_day": round(slope, 4),
        "forecast_3m":   round(last + slope * H_3M,  2),
        "forecast_12m":  round(last + slope * H_12M, 2),
        "volatility_30d": round(vol, 4),
        "rsi":           None,
        "macd_signal":   None,
    }

    if TA_OK:
        try:
            df_ta = pd.DataFrame({"close": closes})
            df_ta.ta.rsi(append=True)
            df_ta.ta.macd(append=True)
            rsi_cols  = [c for c in df_ta.columns if c.startswith("RSI")]
            macd_cols = [c for c in df_ta.columns if "MACDs" in c]
            if rsi_cols:
                result["rsi"] = round(float(df_ta[rsi_cols[0]].iloc[-1]), 2)
            if macd_cols:
                result["macd_signal"] = round(float(df_ta[macd_cols[0]].iloc[-1]), 4)
        except Exception:
            pass

    return result


#  SQLite persistence 
def init_db(db_path: str = DB_PATH):
    """Create database tables. Safe to call multiple times."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT,
            budget          REAL,
            aggressiveness  TEXT,
            horizon_months  INTEGER,
            current_holdings TEXT,
            preferred_sectors TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER,
            ticker         TEXT,
            allocation_usd REAL,
            shares         INTEGER,
            price          REAL,
            report_date    TEXT DEFAULT (date('now'))
        )
    """)
    conn.commit()
    conn.close()


def save_user_profile(profile: dict, db_path: str = DB_PATH) -> int:
    """Insert a user profile and return the new row id."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO user_profiles
            (name, budget, aggressiveness, horizon_months, current_holdings, preferred_sectors)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        profile.get("name", ""),
        profile.get("budget", 0),
        profile.get("aggressiveness", "moderate"),
        profile.get("horizon_months", 12),
        json.dumps(profile.get("current_holdings", {})),
        json.dumps(profile.get("preferred_sectors", [])),
    ))
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    return uid


def save_recommendations(user_id: int, plan: list, db_path: str = DB_PATH):
    """Persist a purchase plan list to the recommendations table."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    for item in plan:
        if item.get("ticker") == "CASH":
            continue
        cur.execute("""
            INSERT INTO recommendations (user_id, ticker, allocation_usd, shares, price)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, item["ticker"], item.get("allocation_usd"),
              item.get("shares"), item.get("price")))
    conn.commit()
    conn.close()


def load_recommendations(user_id: int = None, db_path: str = DB_PATH) -> pd.DataFrame:
    """Load saved recommendations. If user_id is None, returns all rows."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    if user_id:
        df = pd.read_sql("SELECT * FROM recommendations WHERE user_id = ?", conn, params=(user_id,))
    else:
        df = pd.read_sql("SELECT * FROM recommendations", conn)
    conn.close()
    return df


# =============================================================================
# AGENT LAYER
# Used by: agent_definition.ipynb  (Hyunju - AI Engineer)
# =============================================================================

#  Tool implementations 
def tool_screen_stocks(aggressiveness: str, sectors: list = None) -> dict:
    """
    Return candidate tickers for the given strategy.
    Source: SCREENER_STOCKS dict (editable by Hyunju in Notebook 2).
    Sector filter is informational only in this version.
    """
    candidates = SCREENER_STOCKS.get(aggressiveness.lower(), SCREENER_STOCKS["moderate"])
    return {"candidates": candidates, "strategy": aggressiveness}


def tool_technical_indicators(ticker: str, lookback_days: int = 120) -> dict:
    """Compute and return technical indicators for a single ticker."""
    result = compute_indicators(ticker, lookback_days)
    if len(result) <= 1:
        return {"ticker": ticker, "error": "No price data available"}
    return result


def tool_news_sentiment(ticker: str, days: int = 30) -> dict:
    """
    Aggregate news sentiment for a ticker from GNews.
    Returns avg_sentiment, article_count, and top 3 headlines.
    Falls back to neutral sentiment if no articles found.
    """
    df = fetch_news(query=ticker, days=days, max_results=30)
    if df.empty:
        return {"ticker": ticker, "avg_sentiment": 0.0, "article_count": 0, "top_headlines": []}

    # prefer articles that explicitly mention the ticker
    mask = df["tickers"].apply(lambda tl: ticker in tl)
    target = df[mask] if mask.any() else df

    return {
        "ticker":        ticker,
        "avg_sentiment": round(float(target["sentiment"].mean()), 3),
        "article_count": len(target),
        "top_headlines": target["title"].head(3).tolist(),
    }


def tool_fundamentals(ticker: str) -> dict:
    """Return fundamental metrics for a ticker (wrapper for agent tool use)."""
    result = fetch_fundamentals(ticker)
    if not result:
        return {"ticker": ticker, "error": "Fundamentals unavailable"}
    return result


def tool_purchase_plan(budget: float, tickers: list, weights: list) -> list:
    """
    Allocate a budget across tickers using supplied weights.
    Fetches the latest price for each ticker to compute share counts.
    Appends a CASH row for leftover funds.

    Args:
        budget:  Total dollars to invest
        tickers: List of ticker symbols
        weights: Fractional allocations that should sum to ~1.0
    Returns:
        List of dicts: {ticker, weight, allocation_usd, price, shares}
    """
    plan = []
    for ticker, weight in zip(tickers, weights):
        closes = fetch_price_history(ticker, lookback_days=5)
        price  = round(float(closes.iloc[-1]), 2) if not closes.empty else None
        alloc  = round(budget * weight, 2)
        shares = int(alloc // price) if price and price > 0 else 0
        plan.append({
            "ticker":         ticker,
            "weight":         round(weight, 4),
            "allocation_usd": alloc,
            "price":          price,
            "shares":         shares,
        })

    spent    = sum(p["allocation_usd"] for p in plan)
    leftover = round(budget - spent, 2)
    plan.append({"ticker": "CASH", "weight": 0, "allocation_usd": leftover, "price": None, "shares": 0})
    return plan


#  OpenAI function-calling schema 
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "screen_stocks",
            "description": "Get a shortlist of candidate stock tickers based on the user's investment strategy (conservative, moderate, or aggressive) and optional sector preference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "aggressiveness": {
                        "type": "string",
                        "enum": ["conservative", "moderate", "aggressive"],
                        "description": "User's selected risk strategy."
                    },
                    "sectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional sector filter (e.g. ['Technology', 'Healthcare'])."
                    }
                },
                "required": ["aggressiveness"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_indicators",
            "description": "Compute RSI, MACD signal, 30-day volatility, price momentum slope, and 3/12-month price forecasts for a stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
                    "lookback_days": {"type": "integer", "default": 120}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_sentiment",
            "description": "Fetch recent financial news for a ticker and return the average sentiment score and top headlines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "days": {"type": "integer", "default": 30}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": "Retrieve company fundamental data: P/E ratio, EPS, revenue growth, debt-to-equity, free cash flow, market cap, sector, and industry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_purchase_plan",
            "description": "Allocate the user's budget across selected tickers with given weights. Returns dollar allocation and estimated share count per stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {"type": "number", "description": "Total dollars to invest."},
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "weights": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Fractional weights per ticker, must sum to 1.0."
                    }
                },
                "required": ["budget", "tickers", "weights"]
            }
        }
    }
]

#  Tool dispatcher 
_TOOL_MAP = {
    "screen_stocks":            lambda a: tool_screen_stocks(**a),
    "get_technical_indicators": lambda a: tool_technical_indicators(**a),
    "get_news_sentiment":       lambda a: tool_news_sentiment(**a),
    "get_fundamentals":         lambda a: tool_fundamentals(**a),
    "create_purchase_plan":     lambda a: tool_purchase_plan(**a),
}

def dispatch_tool(name: str, args: dict) -> str:
    """Call the named tool and return a JSON string of the result."""
    fn = _TOOL_MAP.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return json.dumps(fn(args), default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


#  System prompt 
def build_system_prompt(profile: dict) -> str:
    """
    Build the agent system prompt from the user profile.
    Hyunju can customize the wording and instructions in Notebook 2.
    """
    return f"""You are PaySprint, a professional AI investment research assistant for retail investors.
Your job is to research stocks and produce a clear, personalized investment plan.

User Profile:
  - Name:           {profile.get('name', 'Investor')}
  - Budget:         ${profile.get('budget', 0):,.2f}
  - Strategy:       {profile.get('aggressiveness', 'moderate')}
  - Horizon:        {profile.get('horizon_months', 12)} months
  - Current stocks: {json.dumps(profile.get('current_holdings', {}))}
  - Preferred sectors: {json.dumps(profile.get('preferred_sectors', []))}

You MUST follow this workflow exactly:
1. Call screen_stocks to get candidate tickers for the user's strategy.
2. For each candidate ticker, call get_technical_indicators, get_news_sentiment, and get_fundamentals.
3. Select the top 3-5 stocks based on momentum, sentiment, and fundamentals.
4. Call create_purchase_plan with budget=${profile.get('budget', 0):,.2f}, the selected tickers, and weights that match the strategy.
5. Write the final investment report with:
   - A 2-3 sentence summary for each selected stock
   - Why you selected it (data-driven reasoning)
   - A formatted purchase plan table
   - MANDATORY RISK DISCLOSURE at the end

Strategy-weight guidelines:
  - conservative: spread equally or weight toward lower-volatility picks
  - moderate: weight slightly toward highest momentum
  - aggressive: weight heavily toward top momentum picks

IMPORTANT: If the user asks anything NOT related to stock research or investment planning
(e.g. cooking, geography, coding), politely decline and say:
"I'm specialized in investment research. I can't help with that, but I'm happy to research
stocks or build an investment plan for you."
"""


#  ReAct orchestrator 
def run_agent(
    profile: dict,
    model: str = None,
    max_turns: int = 12,
    verbose: bool = True,
) -> dict:
    """
    Core ReAct agent loop.

    Alternates between:
      Reason  -> LLM decides what to do next
      Act     -> Call the appropriate tool
      Observe -> Append tool result to message history
      Repeat  -> Until LLM produces the final report (no more tool calls)

    Args:
        profile:   User investment profile dict
        model:     LLM model name (defaults to MODEL_REASONING)
        max_turns: Maximum reasoning steps before forcing a final answer
        verbose:   Print each tool call and its result

    Returns dict with:
        report    - final report text
        model     - model used
        turns     - number of reasoning steps taken
        messages  - full conversation history
        tool_calls - list of {turn, tool, args, result} for tracing
        usage     - cumulative token counts
    """
    client    = _get_client()
    model     = model or MODEL_REASONING
    messages  = [
        {"role": "system", "content": build_system_prompt(profile)},
        {"role": "user",   "content": "Please research stocks and create my personalized investment plan."},
    ]
    tool_log   = []
    total_in   = 0
    total_out  = 0

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=AGENT_TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )
        msg         = response.choices[0].message
        total_in   += response.usage.prompt_tokens
        total_out  += response.usage.completion_tokens

        # No more tool calls -> LLM is done, return the final report
        if not msg.tool_calls:
            return {
                "report":     msg.content or "",
                "model":      model,
                "turns":      turn + 1,
                "messages":   messages,
                "tool_calls": tool_log,
                "usage":      {"prompt_tokens": total_in, "completion_tokens": total_out,
                               "total_tokens": total_in + total_out},
            }

        # Append assistant message with tool calls
        messages.append({
            "role":       "assistant",
            "content":    msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        })

        # Execute each tool call and feed result back
        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            result = dispatch_tool(tc.function.name, args)
            if verbose:
                print(f"  [Turn {turn+1}]  {tc.function.name}({json.dumps(args)[:60]}...)")
                print(f"             -> {result[:100]}...")

            tool_log.append({
                "turn":   turn + 1,
                "tool":   tc.function.name,
                "args":   args,
                "result": json.loads(result),
            })
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    # Exceeded max_turns - ask for final answer now
    messages.append({"role": "user", "content": "Please write the final investment report now based on everything you have gathered."})
    final = client.chat.completions.create(model=model, messages=messages, temperature=0.3)
    total_in  += final.usage.prompt_tokens
    total_out += final.usage.completion_tokens

    return {
        "report":     final.choices[0].message.content or "",
        "model":      model,
        "turns":      max_turns,
        "messages":   messages,
        "tool_calls": tool_log,
        "usage":      {"prompt_tokens": total_in, "completion_tokens": total_out,
                       "total_tokens": total_in + total_out},
    }


def test_rejection(user_message: str, model: str = None) -> dict:
    """
    Send a single off-topic message to the agent and confirm it rejects it
    without calling any tools.

    Returns:
        {user_message, response, tool_calls_made (bool)}
    """
    client  = _get_client()
    model   = model or MODEL_REASONING
    profile = {"name": "Test", "budget": 1000, "aggressiveness": "moderate",
                "horizon_months": 6, "current_holdings": {}, "preferred_sectors": []}
    messages = [
        {"role": "system", "content": build_system_prompt(profile)},
        {"role": "user",   "content": user_message},
    ]
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=AGENT_TOOLS,
        tool_choice="auto", temperature=0.3,
    )
    msg = resp.choices[0].message
    return {
        "user_message":    user_message,
        "response":        msg.content or "",
        "tool_calls_made": bool(msg.tool_calls),
    }


# =============================================================================
# EVALUATION LAYER
# Used by: traces_evaluation.ipynb  (Quang - PM)
# =============================================================================

#  LLM judge 
_JUDGE_PROMPT = """You are an expert evaluator for AI investment tools.
Score the report below on each dimension from 1 (very poor) to 5 (excellent).

User profile:
  - Budget: ${budget:,.2f}
  - Strategy: {aggressiveness}
  - Horizon: {horizon_months} months

Report to evaluate:
---
{report}
---

Scoring dimensions:
1. reasoning_quality  - Is the analysis logical and grounded in actual data from the tools?
2. risk_alignment     - Do the recommended stocks match the user's stated strategy?
3. clarity            - Would a retail investor with no finance background understand this?
4. plan_accuracy      - Are the dollar allocations and share counts mathematically correct?

Return ONLY valid JSON with this exact structure:
{{
  "reasoning_quality": <integer 1-5>,
  "risk_alignment":    <integer 1-5>,
  "clarity":           <integer 1-5>,
  "plan_accuracy":     <integer 1-5>,
  "overall":           <float, average of the 4 scores>,
  "strengths":         "<one sentence about what the report did well>",
  "weaknesses":        "<one sentence about what the report could improve>"
}}"""


def llm_judge(report: str, profile: dict, model: str = None) -> dict:
    """
    Evaluate a report with an LLM judge using a 4-dimension rubric.
    Uses MODEL_SUMMARY (cheaper) because evaluation prompts are simple.

    Returns a dict with scores and qualitative feedback.
    Returns {'error': msg} on failure.
    """
    client = _get_client()
    model  = model or MODEL_SUMMARY
    prompt = _JUDGE_PROMPT.format(
        budget=profile.get("budget", 0),
        aggressiveness=profile.get("aggressiveness", "moderate"),
        horizon_months=profile.get("horizon_months", 12),
        report=report[:3500],
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        return {"error": str(e)}


#  Trace persistence 
def save_trace(trace_data: dict, trace_id: int | str, traces_dir: str = TRACES_DIR):
    """Save a trace dict to a JSON file in traces_dir."""
    Path(traces_dir).mkdir(parents=True, exist_ok=True)
    path = Path(traces_dir) / f"trace_{trace_id}.json"
    with open(path, "w") as f:
        json.dump(trace_data, f, indent=2, default=str)
    print(f"[trace] Saved -> {path}")
    return str(path)


def load_trace(trace_id: int | str, traces_dir: str = TRACES_DIR) -> dict:
    """Load a previously saved trace from disk."""
    path = Path(traces_dir) / f"trace_{trace_id}.json"
    with open(path) as f:
        return json.load(f)


def print_trace_summary(result: dict):
    """Print a compact summary of an agent run for quick inspection."""
    print(f"\n{'='*60}")
    print(f"Model : {result.get('model')}")
    print(f"Turns : {result.get('turns')}")
    print(f"Tokens: {result.get('usage', {}).get('total_tokens', 'N/A')}")
    print(f"\nTools called:")
    for tc in result.get("tool_calls", []):
        print(f"  Turn {tc['turn']} -> {tc['tool']}")
    print(f"\nReport preview (first 400 chars):")
    print(result.get("report", "")[:400])
    print("="*60)


#  Cost calculator 
MODEL_PRICES = {
    "gpt-4o":      {"input": 2.50,  "output": 10.00},  # per 1M tokens
    "gpt-4o-mini": {"input": 0.15,  "output":  0.60},
}

def estimate_cost(result: dict) -> dict:
    """
    Estimate API cost from a run result dict.
    Returns {model, input_tokens, output_tokens, cost_usd}.
    """
    model   = result.get("model", "gpt-4o")
    usage   = result.get("usage", {})
    inp     = usage.get("prompt_tokens", 0)
    out     = usage.get("completion_tokens", 0)
    prices  = MODEL_PRICES.get(model, MODEL_PRICES["gpt-4o"])
    cost    = (inp / 1_000_000 * prices["input"]) + (out / 1_000_000 * prices["output"])
    return {
        "model":         model,
        "input_tokens":  inp,
        "output_tokens": out,
        "cost_usd":      round(cost, 6),
    }


#  Backtesting 
def backtest(tickers: list, report_date_str: str, horizon_days: int = 63,
             benchmark: str = "SPY") -> pd.DataFrame:
    """
    Compare historical returns of tickers against a benchmark
    starting from report_date_str over horizon_days trading days.

    Uses actual price data so results are real (not simulated).

    Args:
        tickers:          List of stock tickers to evaluate
        report_date_str:  Date recommendations were made ('YYYY-MM-DD')
        horizon_days:     Number of trading days to measure return over
        benchmark:        Benchmark ticker (default 'SPY' for S&P 500)
    Returns:
        DataFrame with: ticker, entry_price, exit_price, return_pct, benchmark_return_pct, alpha_pct
    """
    all_tickers = tickers + [benchmark]
    ref_date    = pd.Timestamp(report_date_str)
    rows = []

    for ticker in all_tickers:
        prices = fetch_price_history(ticker, lookback_days=horizon_days + 60)
        if prices.empty:
            rows.append({"ticker": ticker, "error": "no data"})
            continue
        future = prices[prices.index >= ref_date]
        if len(future) < 2:
            rows.append({"ticker": ticker, "error": "not enough data after report date"})
            continue
        entry = float(future.iloc[0])
        exit_ = float(future.iloc[min(horizon_days, len(future) - 1)])
        ret   = (exit_ - entry) / entry * 100
        rows.append({"ticker": ticker, "entry_price": round(entry, 2),
                     "exit_price": round(exit_, 2), "return_pct": round(ret, 2)})

    df = pd.DataFrame(rows)
    if benchmark in df["ticker"].values:
        bm_ret = df.loc[df["ticker"] == benchmark, "return_pct"].values[0]
        df["benchmark_return_pct"] = bm_ret
        df["alpha_pct"] = df["return_pct"] - bm_ret
    return df


#  Consistency test 
def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets. Returns 1.0 if both empty."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def extract_tickers_from_report(report: str) -> set:
    """
    Extract ticker symbols mentioned in a report.
    Uses the same regex patterns as news parsing.
    Filters common English stop-words to reduce false positives.
    """
    STOP = {"I", "A", "AI", "TO", "OR", "THE", "AND", "FOR", "BUY", "SELL",
            "RSI", "EPS", "PE", "ROI", "NOT", "IS", "US", "WITH", "IN", "AT",
            "BE", "IT", "NO", "ON", "DO", "IF", "SO", "WE", "MY", "BY"}
    found = set(re.findall(r'\b([A-Z]{2,5})\b', report))
    return found - STOP


def consistency_test(profile: dict, model: str = None, n_runs: int = 3, verbose: bool = True) -> dict:
    """
    Run the agent n_runs times with the same profile and measure
    how consistent the recommended tickers are across runs.

    Consistency score: average pairwise Jaccard similarity of ticker sets.
    > 0.7 = consistent, 0.4-0.7 = moderate, < 0.4 = inconsistent.
    """
    model   = model or MODEL_REASONING
    reports = []
    for i in range(n_runs):
        if verbose:
            print(f"\n[consistency] Run {i+1}/{n_runs} ...")
        result = run_agent(profile, model=model, verbose=False)
        reports.append(result["report"])
        time.sleep(1)  # avoid rate limits

    ticker_sets = [extract_tickers_from_report(r) for r in reports]
    from itertools import combinations
    pairs  = list(combinations(range(n_runs), 2))
    scores = [jaccard(ticker_sets[i], ticker_sets[j]) for i, j in pairs]
    avg    = round(sum(scores) / len(scores), 3) if scores else 0.0

    label = "CONSISTENT" if avg > 0.7 else ("MODERATE" if avg > 0.4 else "INCONSISTENT")
    return {
        "n_runs":           n_runs,
        "model":            model,
        "ticker_sets":      [sorted(s) for s in ticker_sets],
        "pairwise_jaccard": [round(s, 3) for s in scores],
        "avg_jaccard":      avg,
        "verdict":          label,
    }


# =============================================================================
# QUICK TEST  (python paysprint_agent.py)
# =============================================================================

if __name__ == "__main__":
    import sys

    # Check for API key before doing anything
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        print("Set it with:  $env:OPENAI_API_KEY = 'sk-...'  (PowerShell)")
        print("          or: export OPENAI_API_KEY='sk-...'   (bash)")
        sys.exit(1)

    test_profile = {
        "name":             "Test Investor",
        "budget":           10_000,
        "aggressiveness":   "moderate",
        "horizon_months":   12,
        "current_holdings": {},
        "preferred_sectors": [],
    }

    print("PaySprint Agent - Quick Test Run")
    print(f"Profile: {test_profile['name']} | ${test_profile['budget']:,} | {test_profile['aggressiveness']}")
    print("-" * 60)

    result = run_agent(test_profile, verbose=True)
    print_trace_summary(result)

    cost = estimate_cost(result)
    print(f"\nEstimated cost: ${cost['cost_usd']:.4f}  "
          f"({cost['input_tokens']} in / {cost['output_tokens']} out tokens)")
