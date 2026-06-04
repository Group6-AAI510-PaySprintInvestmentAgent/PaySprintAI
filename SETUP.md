# PaySprint Investment Agent - Setup Guide
**AAI-510 | Group 6**

---

## Quick Start (5 minutes)

### Step 1 - Install Python packages
Open a terminal in the project folder and run:
```
pip install -r requirements.txt
```

### Step 2 - Set your OpenAI API key
Create a file called `.env` in the project folder (same folder as this README).
Paste this inside it - replace the placeholder with your real key:
```
OPENAI_API_KEY=sk-your-actual-key-here
```
Get a key at: https://platform.openai.com/api-keys

>  Never share your .env file or commit it to GitHub. It's already in .gitignore.

### Step 3 - Run the notebooks in order
Open Jupyter Notebook or VS Code, then run:
1. `data_pipeline.ipynb`   -> Reema
2. `agent_definition.ipynb` -> Hyunju
3. `traces_evaluation.ipynb` -> Quang (after notebooks 1 & 2 are done)

---

## File Overview

| File | Who uses it | Purpose |
|------|-------------|---------|
| `paysprint_agent.py` | Everyone (do not edit unless needed) | Complete working backend |
| `data_pipeline.ipynb` | Reema | Data pipeline + customizations |
| `agent_definition.ipynb` | Hyunju | Agent demo + customizations |
| `traces_evaluation.ipynb` | Quang + all | 5 traces, evaluation, commentary |
| `requirements.txt` | Everyone | Python package list |
| `.env` | Everyone | Your API key (create this yourself) |
| `paysprint.db` | Auto-created | Stores user profiles and recommendations |
| `data/` | Auto-created | Saves CSVs and trace JSON files |

---

## What each teammate needs to do

### Reema (Data Engineer) - Notebook 1
**Time estimate: 3-4 hours**
1. Run the notebook top to bottom - verify all data loads correctly
2. Add more tickers to the screener (Task A in the notebook)
3. Add more trusted news sources (Task B)
4. Explore news sentiment for different stocks (Task C)
5. Write the data quality commentary cell (Task D)

### Hyunju (AI Engineer) - Notebook 2
**Time estimate: 3-4 hours**
1. Run the notebook top to bottom - verify the agent produces a report
2. Customize the system prompt (Task A)
3. Adjust the scoring weights (Task B)
4. Write the agent behavior commentary cell (Task C)

### Quang (PM) - Notebook 3
**Time estimate: 4-5 hours (most time is waiting for agent runs)**
1. Run all 5 traces (takes ~15-20 minutes total)
2. Review the LLM judge scores
3. Fill in all  commentary cells
4. Run the backtesting and consistency tests
5. Compile results for the video presentation

---

## Troubleshooting

**"No module named 'gnews'"**
-> Run `pip install gnews` in the terminal

**"No news returned"**
-> GNews has rate limits. Wait 5 minutes and try again.

**"OPENAI_API_KEY not set"**
-> Make sure your .env file is in the project folder and has the key.

**yfinance returns no data for a ticker**
-> The ticker may be delisted or spelled wrong. Try searching it on Yahoo Finance first.

**Agent runs take too long**
-> The agent makes many API calls. If it's too slow, reduce `max_turns` to 8 in `run_agent()`.
