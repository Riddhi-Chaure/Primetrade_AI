Bitcoin Sentiment × Trader Performance Analysis
PrimeTrade.ai Data Science Internship Assignment

![Banner](outputs/pnl_by_sentiment.png)

Overview
Analysis of 211,224 Hyperliquid trades from 32 accounts across 2024,
mapped to daily Bitcoin Fear & Greed Index to uncover how market
sentiment shapes trader behavior and profitability.

Key Findings
Overall Profitability:Traders achieved an impressive ~83.7% win rate overall across 104,408 closed trades.
Sentiment Win Rate:'Extreme Greed' conditions produced the highest win rate (89.0%), while 'Extreme Fear' was highly unfavorable (19.4%).
Trader Archetypes:Identified three distinct trader clusters: high-frequency/low-size, risk-takers (high volatility), and consistent performers.

Dataset
`historical_data.csv`: 211,224 trades, 16 columns
`fear_greed_index.csv`: Daily sentiment, 2018–2025

Tools Used
Python · Pandas · Seaborn · Plotly · Scikit-learn

How to Run
1. Clone repo
2. `pip install -r requirements.txt`
3. Run notebooks in order (01 → 04)