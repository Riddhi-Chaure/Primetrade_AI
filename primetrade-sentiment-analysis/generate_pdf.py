from fpdf import FPDF
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Bitcoin Sentiment x Trader Performance Analysis', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, 'PrimeTrade.ai Data Science Internship Assignment', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_report():
    pdf = PDF()
    pdf.add_page()
    
    # Executive Summary
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. Executive Summary', 0, 1)
    pdf.set_font('Arial', '', 10)
    summary = (
        "This report analyzes 211,224 Hyperliquid trades from 32 accounts across 2024, "
        "mapped to the daily Bitcoin Fear & Greed Index. The objective was to uncover "
        "how market sentiment shapes trader behavior, profitability, and risk appetite.\n\n"
        "Overall, traders achieved an impressive ~83.7% win rate. However, sentiment "
        "played a massive role in returns. While 'Extreme Greed' days saw the highest "
        "average win rates (89.0%), 'Extreme Fear' days were exceptionally unfavorable (19.4%)."
    )
    pdf.multi_cell(0, 6, summary)
    pdf.ln(5)
    
    # Main Insights
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. Five Key Actionable Insights & Recommendations', 0, 1)
    pdf.set_font('Arial', '', 10)
    insights = [
        "1. Win Rate Pattern Analysis: 'Extreme Greed' and 'Greed' categories consistently output the highest win rates (~89% and ~87%). Recommendation: Traders should heavily size up long positions during 'Greed' days as trend momentum is highly supportive.",
        "2. Extreme Fear Risk: 'Extreme Fear' yielded extremely poor outcomes with high variance. Recommendation: Traders should implement strict stop-losses or avoid trading altogether during Extreme Fear, as directional edge breaks down.",
        "3. Contrarian Strategy Fails: The hypothesis that opening Longs during Extreme Fear is a profitable contrarian strategy proved false in this dataset. Contrarian Longs suffered massive losses. Recommendation: Do not catch falling knives; wait for sentiment to revert to 'Neutral' before fading the fear.",
        "4. Fee Erosion Impact: Across 104k closed trades, cumulative fees amounted to a staggering sum, often erasing gross profitability for high-frequency traders. Recommendation: Accounts in Cluster 0 (High Frequency) must optimize their maker/taker execution or reduce churn.",
        "5. Trader Archetypes: K-Means clustering successfully separated 32 accounts into 3 archetypes: Conservative (low trade count, high win rate), Aggressive (high size, volatile PnL), and High-Frequency (massive churn, high fee bleed). Recommendation: Firms should allocate the most capital to 'Conservative' accounts demonstrating steady edge."
    ]
    for i in insights:
        pdf.multi_cell(0, 6, i)
        pdf.ln(3)

    # Charts
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '3. Data Visualizations', 0, 1)
    
    # 1. PnL Distribution
    if os.path.exists('outputs/pnl_by_sentiment.png'):
        pdf.image('outputs/pnl_by_sentiment.png', x=10, w=190)
        pdf.ln(5)
        
    if os.path.exists('outputs/winrate_by_sentiment.png'):
        pdf.image('outputs/winrate_by_sentiment.png', x=10, w=190)
        pdf.ln(5)
        
    pdf.add_page()
    if os.path.exists('outputs/trader_clusters_pca.png'):
        pdf.image('outputs/trader_clusters_pca.png', x=10, w=190)
        pdf.ln(5)
        
    if os.path.exists('outputs/long_short_ratio.png'):
        pdf.image('outputs/long_short_ratio.png', x=10, w=190)
        pdf.ln(5)
        
    if os.path.exists('outputs/daily_pnl_timeline.png'):
        pdf.image('outputs/daily_pnl_timeline.png', x=10, w=190)
        pdf.ln(5)
        
    pdf.output('report/final_analysis_report.pdf', 'F')
    print("Report generated successfully.")

create_report()
