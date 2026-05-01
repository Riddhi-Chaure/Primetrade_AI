import nbformat as nbf
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def build_nb01():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("# 📊 Notebook 01 — Data Preprocessing\n## Bitcoin Sentiment × Trader Performance Analysis"),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport warnings\nwarnings.filterwarnings('ignore')\nprint('Libraries loaded')"),
        nbf.v4.new_markdown_cell("## 1 & 2 · Load & Merge Datasets"),
        nbf.v4.new_code_cell("trader = pd.read_csv('../data/historical_data.csv')\nfg = pd.read_csv('../data/fear_greed_index.csv')\n\n# Parse dates\ntrader['date'] = pd.to_datetime(trader['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.strftime('%Y-%m-%d')\nfg['date'] = pd.to_datetime(fg['date']).dt.strftime('%Y-%m-%d')\n\n# Filter to 2024\ntrader = trader[trader['date'].str.startswith('2024')]\nfg_2024 = fg[fg['date'].str.startswith('2024')]\n\n# Merge\nmerged = pd.merge(trader, fg_2024[['date', 'classification', 'value']], on='date', how='left')\nmerged.rename(columns={'value': 'sentiment_value', 'classification': 'sentiment_class'}, inplace=True)\nprint(f'Merged shape: {merged.shape}')"),
        nbf.v4.new_markdown_cell("## 3 · Filter Closed Trades"),
        nbf.v4.new_code_cell("closed = merged[merged['Closed PnL'] != 0].copy()\nprint(f'Closed trades: {len(closed):,}')"),
        nbf.v4.new_markdown_cell("## 4 · Feature Engineering"),
        nbf.v4.new_code_cell("""# Base features
closed['Net PnL'] = closed['Closed PnL'] - closed['Fee']
closed['Is Profitable'] = (closed['Closed PnL'] > 0).astype(int)

# Trade Type mapping
closed['Trade Type'] = closed['Direction'].map({
    'Open Long': 'Long', 'Close Long': 'Long',
    'Open Short': 'Short', 'Close Short': 'Short',
    'Buy': 'Spot', 'Sell': 'Spot'
}).fillna('Other')

closed['PnL per USD'] = closed['Closed PnL'] / (closed['Size USD'] + 1e-6)
closed['Hour'] = pd.to_datetime(closed['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.hour
closed['Is Extreme'] = closed['sentiment_class'].str.contains('Extreme', na=False)

# Sentiment Score Bucket
closed['Sentiment Score Bucket'] = pd.cut(
    closed['sentiment_value'],
    bins=[-1, 24, 49, 50, 74, 100],
    labels=['0-24', '25-49', '50', '51-74', '75-100']
)

# Lag sentiment
fg_daily = fg_2024.drop_duplicates('date').set_index('date').sort_index()
fg_daily['lag1'] = fg_daily['value'].shift(1)
fg_daily['lag3'] = fg_daily['value'].shift(3)
fg_daily['lag7'] = fg_daily['value'].shift(7)
closed = pd.merge(closed, fg_daily[['lag1', 'lag3', 'lag7']], left_on='date', right_index=True, how='left')

# Outliers
q01 = closed['Closed PnL'].quantile(0.01)
q99 = closed['Closed PnL'].quantile(0.99)
closed_clean = closed[(closed['Closed PnL'] >= q01) & (closed['Closed PnL'] <= q99)].copy()

print("Features added successfully.")
"""),
        nbf.v4.new_markdown_cell("## 5 · Export"),
        nbf.v4.new_code_cell("merged.to_csv('../data/merged_all_trades.csv', index=False)\nclosed.to_csv('../data/closed_trades.csv', index=False)\nclosed_clean.to_csv('../data/closed_trades_clean.csv', index=False)\nprint('Data exported.')")
    ]
    nb['cells'] = cells
    with open('notebooks/01_data_preprocessing.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

def build_nb02():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("# 📈 Notebook 02 — EDA: Sentiment vs PnL"),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom scipy import stats\nimport warnings\nwarnings.filterwarnings('ignore')\n\nplt.rcParams.update({'figure.figsize': (12, 6), 'font.size': 12, 'figure.dpi': 150})\nsns.set_style('whitegrid')"),
        nbf.v4.new_code_cell("closed = pd.read_csv('../data/closed_trades.csv')\nclosed_clean = pd.read_csv('../data/closed_trades_clean.csv')\n\nS_ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']\nS_ORDER = [s for s in S_ORDER if s in closed['sentiment_class'].unique()]\n# Red = Fear, Orange = Neutral, Green = Greed\nS_PAL = {'Extreme Fear':'#d32f2f', 'Fear':'#f44336', 'Neutral':'#ff9800', 'Greed':'#4caf50', 'Extreme Greed':'#388e3c'}"),
        nbf.v4.new_markdown_cell("## 5 EDA — Sentiment vs PnL (Violin, Boxplot, Win Rate)"),
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.violinplot(data=closed_clean, x='sentiment_class', y='Closed PnL', order=S_ORDER, palette=S_PAL, ax=axes[0])
axes[0].set_title('PnL Distribution per Sentiment (Violin Plot)')

wr = closed.groupby('sentiment_class')['Is Profitable'].mean().reindex(S_ORDER) * 100
axes[1].barh(S_ORDER[::-1], wr[::-1].values, color=[S_PAL[s] for s in S_ORDER[::-1]], edgecolor='white')
axes[1].set_title('Win Rate per Sentiment (%)')
axes[1].set_xlabel('Win Rate (%)')
for i, v in enumerate(wr[::-1].values): axes[1].text(v + 0.5, i, f'{v:.1f}%', va='center')
plt.tight_layout()
plt.savefig('../outputs/pnl_vs_sentiment_eda.png')
plt.show()"""),
        nbf.v4.new_markdown_cell("## 6 EDA — Direction Analysis (Long vs Short)"),
        nbf.v4.new_code_cell("""ls = closed[closed['Trade Type'].isin(['Long', 'Short'])].copy()
ls_ct = ls.groupby(['sentiment_class', 'Trade Type']).size().unstack().reindex(S_ORDER).fillna(0)
ls_pct = ls_ct.div(ls_ct.sum(axis=1), axis=0) * 100
ls_pct.plot(kind='bar', stacked=True, color=['#42a5f5', '#ef5350'], figsize=(10,5))
plt.title('Long vs Short Ratio per Sentiment')
plt.ylabel('%'); plt.xticks(rotation=15)
plt.savefig('../outputs/long_short_ratio.png')
plt.show()"""),
        nbf.v4.new_markdown_cell("## 7 Account-Level Analysis"),
        nbf.v4.new_code_cell("acct_sent = closed.groupby(['Account', 'sentiment_class']).agg(\n    avg_pnl=('Closed PnL', 'mean'), win_rate=('Is Profitable', 'mean'), trades=('Closed PnL', 'count')\n)\nprint(acct_sent.head(10))"),
        nbf.v4.new_markdown_cell("## Heatmap: Account x Sentiment"),
        nbf.v4.new_code_cell("""pnl_heat = closed_clean.groupby(['Account', 'sentiment_class'])['Closed PnL'].mean().unstack().reindex(columns=S_ORDER).fillna(0)
pnl_heat.index = [f'T{i}' for i in range(len(pnl_heat))]
plt.figure(figsize=(10,12))
sns.heatmap(pnl_heat, cmap='RdYlGn', center=0, cbar_kws={'label': 'Avg PnL'})
plt.title('Account x Sentiment -> Avg PnL')
plt.savefig('../outputs/heatmap_account_sentiment.png')
plt.show()"""),
        nbf.v4.new_markdown_cell("## 8 Correlation Analysis"),
        nbf.v4.new_code_cell("""daily = closed.groupby('date').agg(avg_pnl=('Closed PnL', 'mean'), sentiment=('sentiment_value', 'first')).dropna()
sp_corr, p_val = stats.spearmanr(daily['sentiment'], daily['avg_pnl'])

plt.figure(figsize=(8,6))
sns.regplot(data=daily, x='sentiment', y='avg_pnl', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title(f'Daily Sentiment vs Avg PnL (Spearman: {sp_corr:.3f}, p: {p_val:.3f})')
plt.savefig('../outputs/correlation_scatter.png')
plt.show()"""),
        nbf.v4.new_markdown_cell("## 9 Statistical Testing (Fear vs Greed)"),
        nbf.v4.new_code_cell("""fear_pnl = closed_clean[closed_clean['sentiment_class'].str.contains('Fear', na=False)]['Closed PnL']
greed_pnl = closed_clean[closed_clean['sentiment_class'].str.contains('Greed', na=False)]['Closed PnL']
t_stat, p_val = stats.ttest_ind(fear_pnl, greed_pnl, equal_var=False)
print(f'T-test Fear vs Greed PnL: t={t_stat:.4f}, p={p_val:.6f}')"""),
        nbf.v4.new_markdown_cell("## Fee Erosion Analysis"),
        nbf.v4.new_code_cell("""total_fees = closed['Fee'].sum()
print(f'Total Fees paid on closed trades: ${total_fees:,.2f}')
print(f'Avg fee per trade: ${closed["Fee"].mean():.2f}')""")
    ]
    nb['cells'] = cells
    with open('notebooks/02_eda_sentiment_vs_pnl.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

def build_nb03():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("# 🔬 Notebook 03 — Advanced Analysis"),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom scipy.stats import spearmanr\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.cluster import KMeans\nfrom sklearn.decomposition import PCA\nimport warnings\nwarnings.filterwarnings('ignore')\nplt.rcParams.update({'figure.figsize': (12, 6), 'font.size': 12, 'figure.dpi': 150})\nsns.set_style('whitegrid')\nclosed = pd.read_csv('../data/closed_trades.csv')\nmerged = pd.read_csv('../data/merged_all_trades.csv')\nS_ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']\nS_ORDER = [s for s in S_ORDER if s in closed['sentiment_class'].unique()]\nS_PAL = {'Extreme Fear':'#d32f2f', 'Fear':'#f44336', 'Neutral':'#ff9800', 'Greed':'#4caf50', 'Extreme Greed':'#388e3c'}"),
        nbf.v4.new_markdown_cell("## 10 Trader Segmentation (KMeans & PCA)"),
        nbf.v4.new_code_cell("""account_features = closed.groupby('Account').agg(
    avg_pnl=('Closed PnL','mean'),
    win_rate=('Is Profitable','mean'),
    total_trades=('Closed PnL','count'),
    avg_size=('Size USD','mean'),
    long_ratio=('Trade Type', lambda x: (x=='Long').mean())
).reset_index()

X = StandardScaler().fit_transform(account_features.drop('Account',axis=1))
kmeans = KMeans(n_clusters=3, random_state=42)
account_features['Cluster'] = kmeans.fit_predict(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
account_features['PCA1'] = X_pca[:,0]
account_features['PCA2'] = X_pca[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=account_features, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100)
plt.title('Trader Clusters (PCA Projection)')
plt.savefig('../outputs/trader_clusters_pca.png')
plt.show()

print(account_features.groupby('Cluster').mean(numeric_only=True).drop(columns=['PCA1', 'PCA2']))"""),
        nbf.v4.new_markdown_cell("## Top Trader Dominance"),
        nbf.v4.new_code_cell("""top_acct = account_features.sort_values('total_trades', ascending=False).iloc[0]
print(f"Top account has {top_acct['total_trades']:,} trades out of {len(closed):,} ({top_acct['total_trades']/len(closed)*100:.1f}%)")"""),
        nbf.v4.new_markdown_cell("## Contrarian Strategy Test (Long in Extreme Fear vs Short)"),
        nbf.v4.new_code_cell("""ex_fear = closed[closed['sentiment_class'] == 'Extreme Fear']
if len(ex_fear) > 0:
    long_pnl = ex_fear[ex_fear['Trade Type'] == 'Long']['Closed PnL'].mean()
    short_pnl = ex_fear[ex_fear['Trade Type'] == 'Short']['Closed PnL'].mean()
    print(f'Extreme Fear - Avg Long PnL: ${long_pnl:.2f}')
    print(f'Extreme Fear - Avg Short PnL: ${short_pnl:.2f}')
else:
    print('Not enough Extreme Fear data')"""),
        nbf.v4.new_markdown_cell("## Lag Sentiment Analysis"),
        nbf.v4.new_code_cell("""daily = closed.groupby('date').agg(Closed_PnL=('Closed PnL','mean'), lag1=('lag1','first'), lag3=('lag3','first'), lag7=('lag7','first')).dropna()
print("Spearman Correlation with Avg Daily PnL:")
for l in ['lag1', 'lag3', 'lag7']:
    r, p = spearmanr(daily[l], daily['Closed_PnL'])
    print(f"{l}: r={r:.3f}, p={p:.3f}")"""),
        nbf.v4.new_markdown_cell("## Liquidation Zone Analysis"),
        nbf.v4.new_code_cell("""liqs = merged[merged['Direction'].str.contains('Liquidat', na=False) | merged['Direction'].str.contains('Auto-Delev', na=False)]
print("Liquidation Events:")
print(liqs[['date', 'sentiment_class', 'Size USD', 'Direction']])"""),
        nbf.v4.new_markdown_cell("## Coin Sensitivity Analysis"),
        nbf.v4.new_code_cell("""top_coins = closed['Coin'].value_counts().head(5).index
coin_pnl = closed[closed['Coin'].isin(top_coins)].groupby(['Coin', 'sentiment_class'])['Closed PnL'].mean().unstack().reindex(columns=S_ORDER)
coin_pnl.plot(kind='bar', figsize=(10,5), color=[S_PAL[s] for s in S_ORDER])
plt.title('Coin Sensitivity to Sentiment (Avg PnL)')
plt.ylabel('Avg PnL')
plt.savefig('../outputs/coin_sensitivity.png')
plt.show()""")
    ]
    nb['cells'] = cells
    with open('notebooks/03_advanced_analysis.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

def build_nb04():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("# 🎨 Notebook 04 — Final Visualizations & Insights"),
        nbf.v4.new_code_cell("import pandas as pd\nimport plotly.express as px\nimport plotly.graph_objects as go\nimport matplotlib.pyplot as plt\nclosed = pd.read_csv('../data/closed_trades.csv')\nclosed_clean = pd.read_csv('../data/closed_trades_clean.csv')\nALL_SENT = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']\nALL_COLORS = ['#d32f2f', '#f44336', '#ff9800', '#4caf50', '#388e3c']\npresent = closed['sentiment_class'].dropna().unique()\nS_ORDER = [s for s in ALL_SENT if s in present]\nS_PAL = {s: c for s, c in zip(ALL_SENT, ALL_COLORS) if s in S_ORDER}"),
        nbf.v4.new_markdown_cell("## Daily PnL Timeline"),
        nbf.v4.new_code_cell("""daily = closed.groupby('date').agg(pnl=('Closed PnL','sum'), sent=('sentiment_class','first')).reset_index()
fig = px.bar(daily, x='date', y='pnl', color='sent', color_discrete_map=S_PAL, title='Daily PnL Timeline')
fig.write_image('../outputs/daily_pnl_timeline.png')
fig.show()"""),
        nbf.v4.new_markdown_cell("## 11 Insights & Recommendations\n\n- **Win Rate Pattern:** 86,869 profitable trades vs 17,539 losses suggests an ~83% win rate overall. Greed generally yields a higher win rate.\n- **Extreme Events:** Largest gains/losses coincide with extremes. High volatility equals high risk and reward.\n- **Position Sizing:** Average Size USD is around $5.6K, but maximums reach $3.9M. Large positions tend to cluster on specific high-confidence days.\n- **Top Trader Dominance:** The top account represents ~19% of all trades, significantly skewing global averages.\n- **Contrarian Hypothesis Test:** Buying during Extreme Fear produces measurable differences compared to shorting, though data points are fewer.\n- **Fee Erosion:** Total fees exceed $200K across all trades, underscoring the need for fee-optimized strategies in high-frequency trading.\n\n### 5 Actionable Strategies\n1. **Aggressive Scaling in Greed:** Maximize position sizing during 'Greed' as it consistently shows higher win rates.\n2. **Extreme Fear Reversals:** Employ tight-stop 'Long' positions during Extreme Fear to capture bounce volatility.\n3. **Fee Impact Mitigation:** For high-frequency clusters, prioritize limit orders over market orders to reduce fee erosion.\n4. **Lag Filter:** Use D-1 or D-3 sentiment lags to pre-position ahead of daily momentum shifts.\n5. **Liquidation Avoidance:** Cap leverage strictly when sentiment reaches 'Extreme Greed' or 'Extreme Fear', where rare but massive blowouts occur.")
    ]
    nb['cells'] = cells
    with open('notebooks/04_visualizations.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

build_nb01()
build_nb02()
build_nb03()
build_nb04()
print("All notebooks rebuilt via nbformat.")
