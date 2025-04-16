# 3_eda_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("cleaned_loksabha_data.csv")

# Set visual style
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# 1. Top 10 parties by total wins
top_parties = df['party'].value_counts().head(10)
top_parties.plot(kind='bar', color='skyblue')
plt.title("Top 10 Political Parties by Total Wins")
plt.xlabel("Party")
plt.ylabel("Number of Wins")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charts/top_parties_wins.png")
plt.show()

# 2. Turnout trend over the years
plt.figure(figsize=(10, 6))
sns.lineplot(x='year', y='Turnout', data=df, marker='o')
plt.title("Voter Turnout Over the Years")
plt.xlabel("Election Year")
plt.ylabel("Turnout Percentage")
plt.grid(True)
plt.savefig("charts/turnout_trend.png")
plt.show()

# 3. Margin of Victory Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['margin'], bins=30, kde=True, color='orange')
plt.title("Distribution of Victory Margins")
plt.xlabel("Margin of Victory")
plt.ylabel("Frequency")
plt.savefig("charts/margin_distribution.png")
plt.show()

# 4. State-wise total elections
plt.figure(figsize=(12, 6))
state_counts = df['state'].value_counts().head(10)
sns.barplot(x=state_counts.values, y=state_counts.index, palette="mako")
plt.title("Top 10 States by Election Participation")
plt.xlabel("Number of Elections")
plt.ylabel("State")
plt.savefig("charts/statewise_elections.png")
plt.show()
