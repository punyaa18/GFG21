# Netflix EDA Project 🎬

# ==========================
# Step 1: Setup
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Plot style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (10,6)

# Pandas display settings
pd.set_option("display.max_columns", None)

# ==========================
# Step 2: Load Data
# ==========================
netflix_df = pd.read_csv('netflix_titles.csv')

print("Initial Dataset Info:")
print(netflix_df.info())
print(netflix_df.head(3))

# ==========================
# Step 3: Data Cleaning & Transformation
# ==========================
# Fill missing text fields
netflix_df['director'] = netflix_df['director'].fillna('Unknown')
netflix_df['cast'] = netflix_df['cast'].fillna('Unknown')

# Fill 'country' with mode (avoid error if column is all NaN)
if netflix_df['country'].notna().any():
    netflix_df['country'] = netflix_df['country'].fillna(netflix_df['country'].mode()[0])
else:
    netflix_df['country'] = "Unknown"

# Drop rows with missing critical values
netflix_df.dropna(subset=['rating','date_added'], inplace=True)

# Convert to datetime (no 'mixed' in some pandas versions → use errors='coerce')
netflix_df['date_added'] = pd.to_datetime(netflix_df['date_added'], errors='coerce')

# Drop rows where conversion failed
netflix_df.dropna(subset=['date_added'], inplace=True)

# Feature Engineering
netflix_df['year_added'] = netflix_df['date_added'].dt.year
netflix_df['month_added'] = netflix_df['date_added'].dt.month
netflix_df['age_on_netflix'] = netflix_df['year_added'] - netflix_df['release_year']

print("\nAfter Cleaning:")
print(netflix_df.info())
print("Remaining nulls:\n", netflix_df.isnull().sum())

# ==========================
# Step 4.1: Content Type Distribution
# ==========================
type_counts = netflix_df['type'].value_counts()

plt.pie(type_counts, labels=type_counts.index,
        autopct='%1.1f%%', startangle=140,
        colors=['#e60023','#221f1f'])
plt.title("Proportion of Movies vs TV Shows")
plt.show()

# ==========================
# Step 4.2: Content Over Time
# ==========================
content_over_time = netflix_df.groupby(['year_added','type']).size().unstack().fillna(0)

content_over_time.plot(kind='line', marker='o')
plt.title("Content Added Over Years (by Type)")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.grid(True)
plt.show()

# ==========================
# Step 4.3: Genre Analysis
# ==========================
genres = netflix_df.assign(genre=netflix_df['listed_in'].str.split(', ')).explode('genre')
top_genres = genres['genre'].value_counts().head(15)

sns.barplot(x=top_genres.values, y=top_genres.index, palette="mako")
plt.title("Top 15 Genres on Netflix")
plt.xlabel("Count")
plt.show()

# ==========================
# Step 4.4: Duration Analysis
# ==========================
movies_df = netflix_df[netflix_df['type']=="Movie"].copy()
tv_df = netflix_df[netflix_df['type']=="TV Show"].copy()

# Clean duration safely
movies_df['duration_min'] = (movies_df['duration']
                             .str.replace(" min","", regex=False)
                             .str.extract(r'(\d+)')
                             .astype(float))

tv_df['seasons'] = (tv_df['duration']
                    .str.replace(" Season","", regex=False)
                    .str.replace("s","", regex=False)
                    .str.extract(r'(\d+)')
                    .astype(float))

# Plot
fig,ax = plt.subplots(1,2,figsize=(16,6))
sns.histplot(movies_df['duration_min'].dropna(), bins=50, kde=True, ax=ax[0], color="skyblue")
ax[0].set_title("Movie Duration Distribution (minutes)")

sns.countplot(x=tv_df['seasons'].dropna(), ax=ax[1], palette="rocket")
ax[1].set_title("TV Show Season Distribution")
plt.show()

# ==========================
# Step 4.5: Country Analysis
# ==========================
countries = netflix_df.assign(country=netflix_df['country'].str.split(', ')).explode('country')
top_countries = countries['country'].value_counts().head(15)

sns.barplot(x=top_countries.values, y=top_countries.index, palette="viridis")
plt.title("Top 15 Content-Producing Countries")
plt.xlabel("Number of Titles")
plt.show()

# ==========================
# Step 4.6: Rating Distribution
# ==========================
sns.countplot(x="rating", data=netflix_df,
              order=netflix_df['rating'].value_counts().index,
              palette="crest")
plt.xticks(rotation=45)
plt.title("Distribution of Content Ratings")
plt.show()

# ==========================
# Step 5: Content Age on Netflix
# ==========================
content_age = netflix_df[netflix_df['age_on_netflix']>=0]

sns.histplot(content_age['age_on_netflix'], bins=40, kde=True)
plt.title("Distribution of Content Age on Netflix")
plt.xlabel("Years between Release & Added")
plt.show()

# ==========================
# Step 6: Multivariate Analysis
# ==========================
top5_genres = genres['genre'].value_counts().index[:5]
g_movies = genres[(genres['type']=="Movie") & (genres['genre'].isin(top5_genres))].copy()

g_movies['duration_min'] = (g_movies['duration']
                            .str.replace(" min","", regex=False)
                            .str.extract(r'(\d+)')
                            .astype(float))

sns.boxplot(x="genre", y="duration_min", data=g_movies, palette="pastel")
plt.title("Movie Duration by Top Genres")
plt.xticks(rotation=45)
plt.show()

# ==========================
# Step 7: Word Cloud
# ==========================
text = " ".join(netflix_df['description'].dropna())
wc = WordCloud(width=900, height=450, background_color="black").generate(text)

plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Descriptions", fontsize=18)
plt.show()

# ==========================
# Step 8: Summary (printed as text)
# ==========================
print("""
🔑 Key Insights:
1. Netflix library is ~70% Movies, ~30% TV Shows
2. Content growth peaked 2016–2019, slowed post-2020
3. Genres: International Movies, Dramas, Comedies dominate
4. US is top producer, India second; global content strategy strong
5. Ratings skewed to adults (TV-MA, TV-14)
6. Most movies ~90–120 min; TV shows mostly 1 season
7. Many titles added same year as release (Originals), but also older licensed content
8. Themes in descriptions: life, family, love, young, friends, discovery
""")
