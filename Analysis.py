import pandas as pd
import numpy as np
import re
import ast
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import networkx as nx
from wordcloud import WordCloud
from pandas.plotting import parallel_coordinates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from ast import literal_eval

# Data Loading and Preprocessing
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Data preprocessing
    numeric_cols = ['rating', 'votes', 'runtime']
    df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.mean()))
    df['year'] = df['year'].dropna()
    
    string_cols = ['genres', 'directors', 'cast', 'plot', 'countries', 'languages']
    df = df.dropna(subset=string_cols)
    for col in string_cols:
        df = df[df[col].str.strip() != '']

    multi_feature_cols = ['genres', 'directors', 'cast', 'countries', 'languages']
    for col in multi_feature_cols:
        df[col] = df[col].apply(lambda x: [item.strip() for item in str(x).split(',')])

    # Standardize age_group
    df['age_group'] = df['age_group'].str.upper().str.strip()
    replacements = {
        r'\+?18': 'NC-17',
        r'NC/?17': 'NC-17',
        r'\bM(/PG)?\b': 'PG',
        r'\bGP\b': 'PG',
        r'\bPG[- ]?13\b': 'PG-13',
        r'\bNOT RATED\b|\bUNRATED\b': 'UNRATED',
        r'\bPASSED\b|\bAPPROVED\b': 'G',
        r'\bTV[\s-]?G\b': 'TV-G',
        r'\bTV[\s-]?PG\b': 'TV-PG',
        r'\bTV[\s-]?14\b': 'TV-14',
        r'\bTV[\s-]?MA\b': 'TV-MA'
    }
    for pattern, replacement in replacements.items():
        df['age_group'] = df['age_group'].str.replace(pattern, replacement, regex=True)
    df['age_group'] = df['age_group'].str.strip()

    # Format runtime and votes
    df['runtime'] = pd.to_timedelta(df['runtime'], unit='m').astype(str).str.extract(r'(\d+:\d+:\d+)')[0]
    df['votes_numeric'] = df['votes']
    df['votes'] = df['votes'].apply(lambda v: f"{v/1_000_000:.1f}M" if v >= 1_000_000 else f"{v/1_000:.0f}K" if v >= 1_000 else str(int(v)))
    
    # Clean plot text
    df['plot'] = df['plot'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))

    # Convert list-like columns
    list_cols = ['genres', 'directors', 'cast', 'countries', 'languages']
    for col in list_cols:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # Runtime in minutes
    def time_to_minutes(t):
        try:
            h, m, s = map(int, str(t).split(":"))
            return h * 60 + m + s / 60
        except:
            return None
    df['runtime_minutes'] = df['runtime'].apply(time_to_minutes)
    
    # Decade column
    df['decade'] = (df['year'] // 10) * 10
    
    return df

# Helper functions
def flatten_list_column(column):
    return [item for sublist in column for item in sublist]

def top_avg_rating(df, col):
    items = []
    for entry in set(flatten_list_column(df[col])):
        subset = df[df[col].apply(lambda x: entry in x)]
        avg_rating = subset['rating'].mean()
        items.append((entry, avg_rating))
    return sorted(items, key=lambda x: x[1], reverse=True)[:5]

# Analysis Functions
def perform_analysis(filtered_df):
    print("\n=== BASIC STATISTICS ===")
    print(f"Total Movies: {len(filtered_df)}")
    print(f"Average Rating: {filtered_df['rating'].mean():.2f}")
    print(f"Earliest Year: {int(filtered_df['year'].min())}")
    
    top_movies = filtered_df.sort_values(by=['rating', 'votes_numeric'], ascending=False).head(1000)
    
    # People Analytics
    top_actors = Counter(flatten_list_column(top_movies['cast'])).most_common(10)
    top_directors = Counter(flatten_list_column(top_movies['directors'])).most_common(10)
    
    print("\n=== TOP ACTORS ===")
    for actor, count in top_actors:
        print(f"{actor}: {count} movies")
    
    print("\n=== TOP DIRECTORS ===")
    for director, count in top_directors:
        print(f"{director}: {count} movies")
    
    # Genre Analysis
    genre_counts = Counter(flatten_list_column(top_movies['genres']))
    genre_percentages = {genre: (count/sum(genre_counts.values()))*100 
                        for genre, count in genre_counts.items()}
    genre_percentages = dict(sorted(genre_percentages.items(), 
                                key=lambda x: x[1], reverse=True))
    
    print("\n=== GENRE DISTRIBUTION ===")
    for genre, percentage in list(genre_percentages.items())[:10]:
        print(f"{genre}: {percentage:.1f}%")
    
    # Geographic Analysis
    top_countries = Counter(flatten_list_column(top_movies['countries'])).most_common(10)
    top_languages = Counter(flatten_list_column(top_movies['languages'])).most_common(10)
    
    print("\n=== TOP COUNTRIES ===")
    for country, count in top_countries:
        print(f"{country}: {count} movies")
    
    print("\n=== TOP LANGUAGES ===")
    for language, count in top_languages:
        print(f"{language}: {count} movies")
    
    # Rating Analysis
    top_directors_by_rating = top_avg_rating(top_movies, 'directors')
    top_actors_by_rating = top_avg_rating(top_movies, 'cast')
    
    print("\n=== TOP DIRECTORS BY RATING ===")
    for director, rating in top_directors_by_rating:
        print(f"{director}: {rating:.2f}")
    
    print("\n=== TOP ACTORS BY RATING ===")
    for actor, rating in top_actors_by_rating:
        print(f"{actor}: {rating:.2f}")
    
    # Correlations
    correlation = filtered_df[['runtime_minutes', 'rating']].corr().iloc[0, 1]
    print(f"\nCorrelation between runtime and rating: {correlation:.3f}")
    
    return {
        'top_movies': top_movies,
        'top_actors': top_actors,
        'top_directors': top_directors,
        'genre_percentages': genre_percentages,
        'top_countries': top_countries,
        'top_languages': top_languages,
        'top_directors_by_rating': top_directors_by_rating,
        'top_actors_by_rating': top_actors_by_rating,
        'correlation': correlation
    }

# Visualization Functions
def create_visualizations(filtered_df, analysis_results):
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 30))
    
    # 1. Rating Distribution
    
    sns.histplot(filtered_df['rating'], bins=20, kde=True, color='blue')
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.show()
    # 2. Runtime Distribution
    
    sns.boxplot(data=filtered_df, x='runtime_minutes', color='lightgreen')
    plt.title("Runtime Distribution")
    plt.xlabel("Runtime (minutes)")
    plt.show()
    # 3. Top Actors
    top_actors_df = pd.DataFrame(analysis_results['top_actors'], columns=["Actor", "Count"])
    sns.barplot(data=top_actors_df, y="Actor", x="Count", palette="viridis")
    plt.title("Top Actors by Appearance Count")
    plt.show()
    
    # 4. Top Directors
    top_dirs_df = pd.DataFrame(analysis_results['top_directors'], columns=["Director", "Count"])
    sns.barplot(data=top_dirs_df, y="Director", x="Count", palette="rocket")
    plt.title("Top Directors by Appearance Count")
    plt.show()
    
    # 5. Actor-Director Network
    df_top50 = filtered_df.sort_values(by=['rating', 'votes_numeric'], ascending=False).head(50).copy()
    G = nx.Graph()
    for _, row in df_top50.iterrows():
        for director in row['directors']:
            for actor in row['cast'][:3]:
                G.add_node(director, type='director')
                G.add_node(actor, type='actor')
                G.add_edge(director, actor)
    
    node_colors = ['#1f77b4' if data['type'] == 'director' else '#ff7f0e' for _, data in G.nodes(data=True)]
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw(G, pos, node_color=node_colors, edge_color='gray', with_labels=True, 
            node_size=700, font_size=8, font_weight='bold')
    plt.title("Actor-Director Collaboration Network")
    plt.show()
    
    # 6. Genre Treemap
    genres_df = pd.DataFrame({
        'Genre': analysis_results['genre_percentages'].keys(),
        'Percentage': analysis_results['genre_percentages'].values()
    })
    squarify.plot(sizes=genres_df['Percentage'], label=genres_df['Genre'],
                 alpha=0.8, color=plt.cm.Paired.colors, pad=True)
    plt.axis('off')
    plt.title("Genre Distribution (Treemap)")
    plt.show()
    
    # 7. Genre Over Time
    df_expanded = pd.DataFrame([
        {'year': row['year'], 'genre': genre}
        for _, row in filtered_df.iterrows()
        for genre in row['genres']
    ])
    genre_counts = df_expanded.groupby(['year', 'genre']).size().unstack(fill_value=0).sort_index()
    genre_counts.plot.area(colormap='tab20')
    plt.title('Genre Popularity Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Movies')
    plt.grid(True)
    plt.show()
    
    # 8. Top Countries
    countries_df = pd.DataFrame(analysis_results['top_countries'], columns=["Country", "Count"])
    sns.barplot(data=countries_df, y="Country", x="Count", palette="mako")
    plt.title("Top Countries by Movie Count")
    plt.show()
    
    # 9. Top Languages
    langs_df = pd.DataFrame(analysis_results['top_languages'], columns=["Language", "Count"])
    sns.barplot(data=langs_df, y="Language", x="Count", palette="flare")
    plt.title("Top Languages by Movie Count")
    plt.show()
    
    # 10. Rating by Decade
    decade_avg = filtered_df.groupby('decade')['rating'].mean().reset_index()
    sns.lineplot(data=decade_avg, x='decade', y='rating', marker='o')
    plt.title("Average Rating by Decade")
    plt.xlabel("Decade")
    plt.ylabel("Average Rating")
    plt.grid(True)
    plt.show()
    
    # 11. Votes vs Rating
    sns.regplot(data=filtered_df, x='votes_numeric', y='rating', 
                scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
    plt.title("Votes vs. Rating")
    plt.xlabel("Votes")
    plt.ylabel("Rating")
    plt.show()
    
    # 12. Runtime vs Rating
    sns.regplot(data=filtered_df, x='runtime_minutes', y='rating', 
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title(f"Runtime vs. Rating (Corr: {analysis_results['correlation']:.3f})")
    plt.xlabel("Runtime (minutes)")
    plt.ylabel("Rating")
    plt.show()
    
    # 13. 3D Scatter Plot
    sample = filtered_df[['rating', 'runtime_minutes', 'votes_numeric', 'age_group']].dropna().sample(500)
    ax = plt.axes(projection='3d')
    ax.scatter(sample['rating'], sample['runtime_minutes'], sample['votes_numeric'], c='teal', alpha=0.5)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Runtime (min)')
    ax.set_zlabel('Votes')
    ax.set_title('3D Scatter Plot: Rating × Runtime × Votes')
    plt.show()
    
    # 14. Parallel Coordinates
    parallel_coordinates(sample, class_column='age_group', colormap=plt.get_cmap("Set1"))
    plt.title("Parallel Coordinates Plot of Movie Features")
    plt.show()
    
    # 15. Machine Learning Features
    ml_df = filtered_df.dropna(subset=['age_group', 'plot', 'genres'])
    ml_df['age_group'] = ml_df['age_group'].astype('category')
    
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    plot_features = tfidf.fit_transform(ml_df['plot'])
    
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(ml_df['genres'])
    
    X = hstack([plot_features, genres_encoded])
    y = ml_df['age_group'].cat.codes
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    feature_names = tfidf.get_feature_names_out().tolist() + list(mlb.classes_)
    importances = clf.feature_importances_
    top_features = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(20)
    
    sns.barplot(data=top_features, y='Feature', x='Importance', palette="magma")
    plt.title("Top 20 Most Important Features")
    plt.show()
    
    # 16. Word Cloud
    text = " ".join(filtered_df['plot'].dropna().tolist())
    wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud from Plot Descriptions")
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "D:\\Project Data Science Tools\\movies_with_imdb_data.csv"
    
    print("Loading and preprocessing data...")
    df = load_data(file_path)
    
    print("\nPerforming analysis...")
    analysis_results = perform_analysis(df)
    
    print("\nCreating visualizations...")
    create_visualizations(df, analysis_results)
    
    print("\nAnalysis complete! All visualizations displayed.")