import streamlit as st
import pandas as pd
import numpy as np
import re
import ast
from collections import Counter, defaultdict
from itertools import combinations
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
from pymongo import MongoClient
from pathlib import Path
import json
import os



# Load data
@st.cache_data
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
st.set_page_config(page_title="LIVERTOOL ANALYSIS", layout="wide")
# Title of your app
st.title("Interactive Web App For Analysis IMDb Website For Movies")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
 
    # Custom CSS for better styling
    st.markdown("""
  <style>
    .main {
        background-color: #121212; /* dark background */
        color: #e0e0e0; /* light text */
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1f1f1f; /* dark tab background */
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #cccccc; /* light tab text */
    }

    .stTabs [aria-selected="true"] {
        background-color: #2c2c2c; /* slightly lighter when selected */
        color: #ffffff;
    }

    .plot-container {
        border: 1px solid #333333; /* dark border */
        border-radius: 4px;
        padding: 20px;
        background-color: #1e1e1e; /* dark plot background */
        margin-bottom: 20px;
    }

    .header-box {
        background-color: #1976d2; /* dark blue shade for headers */
        color: white;
        padding: 10px 15px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
</style>
    """, unsafe_allow_html=True)

    # Title and file upload
    st.title("🎬 Comprehensive Movie Analysis Dashboard")
    # Main content
    st.sidebar.header("Filters")
    min_year, max_year = int(df['year'].min()), int(df['year'].max())
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    min_rating = st.sidebar.slider(
        "Minimum Rating",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5
    )

    # Filter data
    filtered_df = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1]) & 
        (df['rating'] >= min_rating)
    ]

    # Top movies
    top_movies = filtered_df.sort_values(by=['rating', 'votes_numeric'], ascending=False).head(1000)

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

    # Calculate metrics
    top_actors = Counter(flatten_list_column(top_movies['cast'])).most_common(10)
    top_directors = Counter(flatten_list_column(top_movies['directors'])).most_common(10)

    genre_counts = Counter(flatten_list_column(top_movies['genres']))
    genre_percentages = {genre: f"{(count/sum(genre_counts.values()))*100:.1f}%" 
                        for genre, count in genre_counts.items()}
    genre_percentages = dict(sorted(genre_percentages.items(), 
                                key=lambda x: float(x[1][:-1]), reverse=True))

    top_countries = Counter(flatten_list_column(top_movies['countries'])).most_common(10)
    top_languages = Counter(flatten_list_column(top_movies['languages'])).most_common(10)

    top_directors_by_rating = top_avg_rating(top_movies, 'directors')
    top_actors_by_rating = top_avg_rating(top_movies, 'cast')

    correlation = filtered_df[['runtime_minutes', 'rating']].corr().iloc[0, 1]

    # ML Model
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
    y_pred = clf.predict(X_test)

    used_labels = sorted(np.unique(y_test))
    used_names = ml_df['age_group'].cat.categories[used_labels]

    # Create tabs
    tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs([
        "📊 Overview ", 
        "🎭 People Analytics ", 
        "🌍 Geographic & Genre ", 
        "📈 Trends & Relationships ",
        "🤖 Machine Learning Model ",
        "💾 MongoDB & JSON"
    ])

    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Movies", len(filtered_df))
        with col2:
            st.metric("Average Rating", f"{filtered_df['rating'].mean():.2f}")
        with col3:
            st.metric("Earliest Year", int(filtered_df['year'].min()))
        
        st.subheader("Data Preview")
        st.dataframe(filtered_df.head(10))
        
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_df['rating'], bins=20, kde=True, color='blue')
        plt.title("Rating Distribution")
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        st.pyplot(fig)
        
        st.subheader("Runtime Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=filtered_df, x='runtime_minutes', color='lightgreen')
        plt.title("Runtime Distribution")
        plt.xlabel("Runtime (minutes)")
        st.pyplot(fig)

    with tab2:
        st.header("People Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 10 Actors (Frequency)")
            top_actors_df = pd.DataFrame(top_actors, columns=["Actor", "Count"])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=top_actors_df, y="Actor", x="Count", hue="Actor", palette="viridis", legend=False) 
            plt.title("Top Actors by Appearance Count")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Top 10 Directors (Frequency)")
            top_dirs_df = pd.DataFrame(top_directors, columns=["Director", "Count"])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=top_dirs_df, y="Director", x="Count", hue="Director", palette="rocket", legend=False)
            plt.title("Top Directors by Appearance Count")
            st.pyplot(fig)
        
        st.subheader("Top Performers by Average Rating")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 5 Directors**")
            rating_dirs = pd.DataFrame(top_directors_by_rating, 
                                    columns=["Director", "Avg Rating"])
            st.dataframe(rating_dirs.sort_values("Avg Rating", ascending=False))
            
        with col2:
            st.markdown("**Top 5 Actors**")
            rating_actors = pd.DataFrame(top_actors_by_rating, 
                                    columns=["Actor", "Avg Rating"])
            st.dataframe(rating_actors.sort_values("Avg Rating", ascending=False))
        
        st.subheader("Actor-Director Collaboration Network")
        df_top50 = filtered_df.sort_values(by=['rating', 'votes_numeric'], ascending=False).head(50).copy()
        df_top50['directors'] = df_top50['directors'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
        df_top50['cast'] = df_top50['cast'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
        
        G = nx.Graph()
        for _, row in df_top50.iterrows():
            for director in row['directors']:
                for actor in row['cast'][:3]:
                    G.add_node(director, type='director')
                    G.add_node(actor, type='actor')
                    G.add_edge(director, actor)
        
        node_colors = ['#1f77b4' if data['type'] == 'director' else '#ff7f0e' for _, data in G.nodes(data=True)]
        
        fig, ax = plt.subplots(figsize=(16, 12))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        nx.draw(
            G, pos,
            node_color=node_colors,
            edge_color='gray',
            with_labels=True,
            node_size=700,
            font_size=8,
            font_weight='bold'
        )
        plt.title("Actor-Director Collaboration Network (Top Movies)")
        st.pyplot(fig)

    with tab3:
        st.header("Geographic & Genre Analysis")
        
        st.subheader("Genre Distribution")
        genres_df = pd.DataFrame({
            'Genre': genre_percentages.keys(),
            'Percentage': [float(x[:-1]) for x in genre_percentages.values()]
        })
        # Filter out zero values and ensure percentages are positive
        genres_df = genres_df[genres_df['Percentage'] > 0]

        if not genres_df.empty:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(12, 8))
                try:
                    squarify.plot(
                        sizes=genres_df['Percentage'],
                        label=genres_df['Genre'],
                        alpha=0.8,
                        color=plt.cm.Paired.colors,
                        pad=True  # Add padding between rectangles
                    )
                    plt.axis('off')
                    plt.title("Genre Distribution (Treemap)")
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate treemap: {str(e)}")
                    st.write("Alternative genre distribution:")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=genres_df.head(10), y='Genre', x='Percentage')
                    st.pyplot(fig)
            
            with col2:
                st.dataframe(genres_df.head(10))
        else:
            st.warning("No genre data available for visualization")
            
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(12, 8))
            squarify.plot(
                sizes=genres_df['Percentage'],
                label=genres_df['Genre'],
                alpha=0.8,
                color=plt.cm.Paired.colors
            )
            plt.axis('off')
            plt.title("Genre Distribution (Treemap)")
            st.pyplot(fig)
        
        with col2:
            st.dataframe(genres_df.head(10))
        
        st.subheader("Genre Popularity Over Time")
        df_expanded = pd.DataFrame([
            {'year': row['year'], 'genre': genre}
            for _, row in filtered_df.iterrows()
            for genre in row['genres']
        ])
        genre_counts = df_expanded.groupby(['year', 'genre']).size().unstack(fill_value=0).sort_index()
        
        fig, ax = plt.subplots(figsize=(14, 7))
        genre_counts.plot.area(colormap='tab20', ax=ax)
        plt.title('Genre Popularity Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Movies')
        plt.grid(True)
        st.pyplot(fig)
        
        st.subheader("Geographic Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Countries**")
            countries_df = pd.DataFrame(top_countries, columns=["Country", "Count"])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=countries_df, y="Country", x="Count", palette="mako")
            plt.title("Top Countries by Movie Count")
            st.pyplot(fig)
            
        with col2:
            st.markdown("**Top Languages**")
            langs_df = pd.DataFrame(top_languages, columns=["Language", "Count"])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=langs_df, y="Language", x="Count", palette="flare")
            plt.title("Top Languages by Movie Count")
            st.pyplot(fig)

    with tab4:
        st.header("Trends & Relationships")
        
        st.subheader("Average Rating by Decade")
        decade_avg = filtered_df.groupby('decade')['rating'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=decade_avg, x='decade', y='rating', marker='o')
        plt.title("Average Rating by Decade")
        plt.xlabel("Decade")
        plt.ylabel("Average Rating")
        plt.grid(True)
        st.pyplot(fig)
        
        st.subheader("Votes vs. Rating")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.regplot(data=filtered_df, x='votes_numeric', y='rating', 
                    scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
        plt.title("Votes vs. Rating")
        plt.xlabel("Votes")
        plt.ylabel("Rating")
        st.pyplot(fig)
        
        st.subheader("Runtime vs. Rating")
        st.markdown(f"**Correlation Coefficient:** `{correlation:.3f}`")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.regplot(data=filtered_df, x='runtime_minutes', y='rating', 
                    scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        plt.title("Runtime vs. Rating")
        plt.xlabel("Runtime (minutes)")
        plt.ylabel("Rating")
        st.pyplot(fig)
        
        st.subheader("3D Relationship: Rating × Runtime × Votes")
        sample = filtered_df[['rating', 'runtime_minutes', 'votes_numeric', 'age_group']].dropna().sample(500)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sample['rating'], sample['runtime_minutes'], sample['votes_numeric'], c='teal', alpha=0.5)
        ax.set_xlabel('Rating')
        ax.set_ylabel('Runtime (min)')
        ax.set_zlabel('Votes')
        ax.set_title('3D Scatter Plot: Rating × Runtime × Votes')
        st.pyplot(fig)
        
        st.subheader("Parallel Coordinates Plot")
        fig, ax = plt.subplots(figsize=(12, 6))
        parallel_coordinates(sample, class_column='age_group', colormap=plt.get_cmap("Set1"))
        plt.title("Parallel Coordinates Plot of Movie Features")
        st.pyplot(fig)

    with tab5:
        st.header("Machine Learning Model")
        
        st.subheader("Age Group Classification Performance")
        report = classification_report(
            y_test, y_pred, 
            target_names=used_names, 
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap="YlGnBu"))
        
        st.subheader("Most Important Features for Classification")
        feature_names = tfidf.get_feature_names_out().tolist() + list(mlb.classes_)
        importances = clf.feature_importances_
        top_features = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(20)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=top_features, y='Feature', x='Importance', palette="magma")
        plt.title("Top 20 Most Important Features")
        st.pyplot(fig)
        
        st.subheader("Plot Word Cloud")
        text = " ".join(filtered_df['plot'].dropna().tolist())
        wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(15, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud from Plot Descriptions")
        st.pyplot(fig)

    # Download button
    st.sidebar.download_button(
        label="📥 Download Cleaned Data",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name="filtered_movies_data.csv",
        mime="text/csv"
    )
    st.sidebar.markdown("""
        <hr style="border-top: 1px solid #555;" />
        <p style="text-align: center; color: white; font-size: 1em; font-weight:bold">
        "&copy; 2025 LIVERTOOL. All rights reserved."

        </p>
    """, unsafe_allow_html=True)
    # Add this MongoDB section in the new tab
    with tab6:
        st.header("MongoDB Storage & JSON Export")
        
        # MongoDB Connection Section
        st.subheader("MongoDB Connection")
        col1, col2 = st.columns(2)
        
        with col1:
            mongodb_uri = st.text_input(
                "MongoDB URI:",
                "mongodb://localhost:27017/",
                help="Format: mongodb://username:password@host:port/"
            )
        
        with col2:
            db_name = st.text_input(
                "Database Name:",
                "MoviesDB",
                help="Name of the MongoDB database to use"
            )
        
        collection_name = st.text_input(
            "Collection Name:",
            "movies",
            help="Name of the collection to store the data"
        )
        
        # MongoDB Operations Section
        st.subheader("Database Operations")
        
        if st.button("💾 Save to MongoDB", help="Store the current filtered data in MongoDB"):
            try:
                with st.spinner("Connecting to MongoDB and saving data..."):
                    # Connect to MongoDB
                    client = MongoClient(mongodb_uri)
                    db = client[db_name]
                    collection = db[collection_name]
                    
                    # Convert DataFrame to dictionary
                    data = filtered_df.to_dict(orient="records")
                    
                    # Insert data (clear existing data first)
                    collection.delete_many({})
                    result = collection.insert_many(data)
                    
                    st.success(f"✅ Successfully saved {len(result.inserted_ids)} documents to MongoDB!")
                    st.info(f"Database: {db_name}\nCollection: {collection_name}")
                    
            except Exception as e:
                st.error(f"❌ Error saving to MongoDB: {str(e)}")
        
        # JSON Export Section
        st.subheader("JSON Export")
        
        if st.button("📤 Export to JSON", help="Export data from MongoDB to JSON file"):
            try:
                with st.spinner("Exporting data from MongoDB..."):
                    # Connect to MongoDB
                    client = MongoClient(mongodb_uri)
                    db = client[db_name]
                    collection = db[collection_name]
                    
                    # Create output directory
                    output_dir = os.path.join(os.getcwd(), "mongodb_exports")
                    Path(output_dir).mkdir(exist_ok=True)
                    
                    # Create output path
                    output_path = os.path.join(output_dir, f"{collection_name}_export.json")
                    
                    # Fetch data and save as JSON
                    cursor = collection.find({}, {'_id': 0})  # Exclude MongoDB _id field
                    json_data = list(cursor)
                    
                    # Save to file
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    
                    # Show preview
                    st.success(f"✅ Successfully exported {len(json_data)} documents to JSON!")
                    st.code(f"File saved to: {output_path}")
                    
                    # Display JSON preview
                    with st.expander("Preview first document:"):
                        st.json(json_data[0] if json_data else {})
                    
                    # Download button
                    json_str = json.dumps(json_data, indent=2)
                    st.download_button(
                        label="⬇️ Download JSON File",
                        data=json_str,
                        file_name=f"{collection_name}_export.json",
                        mime="application/json",
                        help="Download the complete JSON file"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error exporting from MongoDB: {str(e)}")
        
        # Database Stats Section
        st.subheader("Database Information")
        
        if st.button("🔍 Show Database Stats", help="Display collection statistics"):
            try:
                with st.spinner("Fetching database information..."):
                    client = MongoClient(mongodb_uri)
                    db = client[db_name]
                    collection = db[collection_name]
                    
                    count = collection.count_documents({})
                    sample_doc = collection.find_one({}, {'_id': 0})
                    
                    st.info(f"📊 Collection '{collection_name}' contains {count} documents")
                    
                    if sample_doc:
                        with st.expander("Sample Document"):
                            st.json(sample_doc)
                    
            except Exception as e:
                st.error(f"❌ Error connecting to MongoDB: {str(e)}")


