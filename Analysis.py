import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import numpy as np
# Load data
df = pd.read_csv("D:\\Project Data Science Tools\\movies_with_imdb_data.csv")

# 1. Fill numeric missing values with mean
numeric_cols = [ 'rating', 'votes', 'runtime']
df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.mean()))
df['year'] = df['year'].dropna()
# 2. drop string missing values with mode
string_cols = ['genres', 'directors', 'cast', 'plot', 'countries', 'languages']

# Drop rows where any of the string columns are NaN or empty
df = df.dropna(subset=string_cols)

# Then remove rows where any of the columns are empty strings after stripping whitespace
for col in string_cols:
    df = df[df[col].str.strip() != '']



# 3. Convert comma-separated string values to lists
multi_feature_cols = ['genres', 'directors', 'cast', 'countries', 'languages']
for col in multi_feature_cols:
    df[col] = df[col].apply(lambda x: [item.strip() for item in str(x).split(',')])

# 4. Standardize age_group values using regex
# Ensure uppercased first
df['age_group'] = df['age_group'].str.upper().str.strip()

# Apply replacements
df['age_group'] = df['age_group'].str.replace(r'\+?18', 'NC-17', regex=True)
df['age_group'] = df['age_group'].str.replace(r'NC/?17', 'NC-17', regex=True)
df['age_group'] = df['age_group'].str.replace(r'\bM(/PG)?\b', 'PG', regex=True)
df['age_group'] = df['age_group'].str.replace(r'\bGP\b', 'PG', regex=True)
df['age_group'] = df['age_group'].str.replace(r'\bPG[- ]?13\b', 'PG-13', regex=True)
df['age_group'] = df['age_group'].str.replace(r'\bPG\b', 'PG', regex=True)
df['age_group'] = df['age_group'].str.replace(r'\bNOT RATED\b|\bUNRATED\b', 'UNRATED', regex=True)
df['age_group'] = df['age_group'].str.replace(r'\bPASSED\b|\bAPPROVED\b', 'G', regex=True)
df['age_group'] = df['age_group'].str.replace(r'\bTV[\s-]?G\b', 'TV-G', regex=True)
df['age_group'] = df['age_group'].str.replace(r'\bTV[\s-]?PG\b', 'TV-PG', regex=True)
df['age_group'] = df['age_group'].str.replace(r'\bTV[\s-]?14\b', 'TV-14', regex=True)
df['age_group'] = df['age_group'].str.replace(r'\bTV[\s-]?MA\b', 'TV-MA', regex=True)

# Final cleanup: strip extra spaces
df['age_group'] = df['age_group'].str.strip()


# 5. Format runtime from minutes to HH:MM:SS
df['runtime'] = pd.to_timedelta(df['runtime'], unit='m').astype(str).str.extract(r'(\d+:\d+:\d+)')[0]

# 6. Format votes into K or M
def format_votes(v):
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    elif v >= 1_000:
        return f"{v/1_000:.0f}K"
    else:
        return str(int(v))
df['votes_numeric'] = df['votes']
df['votes'] = df['votes'].apply(format_votes)

# 7. Clean special characters from plot text
df['plot'] = df['plot'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))

# Convert list-like columns stored as strings to actual lists
list_cols = ['genres', 'directors', 'cast', 'countries', 'languages']
for col in list_cols:
    df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Get top 1000 movies based on rating and votes
top_movies = df.sort_values(by=['rating', 'votes_numeric'], ascending=False).head(1000)

# Flatten list-type columns
def flatten_list_column(column):
    return [item for sublist in column for item in sublist]

# 🔹 Top 10 actors and directors by frequency
top_actors = Counter(flatten_list_column(top_movies['cast'])).most_common(10)
top_directors = Counter(flatten_list_column(top_movies['directors'])).most_common(10)

# 🔹 Genre distribution percentage
genre_counts = Counter(flatten_list_column(top_movies['genres']))
total_genres = sum(genre_counts.values())
genre_percentages = {genre: f"{(count/total_genres)*100:.1f}%" for genre, count in genre_counts.items()}
genre_percentages = dict(sorted(genre_percentages.items(), key=lambda x: float(x[1][:-1]), reverse=True))

# 🔹 Top 10 countries and languages
top_countries = Counter(flatten_list_column(top_movies['countries'])).most_common(10)
top_languages = Counter(flatten_list_column(top_movies['languages'])).most_common(10)

# 🔹 Top directors and actors by average rating
def top_avg_rating(df, col):
    items = []
    for entry in set(flatten_list_column(df[col])):
        subset = df[df[col].apply(lambda x: entry in x)]
        avg_rating = subset['rating'].mean()
        items.append((entry, avg_rating))
    return sorted(items, key=lambda x: x[1], reverse=True)[:5]

top_directors_by_rating = top_avg_rating(top_movies, 'directors')
top_actors_by_rating = top_avg_rating(top_movies, 'cast')

# 🔹 Correlation between runtime and rating
def time_to_minutes(t):
    try:
        h, m, s = map(int, str(t).split(":"))
        return h * 60 + m + s / 60
    except:
        return None
df['runtime_minutes'] = df['runtime'].apply(time_to_minutes)
correlation = df[['runtime_minutes', 'rating']].corr().iloc[0, 1]


# Ensure no missing values in required columns
df = df.dropna(subset=['age_group', 'plot', 'genres'])

# Convert age_group to categorical
df['age_group'] = df['age_group'].astype('category')

# TF-IDF Vectorizer for plot text
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
plot_features = tfidf.fit_transform(df['plot'])

# MultiLabelBinarizer for genres (one-hot encode)
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['genres'])

# Combine both features (plot + genres)
X = hstack([plot_features, genres_encoded])
y = df['age_group'].cat.codes  # Use category codes for y

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Handle mismatch in labels and target names
used_labels = sorted(np.unique(y_test))
used_names = df['age_group'].cat.categories[used_labels]

# Print classification report
print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred, target_names=used_names))


# Save cleaned dataset
df.to_csv("cleaned_movies_data.csv", index=False)

# 🔹 Display results
print("🔸 Top 10 Actors:")
print(top_actors)

print("\n🔸 Top 10 Directors:")
print(top_directors)

print("\n🔸 Genre Percentages:")
print(genre_percentages)

print("\n🔸 Top Countries:")
print(top_countries)

print("\n🔸 Top Languages:")
print(top_languages)

print("\n🔸 Top Directors by Average Rating:")
print(top_directors_by_rating)

print("\n🔸 Top Actors by Average Rating:")
print(top_actors_by_rating)

print(f"\n🔸 Correlation between Runtime and Rating: {correlation:.3f}")