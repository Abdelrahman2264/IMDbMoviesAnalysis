import csv
from imdb import IMDb
import concurrent.futures
from tqdm import tqdm
import time
import pickle
import os
from functools import lru_cache

# Configuration
INPUT_CSV = 'c:\\Users\\Abdelrahman.khalaf\\imdb_movies.csv'
OUTPUT_CSV = 'c:\\Users\\Abdelrahman.khalaf\\movies_with_imdb_data.csv'
CACHE_FILE = 'c:\\Users\\Abdelrahman.khalaf\\imdb_cache.pkl'
MAX_WORKERS = 12  # Increased workers
REQUEST_DELAY = 0.1  # Reduced delay
BATCH_SIZE = 100  # Process in batches
CACHE_SIZE = 5000  # Increased cache size

# Initialize IMDb
ia = IMDb()

# Load cache with faster pickle protocol
cache = {}
try:
    with open(CACHE_FILE, 'rb') as f:
        cache.update(pickle.load(f))
except (FileNotFoundError, EOFError, pickle.UnpicklingError):
    pass

@lru_cache(maxsize=CACHE_SIZE)
def get_movie_data(title, agegroup):
    """Optimized movie data fetcher with caching"""
    cache_key = (title.lower(), agegroup.lower())
    if cache_key in cache:
        return cache[cache_key]
    
    try:
        # First try exact match search
        results = ia.search_movie(title)
        if not results:
            return None
        
        # Find best match quickly
        movie = next((r for r in results if r.get('title', '').lower() == title.lower()), results[0])
        
        # Minimal data fetching
        ia.update(movie, info=['main', 'vote details'])
        
        data = {
            'title': title,
            'age_group': agegroup,
            'year': movie.get('year'),
            'rating': movie.get('rating'),
            'votes': movie.get('votes'),
            'genres': '|'.join(movie.get('genres', [])[:5]),
            'directors': '|'.join(d.get('name', '') for d in movie.get('director', [])[:3]),
            'cast': '|'.join(c.get('name', '') for c in movie.get('cast', [])[:5]),
            'runtime': movie.get('runtimes', [''])[0],
            'countries': '|'.join(movie.get('countries', [])[:3]),
            'languages': '|'.join(movie.get('languages', [])[:3]),
        }
        
        cache[cache_key] = data
        return data
    
    except Exception:
        return None

def process_batch(batch):
    """Process a batch of movies efficiently"""
    results = []
    for movie in batch:
        time.sleep(REQUEST_DELAY)
        result = get_movie_data(movie['title'], movie['age_group'])
        if result:
            results.append(result)
    
    # Save cache periodically
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    
    return results

def main():
    try:
        # Fast CSV reading
        with open(INPUT_CSV, 'r', encoding='utf-8') as f:
            movies = [{'title': row['title'], 
                      'age_group': row.get('age_group', 'Not Rated')}
                     for row in csv.DictReader(f) 
                     if row.get('title', '').strip()]

        if not movies:
            print("No valid movies found.")
            return

        print(f"Processing {len(movies)} movies...")

        # Process in parallel batches
        results = []
        batches = [movies[i:i + BATCH_SIZE] for i in range(0, len(movies), BATCH_SIZE)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            for future in tqdm(concurrent.futures.as_completed(futures), 
                            total=len(batches), 
                            desc="Processing"):
                results.extend(future.result())

        if not results:
            print("No valid data found.")
            return

        # Fast CSV writing
        fieldnames = ['title', 'age_group', 'year', 'rating', 'votes', 
                     'genres', 'directors', 'cast', 'runtime', 
                     'countries', 'languages']
        
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nCompleted! Processed {len(results)} movies.")
        print(f"Results saved to {OUTPUT_CSV}")

    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal time: {time.time() - start:.2f} seconds")