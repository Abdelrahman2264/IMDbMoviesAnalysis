import csv
from imdb import IMDb
import concurrent.futures
from tqdm import tqdm
import time
import pickle
import os
from functools import lru_cache

# Configuration
INPUT_CSV = 'D:\\Project Data Science Tools\\IMDb Movies.csv'
OUTPUT_CSV = 'D:\\Project Data Science Tools\\movies_with_imdb_data.csv'
CACHE_FILE = 'D:\\Project Data Science Tools\\imdb_cache.pkl'
MAX_WORKERS = 8
REQUEST_DELAY = 0.5

# Initialize IMDb with caching
ia = IMDb()

# Load existing cache if available
try:
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
except (FileNotFoundError, EOFError, pickle.UnpicklingError):
    cache = {}

@lru_cache(maxsize=1000)
def get_movie_data(title, agegroup):
    """Get movie data with caching and optimized search"""
    cache_key = f"{title}_{agegroup}".lower()  # Unified cache key
    if cache_key in cache:
        return cache[cache_key]
    
    try:
        # Search with smart filtering
        results = ia.search_movie(title)
        if not results:
            return None
        
        # Find best match (exact title match first)
        movie = None
        for result in results:
            if result.get('title', '').lower() == title.lower():
                movie = result
                break
        
        # Fallback to first result if no exact match
        movie = movie or results[0]
        
        # Fetch only needed data sections
        try:
            ia.update(movie, info=['main', 'plot', 'vote details'])
        except Exception as e:
            print(f"\nWarning: Couldn't fetch all data for '{title}': {str(e)}")
        
        # Safely extract all data with proper error handling
        data = {
            'original_title': title,
            'age_group': agegroup,
            #  'imdb_title': movie.get('title'),
            'year': movie.get('year'),
            'rating': movie.get('rating'),
            'votes': movie.get('votes'),
            'genres': ', '.join(movie.get('genres', [])) if movie.get('genres') else '',
            'directors': ', '.join([d.get('name', '') for d in movie.get('director', [])][:3]),
            'cast': ', '.join([c.get('name', '') for c in movie.get('cast', [])][:5]),
            'plot': (movie.get('plot outline') or '').split('\n')[0],
            'runtime': movie.get('runtimes', [''])[0] if movie.get('runtimes') else '',
            'countries': ', '.join(movie.get('countries', [])[:3]),  # Reduced for efficiency
            'languages': ', '.join(movie.get('languages', [])[:3]),   # Reduced for efficiency
        }
        
        cache[cache_key] = data
        return data
    
    except Exception as e:
        print(f"\nError processing '{title}': {str(e)}")
        return None

def process_movie(movie_dict):
    """Wrapper function for parallel processing"""
    time.sleep(REQUEST_DELAY)
    title = movie_dict.get('title', '')
    agegroup = movie_dict.get('age_group', 'Not Rated')
    return get_movie_data(title, agegroup)

def main():
    try:
        # Read input CSV with error handling
        try:
            with open(INPUT_CSV, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                movies = [{'title': row['title'], 
                          'age_group': row.get('age_group', 'Not Rated')} 
                         for row in reader if row.get('title', '').strip()]
        except Exception as e:
            print(f"Error reading input file: {str(e)}")
            return

        if not movies:
            print("No valid movies found in the input file.")
            return

        print(f"Processing {len(movies)} movies...")

        # Process movies in parallel with error handling
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_movie, movie): movie for movie in movies}
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(movies), 
                              desc="Fetching Data"):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    movie = futures[future]
                    print(f"\nError processing movie {movie.get('title')}: {str(e)}")
                    continue

        if not results:
            print("No valid movie data found to output.")
            return

        # Define output fields
        fieldnames = [
            'original_title',
            'age_group',
            # 'imdb_title',
            'year',
            'rating',
            'votes',
            'genres',
            'directors',
            'cast',
            'plot',
            'runtime',
            'countries',
            'languages'
        ]

        # Write output CSV with error handling
        try:
            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        except Exception as e:
            print(f"Error writing output file: {str(e)}")
            return

        # Save cache with error handling
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache file: {str(e)}")

        print(f"\nCompleted! Processed {len(results)}/{len(movies)} movies successfully.")
        print(f"Results saved to {OUTPUT_CSV}")

    except Exception as e:
        print(f"\nFatal error in main execution: {str(e)}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")