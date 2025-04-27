from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import time
import csv
import signal
import sys

# Configuration
MAX_MOVIES = 25000  # Set to None to scrape all available
CSV_FILENAME = 'D:\\Project Data Science Tools\\IMDb Movies.csv'
LOAD_MORE_DELAY = 5  # Seconds to wait between loads
INITIAL_LOAD_DELAY = 5  # Seconds to wait for initial page load
DEBUG = True  # Set to True for verbose logging

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    global shutdown_flag
    print("\nShutting down gracefully...")
    shutdown_flag = True
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def debug_print(message):
    if DEBUG:
        print(message)

# Setup Chrome WebDriver
service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--no-sandbox')
# options.add_argument('--headless')  # Uncomment for headless mode
driver = webdriver.Chrome(service=service, options=options)

def initialize_csv():
    """Create or clear the CSV file and write headers"""
    with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['rank', 'title', 'age_group'])

def append_movie_to_csv(movie_data):
    """Append a single movie's data to the CSV file"""
    with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(movie_data)

def get_current_movie_count():
    """Get count of currently loaded movie elements"""
    try:
        return len(driver.find_elements(By.CSS_SELECTOR, ".ipc-metadata-list-summary-item"))
    except:
        return 0

def scrape_visible_movies(last_count):
    """Scrape newly loaded movies since last count"""
    try:
        movies = driver.find_elements(By.CSS_SELECTOR, ".ipc-metadata-list-summary-item")
        new_movies = movies[last_count:]
        scraped_count = 0
        
        for movie in new_movies:
            if shutdown_flag or (MAX_MOVIES and last_count + scraped_count >= MAX_MOVIES):
                break
                
            try:
                # Extract data with error handling
                rank = last_count + scraped_count + 1
                title_with_rank = movie.find_element(By.CSS_SELECTOR, "h3.ipc-title__text").text
                title = title_with_rank.split('. ', 1)[1] if '. ' in title_with_rank else title_with_rank
                
                # Extract age group from the metadata items
                try:
                    metadata_items = movie.find_elements(
                        By.CSS_SELECTOR, ".sc-5179a348-7.idrYgr.dli-title-metadata-item"
                    )
                    # The age group is typically the third item (index 2)
                    age_group = metadata_items[2].text if len(metadata_items) > 2 else "Not Rated"
                except:
                    age_group = "Not Rated"
                
                # Append to CSV
                append_movie_to_csv([rank, title, age_group])
                scraped_count += 1
                
                if scraped_count % 50 == 0:
                    debug_print(f"Scraped {last_count + scraped_count} movies so far...")
                
            except Exception as e:
                debug_print(f"Error processing movie: {str(e)}")
                continue
                
        return scraped_count
        
    except Exception as e:
        debug_print(f"Error scraping visible movies: {str(e)}")
        return 0
def load_more_movies():
    """Click the '50 more' button to load additional movies"""
    try:
        debug_print("Looking for '50 more' button...")
        
        # Scroll to bottom to make button visible
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        
        # Find the button using the exact class structure
        button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.ipc-see-more__button"))
        )
        
        # Scroll the button into view (just in case)
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
        time.sleep(0.5)
        
        # Click using JavaScript to avoid interception issues
        driver.execute_script("arguments[0].click();", button)
        debug_print("Clicked '50 more' button")
        
        # Wait for new content to load
        WebDriverWait(driver, 10).until(
            EC.invisibility_of_element_located((By.CSS_SELECTOR, ".ipc-page-spinner")))
        time.sleep(LOAD_MORE_DELAY)
        return True
        
    except TimeoutException:
        debug_print("Timed out waiting for '50 more' button")
        return False
    except NoSuchElementException:
        debug_print("Could not find '50 more' button")
        return False
    except Exception as e:
        debug_print(f"Error clicking '50 more': {str(e)}")
        return False

# Main scraping process
try:
    # Initialize CSV
    initialize_csv()
    
    # Load initial page
    debug_print("Loading initial page...")
    driver.get("https://www.imdb.com/search/title/?title_type=feature&sort=num_votes,desc")
    WebDriverWait(driver, INITIAL_LOAD_DELAY).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".ipc-metadata-list-summary-item"))
    )
    
    total_scraped = 0
    last_count = 0
    
    # First scrape of initial movies
    initial_movies = get_current_movie_count()
    if initial_movies > 0:
        debug_print(f"Found {initial_movies} initial movies")
        total_scraped = scrape_visible_movies(0)
        last_count = initial_movies
    
    while not shutdown_flag and (not MAX_MOVIES or total_scraped < MAX_MOVIES):
        # Try to load more movies
        if not load_more_movies():
            debug_print("No more movies to load or error occurred")
            break
            
        # Check if new movies were loaded
        current_count = get_current_movie_count()
        if current_count <= last_count:
            debug_print("No new movies loaded after clicking button")
            time.sleep(5)  # Extra wait in case loading is slow
            current_count = get_current_movie_count()
            if current_count <= last_count:
                break
                
        # Scrape the new movies
        new_scraped = scrape_visible_movies(last_count)
        if new_scraped == 0:
            debug_print("No new movies scraped - possible end of results")
            break
            
        total_scraped += new_scraped
        last_count = current_count
        
        debug_print(f"Total movies scraped: {total_scraped}")
        
        # Small delay between loads
        time.sleep(1)

except Exception as e:
    debug_print(f"Fatal error during scraping: {str(e)}")
finally:
    # Cleanup
    driver.quit()
    print(f"\nScraping complete. Total movies scraped: {total_scraped}")
    print(f"Data saved to {CSV_FILENAME}")