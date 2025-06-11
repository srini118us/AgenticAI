import requests

def scrape_website(url: str) -> str:
    """Scrapes content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error scraping {url}: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred during web scraping: {str(e)}"

if __name__ == "__main__":
    print("--- Web Scraper Simple Example ---")
    # Note: Replace with a real URL if you want to test live, but be mindful of robots.txt
    # and website terms of service. Using a placeholder for now.
    example_url = "http://example.com"
    web_content = scrape_website(example_url)
    print(f"Scraped Content from {example_url}:\n{web_content[:200]}...") # Print first 200 chars 