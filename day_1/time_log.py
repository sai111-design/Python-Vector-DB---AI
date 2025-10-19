# Let me create the timing decorator and async URL fetcher code

# First, the timing decorator that works with both sync and async functions
import asyncio
import functools
import logging
import time
from typing import Callable, Any
import aiohttp
from contextlib import asynccontextmanager

# Set up logging for the examples
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def timer(func: Callable) -> Callable:
    """
    A decorator that logs execution time for both synchronous and asynchronous functions.
    
    This decorator automatically detects whether the decorated function is a coroutine
    and returns the appropriate wrapper (sync or async).
    """
    
    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper for synchronous functions."""
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            logger.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
    
    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper for asynchronous functions."""
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            logger.info(f"Async function '{func.__name__}' executed in {execution_time:.4f} seconds")
    
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Now let's create the async URL fetcher
class AsyncURLFetcher:
    """A class to fetch multiple URLs concurrently using aiohttp."""
    
    def __init__(self, max_concurrent_requests: int = 10, timeout: int = 30):
        """
        Initialize the fetcher with connection limits.
        
        Args:
            max_concurrent_requests: Maximum number of concurrent requests
            timeout: Request timeout in seconds
        """
        self.max_concurrent_requests = max_concurrent_requests
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.connector = aiohttp.TCPConnector(
            limit=max_concurrent_requests,
            limit_per_host=5  # Limit per host to be respectful
        )
    
    @timer
    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> dict:
        """
        Fetch a single URL and return response data.
        
        Args:
            session: aiohttp ClientSession
            url: URL to fetch
            
        Returns:
            dict: Contains url, status, content_length, and content or error
        """
        try:
            async with session.get(url, timeout=self.timeout) as response:
                content = await response.text()
                return {
                    'url': url,
                    'status': response.status,
                    'content_length': len(content),
                    'content': content[:200] + '...' if len(content) > 200 else content,
                    'headers': dict(response.headers),
                    'error': None
                }
        except asyncio.TimeoutError:
            return {
                'url': url,
                'status': None,
                'content_length': 0,
                'content': None,
                'headers': {},
                'error': 'Timeout'
            }
        except Exception as e:
            return {
                'url': url,
                'status': None,
                'content_length': 0,
                'content': None,
                'headers': {},
                'error': str(e)
            }
    
    @timer
    async def fetch_multiple_urls(self, urls: list[str]) -> list[dict]:
        """
        Fetch multiple URLs concurrently.
        
        Args:
            urls: List of URLs to fetch
            
        Returns:
            list: List of response dictionaries
        """
        async with aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout
        ) as session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            
            async def fetch_with_semaphore(url: str) -> dict:
                async with semaphore:
                    return await self.fetch_url(session, url)
            
            # Create tasks for all URLs
            tasks = [fetch_with_semaphore(url) for url in urls]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions that weren't caught
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        'url': urls[i],
                        'status': None,
                        'content_length': 0,
                        'content': None,
                        'headers': {},
                        'error': str(result)
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
    
    async def close(self):
        """Clean up the connector."""
        if self.connector:
            await self.connector.close()


# Example usage and demonstration
@timer
def sync_example():
    """Example synchronous function to demonstrate the timer decorator."""
    time.sleep(1)  # Simulate some work
    return "Sync function completed"

@timer
async def async_example():
    """Example asynchronous function to demonstrate the timer decorator."""
    await asyncio.sleep(1)  # Simulate some async work
    return "Async function completed"

# Sample URLs for testing (using httpbin and other reliable services)
SAMPLE_URLS = [
    'https://httpbin.org/delay/1',
    'https://httpbin.org/json',
    'https://httpbin.org/user-agent',
    'https://httpbin.org/headers',
    'https://httpbin.org/ip',
]


async def main():
    print("Running async URL fetcher example...\n")
    fetcher = AsyncURLFetcher(max_concurrent_requests=3, timeout=10)
    try:
        results = await fetcher.fetch_multiple_urls(SAMPLE_URLS)
        for result in results:
            print(f"URL: {result['url']}")
            print(f"  Status: {result['status']}")
            print(f"  Content length: {result['content_length']}")
            if result['error']:
                print(f"  Error: {result['error']}")
            print()
    finally:
        await fetcher.close()

if __name__ == "__main__":
    # Run sync and async example functions
    print(sync_example())
    print(asyncio.run(async_example()))
    asyncio.run(main())