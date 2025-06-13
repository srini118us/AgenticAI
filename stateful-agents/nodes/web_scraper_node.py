from typing import Any, Dict, List
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from .base_node import BaseNode, NodeState

class WebScraperNode(BaseNode):
    """Node for handling web scraping operations"""
    
    def __init__(self, node_id: str = "web_scraper"):
        super().__init__(node_id)
        self.session = None
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    async def _init_session(self):
        """Initialize aiohttp session if not exists"""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
    
    async def _close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    async def execute(self, input_data: Dict[str, Any]) -> NodeState:
        """Execute web scraping"""
        try:
            self.update_state(status="running")
            
            urls = input_data.get("urls", [])
            if not urls:
                raise ValueError("No URLs provided for web scraping")
            
            if not all(self._is_valid_url(url) for url in urls):
                raise ValueError("Invalid URL format provided")
            
            await self._init_session()
            
            results = []
            for url in urls:
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Remove script and style elements
                            for script in soup(["script", "style"]):
                                script.decompose()
                            
                            # Get text content
                            text = soup.get_text(separator='\n', strip=True)
                            
                            # Clean up text
                            lines = (line.strip() for line in text.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            text = '\n'.join(chunk for chunk in chunks if chunk)
                            
                            results.append({
                                "url": url,
                                "content": text,
                                "status": "success"
                            })
                        else:
                            results.append({
                                "url": url,
                                "status": "failed",
                                "error": f"HTTP {response.status}"
                            })
                except Exception as e:
                    results.append({
                        "url": url,
                        "status": "failed",
                        "error": str(e)
                    })
            
            await self._close_session()
            
            self.update_state(
                status="completed",
                result=results,
                metadata={"num_urls": len(urls), "successful_scrapes": len([r for r in results if r["status"] == "success"])}
            )
            
            return self.state
            
        except Exception as e:
            await self._close_session()
            self.update_state(status="failed", error=str(e))
            return self.state 