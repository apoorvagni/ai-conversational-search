from langchain_community.tools import DuckDuckGoSearchResults
from bs4 import BeautifulSoup
import requests
import logging
import os
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import HumanMessage, AIMessage
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from functools import partial
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from ratelimit import limits, sleep_and_retry
import asyncio
from aiohttp import ClientSession, ClientTimeout

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchChat:
    MODEL_NAME = "mistral-small"
    
    def __init__(self):
        # Initialize Mistral client using LangChain's ChatMistralAI
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable must be set")
        self.client = ChatMistralAI(
            api_key=api_key,
            model=self.MODEL_NAME
        )
        
        # Set default USER_AGENT if not present
        if not os.getenv('USER_AGENT'):
            os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        
        try:
            self.search = DuckDuckGoSearchResults()
            self.conversation = []
            self.headers = {
                'User-Agent': os.getenv('USER_AGENT'),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            self.session = requests.Session()
            retries = Retry(total=3,
                           backoff_factor=0.5,
                           status_forcelist=[500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retries)
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
            self.session.headers.update(self.headers)
            # Add default timeout for all requests
            self.session.request = partial(self.session.request, timeout=(10, 60))
            # Add rate limiter for Mistral API
            self.last_api_call = 0
            self.api_call_interval = 1.1  # Slightly more than 1 second to be safe
        except Exception as e:
            logger.error(f"Error initializing WebSearchChat: {str(e)}")
            raise e

    def wait_for_rate_limit(self):
        """Ensure we respect the 1 RPS rate limit"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.api_call_interval:
            time.sleep(self.api_call_interval - time_since_last_call)
        self.last_api_call = time.time()

    async def fetch_url_async(self, session: ClientSession, url: str) -> Tuple[str, str]:
        """Asynchronous URL fetching"""
        try:
            timeout = ClientTimeout(total=30)
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                        element.decompose()
                    
                    # First try to find article content
                    article = soup.find('article')
                    if article:
                        text = article.get_text(separator='\n', strip=True)
                        if len(text) > 100:
                            return url, text[:4000]
                    
                    # If no article, get main content
                    content = soup.find_all(['main', 'div', 'p'])
                    text = '\n'.join(tag.get_text(strip=True) for tag in content if len(tag.get_text(strip=True)) > 50)
                    return url, text[:4000]
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
        return url, ""

    async def process_urls_async(self, urls: List[str]) -> List[Tuple[str, str]]:
        """Process multiple URLs asynchronously"""
        async with ClientSession(headers=self.headers) as session:
            tasks = [self.fetch_url_async(session, url) for url in urls]
            return await asyncio.gather(*tasks)

    def fetch_and_parse_url(self, url):
        """Custom method to fetch and parse URL content"""
        try:
            # Use session instead of requests.get
            response = self.session.get(url, timeout=(10, 30))
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe', 'noscript']):
                element.decompose()
            
            # First try to find article content
            article = soup.find('article')
            if article:
                text = article.get_text(separator='\n', strip=True)
                if len(text) > 100:
                    return text[:4000]  # Increased max chars
            
            # If no article, look for main content
            content_tags = soup.find_all(['main', 'div', 'p', 'section'])
            text_content = []
            total_length = 0
            max_chars = 4000  # Increased from 2000
            
            for tag in content_tags:
                text = tag.get_text(separator='\n', strip=True)
                if text and len(text) > 50:  # Reduced minimum length
                    text_content.append(text)
                    total_length += len(text)
                    if total_length >= max_chars:
                        break
            
            return '\n'.join(text_content)[:max_chars] if text_content else ""
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return ""

    def summarize_content(self, content: str) -> str:
        """Summarize content using LangChain's summarization"""
        try:
            self.wait_for_rate_limit()  # Add rate limiting
            # Split content into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_text(content)
            docs = [Document(page_content=chunk) for chunk in chunks]
            
            # Create summarization prompt with correct variable name
            prompt = ChatPromptTemplate.from_template("""
                Summarize the following text in a concise manner, focusing on key information:
                {context}
                
                Summary:
            """)
            
            # Create and run chain
            chain = create_stuff_documents_chain(
                self.client,
                prompt,
            )
            
            # Get the result and return it directly since it's already a string
            result = chain.invoke({
                "context": docs
            })
            
            return result  # Remove ["answer"] indexing since result is already a string

        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return content[:2000]  # Fallback to truncation

    def process_url(self, url: str) -> Tuple[str, str]:
        """Process single URL with summarization - synchronous version"""
        try:
            content = self.fetch_and_parse_url(url)
            if len(content) > 2000:  # Threshold for summarization
                content = self.summarize_content(content)
            return url, content
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return url, ""

    def process_web_search(self, query: str, num_results: int = 5):
        try:
            search_results = self.search.run(query)
            urls = []
            
            if isinstance(search_results, str):
                entries = search_results.split('link: ')
                urls = [entry.split('\n')[0].split(',')[0].strip() 
                       for entry in entries[1:] 
                       if entry.strip()][:num_results]
            
            # Use asyncio to fetch URLs concurrently
            results = asyncio.run(self.process_urls_async(urls))
            
            # Filter out empty results and combine content
            content = []
            for url, page_content in results:
                if page_content:
                    if len(page_content) > 2000:
                        self.wait_for_rate_limit()  # Rate limit before summarization
                        page_content = self.summarize_content(page_content)
                    content.append(f"Source: {url}\n{page_content}")
            
            # Final summarization of combined content
            combined_content = "\n\n---\n\n".join(content)
            if len(combined_content) > 4000:
                self.wait_for_rate_limit()  # Rate limit before final summarization
                combined_content = self.summarize_content(combined_content)
            
            logger.info(f"Processed {len(urls)} URLs successfully")
            return combined_content, urls
            
        except Exception as e:
            logger.error(f"Error in web search processing: {str(e)}", exc_info=True)
            return None, None

    def append_to_conversation(self, role, content):
        if role == 'user':
            self.conversation.append(HumanMessage(content=content))
        elif role == 'assistant':
            self.conversation.append(AIMessage(content=content))
        # Keep only last 10 exchanges (20 messages)
        if len(self.conversation) > 20:
            self.conversation = self.conversation[-20:]

    def chat(self):
        print("Web-enhanced Chat started (type 'exit' to end, 'search: your query' to search web)")
        print("-" * 50)

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == 'exit':
                    print("\nChat ended. Goodbye!")
                    break

                current_urls = None
                if user_input.lower().startswith("search: "):
                    search_query = user_input[8:].strip()
                    print("\nSearching the web...", end=" ")
                    context, urls = self.process_web_search(search_query)
                    current_urls = urls
                    if context and urls:
                        print("\nFound relevant information from:", ", ".join(urls))
                        prompt = f"""Based on the following web search results, please provide a detailed analysis of: {search_query}

                        Web Search Context:
                        {context}

                        Please provide a comprehensive answer that:
                        1. Covers all major points from the sources
                        2. Includes specific details and examples
                        3. Maintains factual accuracy"""
                    else:
                        prompt = user_input
                else:
                    prompt = user_input

                # Create messages list with conversation history
                messages = self.conversation + [HumanMessage(content=prompt)]

                print("\nAssistant:", end=" ", flush=True)
                
                # Modified to use LangChain streaming format
                full_response = ""
                for chunk in self.client.stream(messages):
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                        full_response += chunk.content

                print()  # New line after response

                if full_response:
                    self.append_to_conversation('user', prompt)
                    self.append_to_conversation('assistant', full_response)

                if current_urls:
                    print("\nSources:")
                    for idx, url in enumerate(current_urls, 1):
                        print(f"[{idx}] {url}")
                    print()

            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                print("\nAn error occurred. Please try again.")
                continue

if __name__ == "__main__":
    chat_bot = WebSearchChat()
    chat_bot.chat()
