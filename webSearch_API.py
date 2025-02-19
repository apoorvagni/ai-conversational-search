from langchain_community.tools import DuckDuckGoSearchResults
from bs4 import BeautifulSoup
import requests
import logging
import os
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import partial
from langchain.prompts import PromptTemplate
import nltk
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchChat:
    # Mistral AI model list: https://docs.mistral.ai/getting-started/models/models_overview/
    # Smallest Free model: mistral-small
    # Largest Free model: open-mistral-nemo
    MODEL_NAME = "mistral-small-latest"
    
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
            
            # Download only the absolute minimum NLTK data needed
            try:
                # Create tmp directory if it doesn't exist
                os.makedirs('/tmp/nltk_data', exist_ok=True)
                
                # Download the complete punkt tokenizer
                nltk.download('punkt', download_dir='/tmp/nltk_data', quiet=True)
                
                # Add the tmp directory to NLTK's data path
                nltk.data.path.append('/tmp/nltk_data')
            except Exception as e:
                logger.error(f"Error downloading NLTK data: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error initializing WebSearchChat: {str(e)}")
            raise e

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

    def simple_summarize(self, text, max_sentences=5):
        """Simple summarization without heavy dependencies"""
        # Split into sentences using nltk's punkt tokenizer
        try:
            sentences = nltk.sent_tokenize(text)
            # Return first max_sentences sentences
            return " ".join(sentences[:max_sentences])
        except Exception as e:
            logger.error(f"Error in simple summarization: {str(e)}")
            # Fallback to simple length-based truncation
            return text[:1000] + "..."

    async def fetch_url_async(self, session, url):
        try:
            async with session.get(url, timeout=3) as response:  # Reduced timeout to 3 seconds
                if response.status == 200:
                    html = await response.text()
                    # Try to parse the content immediately to check if it's extractable
                    soup = BeautifulSoup(html, 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                    if text and len(text) > 100:  # Basic validation of content
                        return url, html
                    else:
                        logger.warning(f"Skipping {url} - insufficient content")
                        return url, None
                else:
                    logger.warning(f"Failed to fetch {url} - Status: {response.status}")
                    return url, None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return url, None

    async def fetch_all_urls_async(self, urls):
        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = [self.fetch_url_async(session, url) for url in urls]
            return await asyncio.gather(*tasks)

    def process_content_parallel(self, url, html):
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove all unwanted elements
        unwanted_elements = [
            'script', 'style', 'nav', 'footer', 'header', 'aside', 
            'form', 'iframe', 'noscript', 'advertisement', 'menu',
            'search', 'banner', 'sidebar', 'skip', 'language'
        ]
        
        for element in soup.find_all(unwanted_elements):
            element.decompose()
        
        # Remove elements with common advertising/navigation class names
        for element in soup.find_all(class_=lambda x: x and any(word in str(x).lower() for word in [
            'ad', 'ads', 'advertisement', 'banner', 'menu', 'nav', 'sidebar', 'footer', 'header',
            'social', 'sharing', 'newsletter', 'subscription', 'cookie', 'popup'
        ])):
            element.decompose()

        # First try to find the main article content
        article = soup.find(['article', 'main'])
        if article:
            text = article.get_text(separator=' ', strip=True)
        else:
            # Fallback to finding content in paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50)

        # Clean up the text
        text = ' '.join(text.split())  # Remove extra whitespace
        text = text[:2000]  # Limit length
        
        # Create a structured result
        result = {
            "content": text,
            "source": url
        }
        
        return result

    def process_web_search(self, query: str, num_results: int = 5):
        try:
            logger.info(f"Processing web search for: {query}")
            search_results = self.search.run(query)
            
            if not isinstance(search_results, str):
                return None, None

            # Extract URLs
            entries = search_results.split('link: ')
            urls = []
            for entry in entries[1:]:
                url = entry.split('\n')[0].split(',')[0].strip()
                if url and len(urls) < num_results:
                    urls.append(url)

            # Fetch and process URLs
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(self.fetch_all_urls_async(urls))
            loop.close()

            # Process only successful results
            valid_results = [(url, html) for url, html in results if html is not None]
            
            if not valid_results:
                logger.warning("No valid content found from any URLs")
                return None, None

            # Process content in parallel
            with ThreadPoolExecutor(max_workers=len(valid_results)) as executor:
                processed_results = list(executor.map(
                    lambda x: self.process_content_parallel(*x),
                    valid_results
                ))
            
            # Filter out None results
            processed_results = [r for r in processed_results if r]
            if not processed_results:
                return None, None

            # Format the context by combining only the content
            context = "\n\n".join(result['content'] for result in processed_results)
            successful_urls = [result['source'] for result in processed_results]

            logger.info(f"Found {len(successful_urls)} valid URLs")
            logger.info(f"Processed {len(processed_results)} results")

            return context, successful_urls

        except Exception as e:
            logger.error(f"Error in web search processing: {str(e)}", exc_info=True)
            return None, None

    def append_to_conversation(self, role, content):
        if role == 'user':
            self.conversation.append(HumanMessage(content=content))
        elif role == 'assistant':
            self.conversation.append(AIMessage(content=content))
        # Keep only last 3 exchanges (6 messages)
        if len(self.conversation) > 6:
            self.conversation = self.conversation[-6:]

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
                messages = [
                    SystemMessage(content="You are a helpful assistant. Use previous messages for context when relevant."),
                    *self.conversation,
                    HumanMessage(content=prompt)
                ]

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
    
    # Classify if the query needs web search
    def classify_query_needs_search(self, query: str) -> bool:
        """
        Determines if a query needs web search.
        Returns True if web search is needed, False otherwise.
        """
        classification_prompt = f"""Quickly determine if this query requires current or factual information from the web. 
        Query: "{query}"
        
        Reply with just 'YES' or 'NO'. Consider:
        - Questions about current events, news, or facts need web search
        - General chat, opinions, or coding questions don't need search
        - Questions about specific products, people, or events need search
        """
        
        try:
            classification_messages = [HumanMessage(content=classification_prompt)]
            response = self.client.invoke(classification_messages).content.strip().upper()
            # Add a small delay to respect rate limits
            time.sleep(1.1)  # Wait 1.1 seconds before next API call
            logger.info(f"Query classification for '{query}': {response}")
            return 'YES' in response  # More flexible check for "YES" in the response
        except Exception as e:
            logger.error(f"Error in query classification: {str(e)}")
            return False

if __name__ == "__main__":
    chat_bot = WebSearchChat()
    chat_bot.chat()
