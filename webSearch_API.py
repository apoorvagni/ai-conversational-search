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
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

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

    def simple_summarize(self, text, max_sentences=5):
        """Enhanced summarization using sumy's LSA algorithm"""
        try:
            logger.info(f"Starting summarization of text ({len(text)} characters)")
            
            # Initialize the LSA summarizer
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")
            
            # Get summary sentences
            summary_sentences = summarizer(parser.document, max_sentences)
            summary = " ".join([str(sentence) for sentence in summary_sentences])
            
            logger.info(f"Summarization complete:")
            logger.info(f"- Original length: {len(text)} characters")
            logger.info(f"- Summary length: {len(summary)} characters")
            logger.info(f"- Compression ratio: {(len(summary) / len(text)) * 100:.1f}%")
            logger.info(f"- Sentences in summary: {len(summary_sentences)}")
            
            return summary if summary else text[:1000] + "..."
            
        except Exception as e:
            logger.error(f"Error in sumy summarization: {str(e)}")
            # Fallback to simple length-based truncation
            logger.warning("Falling back to simple truncation")
            return text[:1000] + "..."

    async def fetch_url_async(self, session, url):
        try:
            timeout = aiohttp.ClientTimeout(total=3, connect=1)
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    html = await response.text()
                    if len(html) > 500:
                        return url, html
                return url, None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return url, None

    async def fetch_all_urls_async(self, urls):
        connector = aiohttp.TCPConnector(limit=10)
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(
            headers=self.headers, 
            connector=connector,
            timeout=timeout
        ) as session:
            tasks = [self.fetch_url_async(session, url) for url in urls]
            return await asyncio.gather(*tasks, return_exceptions=True)

    def process_content_parallel(self, url, html):
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        unwanted_elements = [
            'script', 'style', 'nav', 'footer', 'header', 'aside', 
            'form', 'iframe', 'noscript', 'advertisement', 'menu',
            'search', 'banner', 'sidebar', 'skip', 'language'
        ]
        
        for element in soup.find_all(unwanted_elements):
            element.decompose()
        
        # Progressive content extraction
        for container in ['article', 'main', 'div.content', 'div.article']:
            main_content = soup.find(container)
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
                if len(text) > 100:
                    return {
                        "content": text[:4000],
                        "source": url
                    }
        
        # Final fallback to paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50)
        
        return {
            "content": text[:4000] if text else "",
            "source": url
        }

    def process_web_search(self, query: str, num_results: int = 10):
        try:
            logger.info(f"Processing web search for: {query}")
            all_urls = set()
            search_queries = [
                query,
                f"{query} latest",
                f"{query} detailed"
            ]
            
            # Track search attempts and URLs found
            for idx, search_query in enumerate(search_queries[:2]):
                if len(all_urls) >= num_results:
                    break
                    
                logger.info(f"Search attempt {idx + 1} with query: {search_query}")
                search_results = self.search.run(search_query)
                if not isinstance(search_results, str):
                    logger.warning(f"Search attempt {idx + 1} failed: Invalid results")
                    continue

                # Extract URLs from this search attempt
                new_urls = set()
                entries = search_results.split('link: ')
                for entry in entries[1:]:
                    url = entry.split('\n')[0].split(',')[0].strip()
                    if url:
                        new_urls.add(url)
                
                all_urls.update(new_urls)
                logger.info(f"Search attempt {idx + 1} found {len(new_urls)} new URLs. Total unique URLs: {len(all_urls)}")

            urls = list(all_urls)[:num_results * 2]
            logger.info(f"Selected {len(urls)} URLs for processing")
            
            # Fetch and process URLs
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(self.fetch_all_urls_async(urls))
            loop.close()

            # Log fetch results
            valid_results = [(url, html) for url, html in results if html is not None]
            failed_urls = [url for url, html in results if html is None]
            logger.info(f"URL fetching complete:")
            logger.info(f"- Successfully fetched: {len(valid_results)} URLs")
            logger.info(f"- Failed to fetch: {len(failed_urls)} URLs")
            if failed_urls:
                logger.debug(f"Failed URLs: {failed_urls}")

            if not valid_results:
                logger.warning("No valid content found from any URLs")
                return None, None

            # Process content in parallel
            with ThreadPoolExecutor(max_workers=len(valid_results)) as executor:
                processed_results = list(executor.map(
                    lambda x: self.process_content_parallel(*x),
                    valid_results
                ))
            
            # Log content processing results
            processed_results = [r for r in processed_results if r][:num_results]
            logger.info(f"Content processing complete:")
            logger.info(f"- Successfully processed: {len(processed_results)} documents")
            
            if not processed_results:
                logger.warning("No valid content after processing")
                return None, None

            # Log content lengths
            for idx, result in enumerate(processed_results):
                content_length = len(result['content'])
                logger.debug(f"Document {idx + 1} from {result['source']}: {content_length} characters")

            # Combine and summarize content
            combined_content = "\n\n".join(result['content'] for result in processed_results)
            logger.info(f"Combined content length: {len(combined_content)} characters")
            
            summarized_content = self.simple_summarize(combined_content, max_sentences=10)
            logger.info(f"Summarized content length: {len(summarized_content)} characters")
            
            successful_urls = [result['source'] for result in processed_results]
            logger.info(f"Final results:")
            logger.info(f"- URLs used in summary: {len(successful_urls)}")
            logger.debug(f"- Final URLs: {successful_urls}")

            return summarized_content, successful_urls

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
