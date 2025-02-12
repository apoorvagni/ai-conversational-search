from langchain_community.tools import DuckDuckGoSearchResults
from bs4 import BeautifulSoup
import requests
import logging
import os
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import HumanMessage, AIMessage

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
        except Exception as e:
            logger.error(f"Error initializing WebSearchChat: {str(e)}")
            raise e

    def fetch_and_parse_url(self, url):
        """Custom method to fetch and parse URL content"""
        try:
            headers = {
                'User-Agent': os.getenv('USER_AGENT'),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=10)
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

    def process_web_search(self, query: str, num_results: int = 5):
        try:
            logger.info(f"Processing web search for: {query}")
            search_results = self.search.run(query)
            
            # Parse results to extract URLs and snippets
            urls = []
            content = []
            
            if isinstance(search_results, str):
                entries = search_results.split('link: ')
                
                for entry in entries[1:]:  # Skip first empty entry
                    lines = entry.split('\n')
                    url = lines[0].split(',')[0].strip()
                    
                    if url and len(urls) < num_results:  # Add this condition
                        urls.append(url)
                        try:
                            # Directly fetch and add content
                            logger.info(f"Fetching content from URL: {url}")
                            page_content = self.fetch_and_parse_url(url)
                            if page_content:
                                logger.info(f"Successfully fetched content from {url}")
                                content.append(f"Source: {url}\n{page_content}")
                            else:
                                logger.warning(f"No content retrieved from {url}")
                                
                            # Add the snippet too
                            prev_entry = entries[entries.index(entry) - 1]
                            if 'snippet: ' in prev_entry:
                                snippet = prev_entry.split('snippet: ')[1].split(',')[0].strip()
                                content.append(f"Snippet from {url}:\n{snippet}")
                                
                        except Exception as e:
                            logger.error(f"Error processing URL {url}: {str(e)}")
                            continue

            if not content:
                logger.warning("No content found from search results")
                return None, None
            
            # Join all content with separators
            context = "\n\n---\n\n".join(content)
            logger.info(f"Final context length: {len(context)} chars")
            logger.info(f"Final URLs collected: {urls}")
            
            return context, urls
            
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
