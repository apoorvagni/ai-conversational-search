# AI-Powered Conversational Search Engine

An intelligent search engine that combines real-time web search with AI-generated conversational responses, delivering contextual answers with source attribution.

## Overview

This project implements a modern search experience that goes beyond traditional keyword matching. It fetches real-time information from the web, processes and summarizes content from multiple sources, and generates natural language responses through AI, creating an intuitive conversational interface.

## Key Features

- **Intelligent Web Search**: Performs multi-query searches across the web using DuckDuckGo API
- **AI-Powered Responses**: Leverages Mistral AI to generate contextual, conversational answers
- **Content Synthesis**: Automatically scrapes, processes, and summarizes content from multiple sources
- **Adaptive Explanations**: Supports multiple response modes (standard, simplified, detailed)
- **Session Management**: Maintains conversation context for natural follow-up interactions
- **Real-Time Streaming**: Provides progressive response updates for better user experience
- **Source Attribution**: Returns cited sources for transparency and verification

## Technical Highlights

### Architecture
- **Backend**: Flask-based REST API with async content fetching
- **AI Integration**: LangChain framework with Mistral AI language model
- **Content Processing**: Advanced NLP with LSA-based summarization (sumy library)
- **Web Scraping**: BeautifulSoup for intelligent content extraction
- **Parallel Processing**: Asyncio and ThreadPoolExecutor for concurrent operations

### Key Technologies
- Python, Flask, LangChain
- Mistral AI (Large Language Model)
- DuckDuckGo Search API
- Natural Language Processing (NLTK, sumy)
- Async/Await patterns for performance optimization

### Deployment Ready
- Docker containerization with AWS Lambda support
- EC2 deployment with Gunicorn
- Vercel serverless deployment
- CORS-enabled for cross-origin requests

## How It Works

1. **Query Reception**: User submits a question through the API
2. **Web Search**: System performs targeted searches and fetches relevant content
3. **Content Processing**: Scrapes and extracts main content from multiple URLs
4. **Summarization**: Uses LSA algorithm to distill key information
5. **AI Generation**: Mistral AI generates a conversational response based on summarized context
6. **Response Delivery**: Streams response with source citations back to the user

## API Endpoints

### `/api/chat` (POST)
Submit queries and receive AI-generated responses with web context

**Parameters:**
- `message`: User query
- `simplify=true`: Get simplified explanations
- `detailed=true`: Get in-depth analysis

**Response:**
- Streaming text response
- Source URLs
- Session management via `X-Client-ID` header

### `/api/healthcheck` (GET)
Health status endpoint for monitoring

## Use Cases

- Research assistance with real-time information
- Educational content with adaptive complexity
- Technical documentation queries
- Current events and news analysis
- General knowledge questions with source verification

## Getting Started

See [setup.md](setup.md) for detailed deployment instructions including Docker, EC2, and local development setup.

## Environment Variables

```bash
MISTRAL_API_KEY=your_mistral_api_key
USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
```

---

**Note**: This project demonstrates the integration of modern AI capabilities with traditional search engines to create a more intuitive and informative search experience.
