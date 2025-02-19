from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from webSearch_API import WebSearchChat
from langchain.schema import HumanMessage
import os
import json
from functools import lru_cache
from flask_compress import Compress
import time
import logging

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["POST", "GET", "OPTIONS"],
        "allow_headers": "*",
        "expose_headers": "*",
    }
})

# Add these near the top after creating the Flask app
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['JSON_SORT_KEYS'] = False  # Reduces JSON processing overhead

# Add after creating Flask app
Compress(app)

chat_bot = None  # Initialize as None at module level

headers = {
    'User-Agent': os.getenv('USER_AGENT'),
    'Accept': 'text/html,*/*;q=0.9',
    'Accept-Encoding': 'gzip',
}

logger = logging.getLogger(__name__)

@app.route('/api/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({'status': 'healthy'})

@app.route('/api/chat', methods=['POST'])
def chat():
    global chat_bot
    
    if chat_bot is None:
        try:
            chat_bot = WebSearchChat()
        except ValueError as e:
            logger.error(f"Failed to initialize chat bot: {str(e)}")
            return jsonify({'error': str(e)}), 500

    try:
        data = request.json
        user_input = data.get('message', '')
        
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        # Log the classification result
        needs_search = chat_bot.classify_query_needs_search(user_input)
        logger.info(f"Query '{user_input}' needs search: {needs_search}")

        # Handle web search if needed
        urls = None
        if needs_search:
            search_query = user_input  # Use the original query for search
            context, urls = chat_bot.process_web_search(search_query)
            logger.info(f"Web search results - Context found: {bool(context)}, URLs found: {bool(urls)}")
            if context and urls:
                logger.info(f"Using web search context with {len(urls)} sources")

                prompt = f"""Craft a detailed response to: {search_query}
                
                Context:
                {context}
                
                Response requirements:
                - First paragraph: Direct, comprehensive answer
                - Subsequent sections: Break down key aspects using subheadings
                - Include relevant details
                - Format the entire response in Markdown for proper headings, lists and emphasis"""
            else:
                logger.warning("Web search requested but no results found")
                prompt = user_input
        else:
            logger.info("Using direct query without web search")
            prompt = user_input

        # Always include last 3 exchanges (6 messages) from conversation history
        recent_conversation = chat_bot.conversation[-6:] if len(chat_bot.conversation) > 6 else chat_bot.conversation
        messages = recent_conversation + [HumanMessage(content=prompt)]

        def generate():
            try:
                response_content = ""
                for chunk in chat_bot.client.stream(messages):
                    if chunk and chunk.content:
                        yield f"data: {json.dumps({'chunk': chunk.content})}\n\n"
                        response_content += chunk.content
                
                if response_content:
                    chat_bot.append_to_conversation('user', prompt)
                    chat_bot.append_to_conversation('assistant', response_content)
                
                yield f"data: {json.dumps({'sources': urls if urls else None})}\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            content_type='text/event-stream'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.teardown_appcontext
def cleanup(error):
    global chat_bot
    if chat_bot and hasattr(chat_bot, 'session'):
        chat_bot.session.close()

# Remove or modify this part
if __name__ == '__main__':
    # Only use this for local development
    app.run(debug=True) 