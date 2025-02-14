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
    
    # Initialize chat_bot only when needed for chat endpoint
    if chat_bot is None:
        try:
            chat_bot = WebSearchChat()
        except ValueError as e:
            return jsonify({'error': str(e)}), 500

    try:
        data = request.json
        user_input = data.get('message', '')
        
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        # Handle web search queries
        if user_input.lower().startswith("search: "):
            search_query = user_input[8:].strip()
            context, urls = chat_bot.process_web_search(search_query)
            
            if context and urls:
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
            urls = None

        messages = chat_bot.conversation + [HumanMessage(content=prompt)]

        # Add this to the chat() function before processing
        if len(chat_bot.conversation) > 40:
            chat_bot.conversation = chat_bot.conversation[-10:]  # Keep only last 5 exchanges

        def generate():
            buffer = ""
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    # Remove timeout parameter from stream call
                    for chunk in chat_bot.client.stream(messages):
                        if chunk and chunk.content:
                            buffer += chunk.content
                            if len(buffer) >= 500:
                                yield f"data: {json.dumps({'chunk': buffer})}\n\n"
                                buffer = ""
                    break  # Success, exit the retry loop
                except Exception as e:
                    logger.error(f"Streaming error (attempt {retry_count + 1}/{max_retries}): {str(e)}")
                    retry_count += 1
                    if retry_count == max_retries:
                        error_msg = f"Failed after {max_retries} attempts: {str(e)}"
                        logger.error(error_msg)
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        return
                    time.sleep(1)
                    continue
            
            # Send any remaining buffer
            if buffer:
                yield f"data: {json.dumps({'chunk': buffer})}\n\n"
            
            # Update conversation and send sources
            if buffer:
                chat_bot.append_to_conversation('user', prompt)
                chat_bot.append_to_conversation('assistant', buffer)
            
            yield f"data: {json.dumps({'sources': urls if urls else None})}\n\n"

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