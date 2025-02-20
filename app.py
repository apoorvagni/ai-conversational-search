from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from webSearch_API import WebSearchChat
from langchain.schema import HumanMessage, SystemMessage
import os
import json
from functools import lru_cache
from flask_compress import Compress
import time
import logging
from datetime import datetime, timedelta
import uuid

app = Flask(__name__)
# Simple wildcard CORS setup
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["POST", "OPTIONS", "GET"],
        "allow_headers": "*",
        "expose_headers": "*"
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

class ChatSession:
    def __init__(self):
        self.chat_bot = None
        self.last_activity = datetime.now()

    def get_or_create_bot(self):
        if self.chat_bot is None:
            self.chat_bot = WebSearchChat()
        return self.chat_bot

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.session_timeout = 3600  # 1 hour timeout

    def get_session(self, client_id):
        # Clean up old sessions periodically
        if len(self.sessions) > 0:  # Only clean if there are sessions
            self._cleanup_old_sessions()
        
        if client_id not in self.sessions:
            self.sessions[client_id] = ChatSession()
        self.sessions[client_id].last_activity = datetime.now()
        return self.sessions[client_id]

    def _cleanup_old_sessions(self):
        current_time = datetime.now()
        expired_sessions = [
            cid for cid, session in self.sessions.items()
            if current_time - session.last_activity > timedelta(seconds=self.session_timeout)
        ]
        for cid in expired_sessions:
            if self.sessions[cid].chat_bot:
                self.sessions[cid].chat_bot.session.close()
            del self.sessions[cid]

# Initialize the session manager at module level
session_manager = SessionManager()

@app.route('/api/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({'status': 'healthy'})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get('message', '')
        client_id = request.headers.get('X-Client-ID')
        
        if not client_id:
            client_id = str(uuid.uuid4())
            
        if not user_input:
            return jsonify({'error': 'No message provided', 'client_id': client_id}), 400

        # Get the session-specific chat bot
        chat_session = session_manager.get_session(client_id)
        chat_bot = chat_session.get_or_create_bot()
        
        # Log the classification result
        needs_search = chat_bot.classify_query_needs_search(user_input)
        logger.info(f"Query '{user_input}' needs search: {needs_search}")

        # Handle web search if needed
        urls = None
        if needs_search:
            search_query = user_input
            context, urls = chat_bot.process_web_search(search_query)
            if context and urls:
                prompt = f"""**Role**: Expert search assistant specializing in concise yet comprehensive answers.  
                    **User Query**: "{search_query}"

                    **Context** (max 3 most relevant points):  
                    {context}

                    **Response Framework**:  
                    1. 🎯 **Direct Answer**: First paragraph answers query directly (<75 words)  
                    2. 🔍 **Detail Control**:  
                    - Default: 3-5 key bullet points with essential information  
                    - If user says "detailed", "expand", or "explain": Use ## subheadings + paragraphs  
                    3. 🔗 **Context Linking**:  
                    - {'Acknowledge connection to previous query' if context else 'No context reference'}  
                    - Highlight contradictions with: "Update: New data shows..."  
                    4. ✨ **Format**:  
                    - Always use markdown  
                    - Prioritize: bold key terms, lists, tables when applicable  
                    - Add "📌 Need more details? Ask to 'expand' any point" at end"""
            else:
                prompt = user_input
        else:
            prompt = user_input

        # Always include last 3 exchanges (6 messages) from conversation history
        recent_conversation = chat_bot.conversation[-6:] if len(chat_bot.conversation) > 6 else chat_bot.conversation
        
        # Create messages with explicit system message and context
        messages = [
            SystemMessage(content="""You are a helpful assistant with these requirements:
            1. For follow-up questions about a topic, reference information from previous messages
            2. When new information conflicts with previous context, prioritize the new information
            3. Stay focused on the current query while incorporating relevant past context
            4. If unsure about connecting previous context, focus solely on the current query"""),
            *recent_conversation,
            HumanMessage(content=prompt)
        ]

        def generate():
            try:
                response_content = ""
                for chunk in chat_bot.client.stream(messages):
                    if chunk and chunk.content:
                        yield f"data: {json.dumps({'chunk': chunk.content, 'client_id': client_id})}\n\n"
                        response_content += chunk.content
                
                if response_content:
                    chat_bot.append_to_conversation('user', prompt)
                    chat_bot.append_to_conversation('assistant', response_content)
                
                yield f"data: {json.dumps({'sources': urls if urls else None, 'client_id': client_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e), 'client_id': client_id})}\n\n"

        response = Response(
            stream_with_context(generate()),
            content_type='text/event-stream'
        )
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        return response

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'client_id': client_id}), 500

@app.teardown_appcontext
def cleanup(error):
    global chat_bot
    if chat_bot and hasattr(chat_bot, 'session'):
        chat_bot.session.close()

# Remove or modify this part
if __name__ == '__main__':
    # Only use this for local development
    app.run(debug=True) 