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
        
        # Get explanation mode from query parameters
        explain_like_5 = request.args.get('simplify', 'false').lower() == 'true'
        in_depth = request.args.get('detailed', 'false').lower() == 'true'
        
        if not client_id:
            client_id = str(uuid.uuid4())
            
        if not user_input:
            return jsonify({'error': 'No message provided', 'client_id': client_id}), 400

        # Get the session-specific chat bot
        chat_session = session_manager.get_session(client_id)
        chat_bot = chat_session.get_or_create_bot()

        # Process the request
        return process_chat_request(chat_bot, user_input, client_id, explain_like_5, in_depth)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'client_id': client_id if 'client_id' in locals() else None}), 500

def process_chat_request(chat_bot, user_input, client_id, explain_like_5=False, in_depth=False):
    """Process a chat request with optional explanation modes."""
    try:
        # Perform web search
        context, urls = chat_bot.process_web_search(user_input)
        
        # Generate the appropriate prompt based on explanation mode
        prompt = generate_prompt(user_input, context, explain_like_5, in_depth)
        
        # Get conversation history
        recent_conversation = chat_bot.conversation[-6:] if len(chat_bot.conversation) > 6 else chat_bot.conversation
        
        # Create messages with system message and context
        messages = create_messages(recent_conversation, prompt, explain_like_5, in_depth)
        
        # Generate and return the streaming response
        return create_streaming_response(chat_bot, messages, prompt, client_id, urls)
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise

def generate_prompt(user_input, context, explain_like_5=False, in_depth=False):
    """Generate an appropriate prompt based on the explanation mode."""
    if not context:
        # If no context, just use the user input with appropriate instruction
        if explain_like_5:
            return f"""Explain this to me as if I'm 5 years old: {user_input}
            
            Always use markdown formatting in your response."""
        elif in_depth:
            return f"""Provide a detailed, in-depth explanation of: {user_input}
            
            Always use markdown formatting in your response."""
        else:
            return f"""{user_input}
            
            Always use markdown formatting in your response."""
    
    # If we have context, create a more detailed prompt
    if explain_like_5:
        return f"""**Role**: Friendly educator who explains complex topics.
            **User Query**: "{user_input}"

            **Context**:  
            {context}

            **Response Framework**:
            1. Use simple words and short sentences a 5-year-old would understand
            2. Use a warm, encouraging tone
            3. ✨ **Format**:  
               - Always use markdown
               - Use emoji where appropriate"""
    elif in_depth:
        return f"""**Role**: Comprehensive analysis of the topic.  
            **User Query**: "{user_input}"

            **Context**:  
            {context}

            **Response Framework**:
            1. Begin with an overview of the topic
            2. Explain in detail and address limitations, challenges, or open questions
            3. ✨ **Format**:  
               - Always use markdown"""
    else:
        return f"""**Role**: Expert search assistant specializing in concise yet comprehensive answers.  
            **User Query**: "{user_input}"

            **Context**:  
            {context}

            **Response Framework**:
            1. Begin with a clear, concise answer with essential information
            2. Provide 3-5 key supporting points
            3. If there are connections to previous queries, acknowledge them naturally in your response
            4. ✨ **Format**:  
            - Always use markdown  
            - Prioritize: bold key terms, lists, tables when applicable"""

def create_messages(recent_conversation, prompt, explain_like_5=False, in_depth=False):
    """Create the message list for the chat model."""
    # Create a system message based on the explanation mode
    if explain_like_5:
        system_content = """You are a helpful assistant that explains complex topics in simple terms that a 5-year-old child would understand.
        1. End with humorous and engaging language
        2. Always use markdown formatting"""
    elif in_depth:
        system_content = """You are an assistant that provides detailed explanations.
        1. Explain complex concepts in detail with proper terminology
        2. Always use markdown formatting"""
    else:
        system_content = """You are a helpful assistant with these requirements:
        1. For follow-up questions about a topic, reference information from previous messages
        2. When new information conflicts with previous context, prioritize the new information
        3. Stay focused on the current query while incorporating relevant past context
        4. If unsure about connecting previous context, focus solely on the current query
        5. Always use markdown formatting"""
    
    return [
        SystemMessage(content=system_content),
        *recent_conversation,
        HumanMessage(content=prompt)
    ]

def create_streaming_response(chat_bot, messages, prompt, client_id, urls):
    """Create and return a streaming response."""
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
            
            # Just include sources without explanation options flag
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

@app.teardown_appcontext
def cleanup(error):
    global chat_bot
    if chat_bot and hasattr(chat_bot, 'session'):
        chat_bot.session.close()

# Remove or modify this part
if __name__ == '__main__':
    # Only use this for local development
    app.run(debug=True) 