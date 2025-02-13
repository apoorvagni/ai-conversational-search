from flask import Flask, request, jsonify, Response, stream_with_context
from webSearch_API import WebSearchChat
from langchain.schema import HumanMessage
import os
import json

app = Flask(__name__)
chat_bot = None  # Initialize as None at module level

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

        def generate():
            full_response = ""
            for chunk in chat_bot.client.stream(messages):
                if chunk.content:
                    full_response += chunk.content
                    # Send the chunk as a Server-Sent Event
                    yield f"data: {json.dumps({'chunk': chunk.content})}\n\n"
            
            # Update conversation history after full response
            if full_response:
                chat_bot.append_to_conversation('user', prompt)
                chat_bot.append_to_conversation('assistant', full_response)
            
            # Send the sources as the final event
            yield f"data: {json.dumps({'sources': urls if urls else None})}\n\n"

        return Response(
            stream_with_context(generate()),
            content_type='text/event-stream'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Remove or modify this part
if __name__ == '__main__':
    # Only use this for local development
    app.run(debug=True) 