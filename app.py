from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# In-memory chat history
chat_history = []


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'openclaw'})


@app.route('/')
def home():
    """Home page - displays welcome message"""
    return render_template('index.html')


@app.route('/chat')
def chat():
    """Chat interface page"""
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chat messages"""
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    user_message = data['message']

    # Add user message to history
    chat_history.append({'role': 'user', 'content': user_message})

    # Simple echo response (replace with AI logic as needed)
    response = f"You said: {user_message}. I'm your personal AI assistant!"

    # Add assistant response to history
    chat_history.append({'role': 'assistant', 'content': response})

    return jsonify({
        'response': response,
        'history': chat_history
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get chat history"""
    return jsonify({'history': chat_history})


@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Clear chat history"""
    chat_history.clear()
    return jsonify({'message': 'Chat history cleared'})


if __name__ == '__main__':
    app.run(debug=True)
