import os
import json
import uuid
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, g
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Request ID middleware
@app.before_request
def before_request():
    g.request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{g.request_id}] {request.method} {request.path}")

@app.after_request
def after_request(response):
    logger.info(f"[{g.request_id}] Status: {response.status_code}")
    response.headers['X-Request-ID'] = g.request_id
    return response

# Configuration
app.config['ANTHROPIC_API_KEY'] = os.environ.get('ANTHROPIC_API_KEY', '')
app.config['DEFAULT_MODEL'] = os.environ.get('DEFAULT_MODEL', 'claude-3-5-sonnet-20241022')
app.config['MAX_MESSAGES'] = int(os.environ.get('MAX_MESSAGES', 100))  # Max messages per conversation

# Initialize Anthropic client (lazy loading)
_client = None

def get_anthropic_client():
    """Get or create Anthropic client"""
    global _client
    if _client is None and app.config['ANTHROPIC_API_KEY']:
        _client = Anthropic(api_key=app.config['ANTHROPIC_API_KEY'])
    return _client

# In-memory chat storage (per conversation)
conversations = {}
current_conversation_id = 'default'


def get_conversation(conv_id=None):
    """Get or create a conversation"""
    if conv_id is None:
        conv_id = current_conversation_id
    if conv_id not in conversations:
        conversations[conv_id] = {
            'id': conv_id,
            'messages': [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
    return conv_id


@app.route('/health')
def health():
    """Health check endpoint"""
    client = get_anthropic_client()
    api_configured = client is not None and bool(app.config['ANTHROPIC_API_KEY'])
    return jsonify({
        'status': 'healthy',
        'service': 'openclaw',
        'api_configured': api_configured
    })


@app.route('/')
def home():
    """Home page - displays welcome message"""
    return render_template('index.html')


@app.route('/chat')
def chat():
    """Chat interface page"""
    return render_template('chat.html')


@app.route('/chat/<conv_id>')
def chat_conversation(conv_id):
    """Load specific conversation"""
    return render_template('chat.html', conversation_id=conv_id)


@app.route('/api/conversations', methods=['GET'])
def list_conversations():
    """List all conversations"""
    conv_list = []
    for conv_id, conv in conversations.items():
        conv_list.append({
            'id': conv_id,
            'title': conv.get('title', f'Conversation {conv_id}'),
            'message_count': len(conv['messages']),
            'created_at': conv['created_at'],
            'updated_at': conv['updated_at']
        })
    # Sort by updated_at descending
    conv_list.sort(key=lambda x: x['updated_at'], reverse=True)
    return jsonify({'conversations': conv_list})


@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    """Create a new conversation"""
    data = request.get_json() or {}
    conv_id = data.get('id', f'conv_{datetime.now().timestamp()}')
    get_conversation(conv_id)
    return jsonify({'id': conv_id, 'status': 'created'})


@app.route('/api/conversations/<conv_id>', methods=['GET'])
def get_conversation_messages(conv_id):
    """Get messages for a specific conversation"""
    conv = conversations.get(conv_id)
    if not conv:
        return jsonify({'error': 'Conversation not found'}), 404
    return jsonify({
        'id': conv_id,
        'messages': conv['messages'],
        'title': conv.get('title', f'Conversation {conv_id}')
    })


@app.route('/api/conversations/<conv_id>', methods=['DELETE'])
def delete_conversation(conv_id):
    """Delete a conversation"""
    if conv_id in conversations:
        del conversations[conv_id]
        return jsonify({'status': 'deleted'})
    return jsonify({'error': 'Conversation not found'}), 404


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chat messages with Claude AI"""
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    conv_id = data.get('conversation_id', current_conversation_id)
    user_message = data['message']

    # Get or create conversation
    get_conversation(conv_id)
    conv = conversations[conv_id]

    # Add user message to conversation
    user_msg = {
        'role': 'user',
        'content': user_message,
        'timestamp': datetime.now().isoformat()
    }
    conv['messages'].append(user_msg)

    # Check if Claude API is configured
    client = get_anthropic_client()
    if not client or not app.config['ANTHROPIC_API_KEY']:
        # Fallback response when no API key
        response_content = f"You said: {user_message}. This is a demo response. Configure ANTHROPIC_API_KEY to enable AI responses."
    else:
        try:
            # Convert conversation history to Claude format
            claude_messages = []
            for msg in conv['messages'][:-1]:  # Exclude current message
                claude_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

            # Call Claude API
            message = client.messages.create(
                model=app.config['DEFAULT_MODEL'],
                max_tokens=4096,
                messages=[*claude_messages, {'role': 'user', 'content': user_message}]
            )

            response_content = message.content[0].text
        except Exception as e:
            response_content = f"Error: {str(e)}"

    # Add assistant response to conversation
    assistant_msg = {
        'role': 'assistant',
        'content': response_content,
        'timestamp': datetime.now().isoformat()
    }
    conv['messages'].append(assistant_msg)

    # Update conversation timestamp
    conv['updated_at'] = datetime.now().isoformat()

    # Auto-generate title from first message if not set
    if 'title' not in conv:
        conv['title'] = user_message[:50] + ('...' if len(user_message) > 50 else '')

    return jsonify({
        'response': response_content,
        'conversation_id': conv_id,
        'message': assistant_msg
    })


@app.route('/api/chat/stream', methods=['POST'])
def api_chat_stream():
    """Streaming API endpoint for chat messages with Claude AI"""
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    conv_id = data.get('conversation_id', current_conversation_id)
    user_message = data['message']

    # Get or create conversation
    get_conversation(conv_id)
    conv = conversations[conv_id]

    # Add user message to conversation
    user_msg = {
        'role': 'user',
        'content': user_message,
        'timestamp': datetime.now().isoformat()
    }
    conv['messages'].append(user_msg)

    # Check if Claude API is configured
    client = get_anthropic_client()
    if not client or not app.config['ANTHROPIC_API_KEY']:
        # Fallback response
        response_content = f"You said: {user_message}. Configure ANTHROPIC_API_KEY to enable AI responses."
        yield f"data: {json.dumps({'type': 'content', 'content': response_content})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    else:
        try:
            # Convert conversation history to Claude format
            claude_messages = []
            for msg in conv['messages'][:-1]:
                claude_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

            # Stream from Claude API
            with client.messages.stream(
                model=app.config['DEFAULT_MODEL'],
                max_tokens=4096,
                messages=[*claude_messages, {'role': 'user', 'content': user_message}]
            ) as stream:
                full_response = ""
                for text in stream.text_stream:
                    full_response += text
                    yield f"data: {json.dumps({'type': 'content', 'content': text})}\n\n"

                # Add complete response to conversation
                assistant_msg = {
                    'role': 'assistant',
                    'content': full_response,
                    'timestamp': datetime.now().isoformat()
                }
                conv['messages'].append(assistant_msg)
                conv['updated_at'] = datetime.now().isoformat()

                yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    # Auto-generate title if not set
    if 'title' not in conv:
        conv['title'] = user_message[:50] + ('...' if len(user_message) > 50 else '')


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get chat history for current conversation"""
    conv = conversations.get(current_conversation_id, {'messages': []})
    return jsonify({'history': conv['messages']})


@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Clear chat history for current conversation"""
    if current_conversation_id in conversations:
        conversations[current_conversation_id]['messages'] = []
    return jsonify({'message': 'Chat history cleared'})


@app.route('/api/export/<conv_id>', methods=['GET'])
def export_conversation(conv_id):
    """Export conversation as JSON or markdown"""
    format_type = request.args.get('format', 'json')

    conv = conversations.get(conv_id)
    if not conv:
        return jsonify({'error': 'Conversation not found'}), 404

    if format_type == 'markdown':
        md = f"# {conv.get('title', f'Conversation {conv_id}')}\n\n"
        md += f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

        for msg in conv['messages']:
            role = "User" if msg['role'] == 'user' else "Assistant"
            timestamp = msg.get('timestamp', '')
            md += f"## {role} ({timestamp})\n\n{msg['content']}\n\n---\n\n"

        return Response(
            md,
            mimetype='text/markdown',
            headers={'Content-Disposition': f'attachment; filename=conversation_{conv_id}.md'}
        )
    else:
        return Response(
            json.dumps(conv, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename=conversation_{conv_id}.json'}
        )


@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get application settings (non-sensitive)"""
    return jsonify({
        'model': app.config['DEFAULT_MODEL'],
        'api_configured': bool(app.config['ANTHROPIC_API_KEY']),
        'max_messages': app.config['MAX_MESSAGES']
    })


@app.route('/api/conversations/<conv_id>/title', methods=['PUT'])
def update_conversation_title(conv_id):
    """Update conversation title"""
    data = request.get_json() or {}
    new_title = data.get('title')
    
    if not new_title:
        return jsonify({'error': 'Title is required'}), 400
    
    conv = conversations.get(conv_id)
    if not conv:
        return jsonify({'error': 'Conversation not found'}), 404
    
    conv['title'] = new_title
    conv['updated_at'] = datetime.now().isoformat()
    
    return jsonify({'id': conv_id, 'title': new_title, 'status': 'updated'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
