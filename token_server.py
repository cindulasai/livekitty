"""
LiveKit Token Generation Server  
Generates valid JWT tokens for client connections
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt
import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for browser requests

# LiveKit credentials
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

@app.route('/token', methods=['GET'])
def generate_token():
    """Generate a LiveKit access token"""
    
    # Get parameters from query string
    room_name = request.args.get('room', 'voice-room')
    participant_name = request.args.get('identity', 'user')
    
    # Create JWT token
    token = jwt.encode(
        {
            'exp': int(time.time()) + 86400,  # Token expires in 24 hours
            'iss': LIVEKIT_API_KEY,
            'nbf': int(time.time()) - 10,
            'sub': participant_name,
            'video': {
                'room': room_name,
                'roomJoin': True,
                'canPublish': True,
                'canSubscribe': True,
            },
            'metadata': '',
            'name': participant_name,
        },
        LIVEKIT_API_SECRET,
        algorithm='HS256'
    )
    
    return jsonify({
        'token': token,
        'url': 'ws://localhost:7880'
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Process chat message with Ollama"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Query Ollama
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3:8b")
        
        api_url = f"{ollama_url}/chat/completions"
        payload = {
            "model": ollama_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Give brief, natural answers in 1-2 sentences suitable for speaking."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        response = requests.post(api_url, json=payload, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                ai_response = data['choices'][0]['message']['content'].strip()
                return jsonify({'response': ai_response})
        
        return jsonify({'error': 'Failed to get AI response'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("=" * 70)
    print("🔑 LiveKit Token Server Starting...")
    print("=" * 70)
    print(f"   API Key: {LIVEKIT_API_KEY}")
    print(f"   Running on: http://localhost:5000")
    print(f"   Token endpoint: http://localhost:5000/token")
    print(f"   Chat endpoint: http://localhost:5000/chat")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=False)
