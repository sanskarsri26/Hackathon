import os
import re
from google.ai import generativelanguage_v1beta as glm
from google.api_core import client_options as client_options_lib
from flask import Flask, request, jsonify, send_file, send_from_directory
from gtts import gTTS
import tempfile
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# Load API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client_options = client_options_lib.ClientOptions(
    api_endpoint="generativelanguage.googleapis.com",
    api_key=api_key
)
client = glm.GenerativeServiceClient(client_options=client_options)

system_prompt = """
You are an AI debate coach designed to help users prepare for debates, interviews, and Model United Nations (MUNs). Your tasks include:

1. Cross-questioning: Ask probing questions to challenge the user's arguments and help them think critically.
2. Identifying fallacies: Point out logical fallacies in the user's statements and explain how to avoid them.
3. Improving speaking style: Provide feedback on the user's language, tone, and delivery to enhance their public speaking skills.
4. Offering constructive feedback: Give specific suggestions for improvement while maintaining a supportive and encouraging tone.
5. Simulating different scenarios: Act as various debate opponents or interviewers to help the user practice for different situations.

Adapt your responses based on the user's skill level and the specific type of event they're preparing for (debate, interview, or MUN).

Important: Provide your responses in a conversational tone, without using markdown formatting, asterisks, or other special symbols. Use natural language and avoid unnecessary headings or list markers. Your response should be fluent and easy to read aloud.
"""

conversation_history = [
    glm.Content(parts=[glm.Part(text=system_prompt)], role="user"),
    glm.Content(parts=[glm.Part(text="Understood. I'm ready to help you prepare for your debate. What specific topic or aspect would you like to focus on?")], role="model")
]

def clean_response(text):
    # Remove markdown headers
    text = re.sub(r'#{1,6}\s?', '', text)
    # Remove asterisks and underscores used for emphasis
    text = re.sub(r'[*_]{1,2}', '', text)
    # Remove bullet points and numbered lists
    text = re.sub(r'^\s*[-*+]\s|\d+\.\s', '', text, flags=re.MULTILINE)
    # Remove extra newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove any remaining special characters
    text = re.sub(r'[^\w\s.,?!]', '', text)
    return text.strip()

def get_gemini_response(user_input):
    global conversation_history
    if not user_input.strip():
        return "Please provide a non-empty message."
    try:
        conversation_history.append(glm.Content(parts=[glm.Part(text=user_input)], role="user"))
        request = glm.GenerateContentRequest(
            model="models/gemini-pro",
            contents=conversation_history,
        )
        response = client.generate_content(request)
        raw_response = response.candidates[0].content.parts[0].text
        cleaned_response = clean_response(raw_response)
        conversation_history.append(glm.Content(parts=[glm.Part(text=cleaned_response)], role="model"))
        return cleaned_response
    except Exception as e:
        print(f"Error in get_gemini_response: {type(e).__name__}: {str(e)}")
        return f"An error occurred while processing your request: {type(e).__name__}. Please try again."

def text_to_speech_gtts(text):
    tts = gTTS(text=text, lang='en')
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"speech_{os.urandom(8).hex()}.mp3")
    tts.save(temp_file)
    return temp_file

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_input = request.json.get('message', '')
    if not user_input.strip():
        return jsonify({'error': 'Please provide a non-empty message'}), 400
    
    try:
        response_text = get_gemini_response(user_input)
        audio_file = text_to_speech_gtts(response_text)
        
        return jsonify({
            'response': response_text,
            'audio_url': f'/audio/{os.path.basename(audio_file)}'
        })
    except Exception as e:
        print(f"Error in chat_endpoint: {e}")
        return jsonify({'error': 'An error occurred processing your request'}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    temp_dir = tempfile.gettempdir()
    return send_file(os.path.join(temp_dir, filename), mimetype='audio/mp3')

if __name__ == '__main__':
    app.run(debug=True)