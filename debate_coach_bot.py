import os
import re
import yaml
import torch
import torchaudio
from flask import Flask, request, jsonify, send_file, send_from_directory
from dotenv import load_dotenv
import google.generativeai as genai
from speechbrain.inference import Tacotron2
from speechbrain.inference import HIFIGAN

load_dotenv()

app = Flask(__name__)

# Load configuration
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Load API key from environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel(config['model_name'])

# Initialize chat with system prompt
chat = model.start_chat(history=[])
chat.send_message(config['system_prompt'])

class TTSEngine:
    def __init__(self):
        self.tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="pretrained_models/tts-tacotron2")
        self.hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_models/tts-hifigan")

    def generate_speech(self, text, output_file):
        # Split the text into sentences
        sentences = re.split('(?<=[.!?]) +', text)
        all_waveforms = []

        for sentence in sentences:
            # Generate mel spectrogram
            mel_output, mel_length, alignment = self.tacotron2.encode_text(sentence)
            
            # Generate waveform
            waveforms = self.hifi_gan.decode_batch(mel_output)
            all_waveforms.append(waveforms[0])

        # Concatenate all waveforms
        final_waveform = torch.cat(all_waveforms, dim=1)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the audio
        torchaudio.save(output_file, final_waveform.cpu(), self.tacotron2.hparams.sample_rate)
        return output_file

tts_engine = TTSEngine()

def clean_response(text):
    # Remove meta-text patterns
    patterns = [
        r'Analysis:\s*',
        r'Thoughtful tone Mode:\s*',
        r'\[EMOTION:[^\]]+\]\s*',
        r'Here\'s a detailed analysis:\s*',
        # Add more patterns as needed
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_gemini_response(user_input):
    if not user_input.strip():
        return "Please provide a non-empty message."
    try:
        response = chat.send_message(user_input)
        cleaned_response = clean_response(response.text)
        # Truncate response if it's too long
        if len(cleaned_response) > config['max_response_length']:
            cleaned_response = cleaned_response[:config['max_response_length']] + "..."
        return cleaned_response
    except Exception as e:
        print(f"Error in get_gemini_response: {type(e).__name__}: {str(e)}")
        return f"An error occurred while processing your request: {type(e).__name__}. Please try again."

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
        
        # Generate speech using the TTS module
        audio_file = f"temp_audio_{os.urandom(8).hex()}.wav"
        full_audio_path = os.path.join(os.getcwd(), audio_file)
        tts_engine.generate_speech(response_text, full_audio_path)
        
        return jsonify({
            'response': response_text,
            'audio_url': f'/audio/{audio_file}'
        })
    except Exception as e:
        print(f"Error in chat_endpoint: {e}")
        return jsonify({'error': 'An error occurred processing your request'}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_file(filename, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(debug=True)