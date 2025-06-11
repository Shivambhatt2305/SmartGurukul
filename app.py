from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import base64
import tempfile
import os
import logging
from gtts import gTTS
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import requests
import json
import time
import urllib.parse
import re

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key='AIzaSyCr46nkrI0cmCpYybRg8uMtsnAHzQpb1VM')
model = genai.GenerativeModel('gemini-1.5-flash')

# Global variables for voice management
custom_voice_data = None
voice_samples_dir = "voice_samples"
os.makedirs(voice_samples_dir, exist_ok=True)

# ElevenLabs API configuration (optional - for better voice cloning)
ELEVENLABS_API_KEY = "your_elevenlabs_api_key_here"  # Replace with your API key
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# Google Custom Search API configuration (replace with your keys)
GOOGLE_API_KEY = "AIzaSyCr46nkrI0cmCpYybRg8uMtsnAHzQpb1VM"  # Replace with your Google API key
GOOGLE_CSE_ID = "your_cse_id_here"  # Replace with your Custom Search Engine ID

# Placeholder for ChatGPT API (not implemented)
CHATGPT_API_KEY = "your_chatgpt_api_key_here"  # Replace with your OpenAI API key if used
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def fetch_google_search(query):
    """Fetch search results from Google Custom Search API"""
    try:
        if GOOGLE_API_KEY == "your_google_api_key_here" or GOOGLE_CSE_ID == "your_cse_id_here":
            return None  # Skip if API keys not configured
        
        url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&q={urllib.parse.quote(query)}"
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json().get('items', [])
            snippets = [item.get('snippet', '') for item in results[:3]]  # Get top 3 snippets
            return "\n".join(snippets)
        else:
            logger.warning(f"Google Search API error: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching Google search: {str(e)}")
        return None

def fetch_chatgpt_answer(question):
    """Placeholder for ChatGPT API call (not implemented)"""
    if CHATGPT_API_KEY == "your_chatgpt_api_key_here":
        return None
    
    try:
        headers = {
            "Authorization": f"Bearer {CHATGPT_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 500
        }
        response = requests.post(OPENAI_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            logger.warning(f"ChatGPT API error: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching ChatGPT answer: {str(e)}")
        return None

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        context = data.get('context', '').strip()
        
        if not question:
            return jsonify({'answer': '<p>Please ask a valid question.</p>', 'success': False})

        # Initialize response
        answer = ""
        source = "document"
        
        # Try to answer from document context first if provided
        if context:
            prompt = f"""You are an experienced, patient, and knowledgeable teacher. Your goal is to explain concepts clearly and thoroughly, just like a great educator would in a classroom.

Document Content:
{context}

Student Question: {question}

Instructions for your response:
1. Act as a caring, knowledgeable teacher who wants students to truly understand
2. Provide detailed explanations with examples when possible
3. Break down complex concepts into simpler parts
4. Use analogies and real-world examples to make concepts clearer
5. Structure your response with clear sections if the topic is complex
6. Encourage further learning by suggesting related concepts
7. If the question cannot be fully answered from the document, indicate that clearly
8. Use a warm, encouraging tone that makes learning enjoyable
9. Include step-by-step explanations when appropriate
10. Summarize key points at the end if the explanation is lengthy

Teacher's Response:"""
            
            response = model.generate_content(prompt)
            answer = response.text.strip()
            
            # Check if the response indicates the question wasn't answered from context
            if "not in the document" in answer.lower() or "no information" in answer.lower() or not answer:
                answer = ""
        
        # If no answer from document or no context, fetch from external sources
        if not answer:
            source = "external"
            intermediate_response = {
                'answer': "<p class='note'>This answer is not in our database. I'll fetch it from external sources (Gemini, Google) in a few seconds...</p>",
                'success': True,
                'is_intermediate': True
            }
            logger.info("Fetching answer from external sources")
            
            # Simulate delay for external fetch
            time.sleep(2)
            
            # Try Gemini first
            external_prompt = f"""You are a knowledgeable teacher. Provide a detailed, clear, and engaging explanation for the following question: {question}

Instructions:
1. Explain as if teaching to an intermediate learner
2. Use examples and analogies
3. Break down complex concepts
4. Maintain an encouraging tone
5. Include practical applications
6. Summarize key points

Response:"""
            
            try:
                external_response = model.generate_content(external_prompt)
                answer = external_response.text.strip()
            except Exception as e:
                logger.warning(f"Gemini fetch failed: {str(e)}")
                answer = ""
            
            # If Gemini fails, try Google search
            if not answer:
                search_results = fetch_google_search(question)
                if search_results:
                    prompt_with_search = f"""You are a knowledgeable teacher. Use the following web search results to provide a detailed, clear, and engaging explanation for the question: {question}

Search Results:
{search_results}

Instructions:
1. Explain as if teaching to an intermediate learner
2. Use examples and analogies
3. Maintain an encouraging tone
4. Summarize key points
5. If search results are insufficient, provide a general explanation based on common knowledge

Response:"""
                    try:
                        search_response = model.generate_content(prompt_with_search)
                        answer = search_response.text.strip()
                    except Exception as e:
                        logger.warning(f"Search-based answer failed: {str(e)}")
            
            # Format the external answer
            if answer:
                answer = format_teacher_response(answer)
                answer = f"<p class='note'>Note: This explanation was fetched from external sources (Gemini and web search) as the question was not covered in the provided document.</p>{answer}"
            else:
                answer = "<p>I'm sorry, I couldn't find a detailed answer from my external sources. Could you please rephrase your question or provide more context?</p>"

        # Format document-based answer if applicable
        if source == "document":
            answer = format_teacher_response(answer)

        if not answer:
            answer = "<p>I'd be happy to help explain this topic! Could you please rephrase your question or provide more specific details about what you'd like to learn?</p>"
        
        logger.info(f"Student Question: {question[:100]}...")
        logger.info(f"Teacher Response: {answer[:100]}...")
        
        return jsonify({
            'answer': answer,
            'success': True,
            'response_type': 'teacher_explanation',
            'source': source
        })
        
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}")
        return jsonify({
            'answer': '<p>I apologize, but I encountered a technical issue while preparing your explanation. Please try asking your question again.</p>',
            'success': False,
            'error': str(e)
        }), 500

def format_teacher_response(response):
    """Format the response to be more teacher-like in HTML"""
    # Split the response into paragraphs
    paragraphs = response.split('\n\n')
    formatted_response = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Handle headings
        if para.startswith('# '):
            formatted_response.append(f"<h1>{para[2:].strip()}</h1>")
        elif para.startswith('## '):
            formatted_response.append(f"<h2>{para[3:].strip()}</h2>")
        elif para.startswith('### '):
            formatted_response.append(f"<h3>{para[4:].strip()}</h3>")
        # Handle lists
        elif para.startswith('- ') or para.startswith('* '):
            items = para.split('\n')
            list_items = []
            for item in items:
                if item.startswith('- ') or item.startswith('* '):
                    list_items.append(f"<li>{item[2:].strip()}</li>")
            formatted_response.append(f"<ul>{''.join(list_items)}</ul>")
        elif re.match(r'^\d+\.\s', para):
            items = para.split('\n')
            list_items = []
            for item in items:
                if re.match(r'^\d+\.\s', item):
                    list_items.append(f"<li>{re.sub(r'^\d+\.\s', '', item).strip()}</li>")
            formatted_response.append(f"<ol>{''.join(list_items)}</ol>")
        else:
            # Handle inline formatting: *text* for emphasis, **text** for strong
            para = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', para)
            para = re.sub(r'\*(.*?)\*', r'<em>\1</em>', para)
            formatted_response.append(f"<p>{para}</p>")

    # Add greeting and conclusion if needed
    result = ''.join(formatted_response)
    if len(response) > 500:
        if not any(greeting.lower() in result.lower()[:100] for greeting in ['great question', 'excellent']):
            result = "<p>Great question! Let me explain this in detail.</p>" + result
        
        if not any(ending.lower() in result.lower()[-200:] for ending in ['hope this helps', 'feel free to ask', 'any questions']):
            result += "<p>I hope this explanation helps you understand the concept better! Feel free to ask if you need clarification on any part.</p>"
    
    return result

@app.route('/upload_voice_sample', methods=['POST'])
def upload_voice_sample():
    """Upload a voice sample for voice cloning"""
    try:
        if 'voice_file' not in request.files:
            return jsonify({'error': 'No voice file provided', 'success': False}), 400
        
        voice_file = request.files['voice_file']
        if voice_file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        filename = f"voice_sample_{len(os.listdir(voice_samples_dir))}.wav"
        filepath = os.path.join(voice_samples_dir, filename)
        
        temp_path = os.path.join(voice_samples_dir, "temp_" + voice_file.filename)
        voice_file.save(temp_path)
        
        audio_data, sample_rate = librosa.load(temp_path, sr=22050)
        sf.write(filepath, audio_data, sample_rate)
        
        os.remove(temp_path)
        
        global custom_voice_data
        custom_voice_data = {
            'filepath': filepath,
            'sample_rate': sample_rate,
            'duration': len(audio_data) / sample_rate
        }
        
        logger.info(f"Voice sample uploaded: {filename}")
        
        return jsonify({
            'message': 'Voice sample uploaded successfully! I will now use this voice for explanations.',
            'success': True,
            'voice_duration': custom_voice_data['duration']
        })
        
    except Exception as e:
        logger.error(f"Error uploading voice sample: {str(e)}")
        return jsonify({
            'error': f'Failed to upload voice sample: {str(e)}',
            'success': False
        }), 500

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """Generate speech with custom or default voice"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        page_number = data.get('page', 0)
        use_custom_voice = data.get('use_custom_voice', True)
        
        if not text:
            return jsonify({'error': 'No text provided', 'success': False}), 400
        
        if len(text) > 5000:
            text = text[:5000] + "... I'll continue with the rest in the next segment."
        
        text = enhance_text_for_speech(text)
        
        audio_data = None
        
        if use_custom_voice and custom_voice_data:
            try:
                audio_data = generate_custom_voice_tts(text)
                logger.info("Used custom voice for TTS")
            except Exception as e:
                logger.warning(f"Custom voice TTS failed: {str(e)}, falling back to default")
        
        if not audio_data and ELEVENLABS_API_KEY != "your_elevenlabs_api_key_here":
            try:
                audio_data = generate_elevenlabs_tts(text)
                logger.info("Used ElevenLabs for TTS")
            except Exception as e:
                logger.warning(f"ElevenLabs TTS failed: {str(e)}, falling back to gTTS")
        
        if not audio_data:
            audio_data = generate_gtts(text)
            logger.info("Used gTTS for TTS")
        
        return jsonify({
            'audio_data': audio_data,
            'page': page_number,
            'success': True,
            'voice_type': 'custom' if (use_custom_voice and custom_voice_data) else 'default'
        })
        
    except Exception as e:
        logger.error(f"Error in TTS endpoint: {str(e)}")
        return jsonify({
            'error': f'TTS generation failed: {str(e)}',
            'success': False
        }), 500

def enhance_text_for_speech(text):
    """Enhance text to sound more natural when spoken by a teacher"""
    text = text.replace('. ', '. ... ')
    text = text.replace('!', '! ... ')
    text = text.replace('?', '? ... ')
    text = text.replace(':', ': ... ')
    
    emphasis_words = ['important', 'key', 'remember', 'note', 'crucial', 'essential']
    for word in emphasis_words:
        text = text.replace(word, f"*{word}*")
    
    return text

def generate_custom_voice_tts(text):
    """Generate TTS using custom voice (simplified implementation)"""
    tts = gTTS(text=text, lang='en', slow=False)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tts.save(tmp_file.name)
        
        audio_data, sr = librosa.load(tmp_file.name)
        
        if custom_voice_data:
            audio_data = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=2)
        
        modified_path = tmp_file.name.replace('.mp3', '_modified.wav')
        sf.write(modified_path, audio_data, sr)
        
        with open(modified_path, 'rb') as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        
        os.unlink(tmp_file.name)
        os.unlink(modified_path)
        
        return audio_base64

def generate_elevenlabs_tts(text):
    """Generate TTS using ElevenLabs API"""
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice - sounds professional
    
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5,
            "style": 0.5,
            "use_speaker_boost": True
        }
    }
    
    response = requests.post(
        f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}",
        json=data,
        headers=headers
    )
    
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    else:
        raise Exception(f"ElevenLabs API error: {response.status_code}")

def generate_gtts(text):
    """Generate TTS using Google Text-to-Speech (fallback)"""
    tts = gTTS(text=text, lang='en', slow=False)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tts.save(tmp_file.name)
        
        with open(tmp_file.name, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        os.unlink(tmp_file.name)
        return audio_data

@app.route('/get_voice_status', methods=['GET'])
def get_voice_status():
    """Get current voice configuration status"""
    return jsonify({
        'has_custom_voice': custom_voice_data is not None,
        'custom_voice_info': custom_voice_data if custom_voice_data else None,
        'elevenlabs_available': ELEVENLABS_API_KEY != "your_elevenlabs_api_key_here",
        'success': True
    })

@app.route('/reset_voice', methods=['POST'])
def reset_voice():
    """Reset to default voice"""
    global custom_voice_data
    
    if custom_voice_data and os.path.exists(custom_voice_data['filepath']):
        os.remove(custom_voice_data['filepath'])
    
    custom_voice_data = None
    
    return jsonify({
        'message': 'Voice reset to default successfully!',
        'success': True
    })

@app.route('/explain_topic', methods=['POST'])
def explain_topic():
    """Dedicated endpoint for detailed topic explanations"""
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        context = data.get('context', '').strip()
        detail_level = data.get('detail_level', 'medium')
        
        if not topic:
            return jsonify({'answer': '<p>Please provide a topic to explain.</p>', 'success': False})
        
        source = "document"
        answer = ""
        
        level_instructions = {
            'basic': "Explain this as if teaching to beginners. Use simple language, basic examples, and avoid technical jargon.",
            'medium': "Provide a comprehensive explanation suitable for intermediate learners. Include examples and some technical details.",
            'advanced': "Give an in-depth, detailed explanation with technical specifics, advanced examples, and theoretical background."
        }
        
        # Try document context first if provided
        if context:
            prompt = f"""You are a master teacher with decades of experience. You have a gift for making complex topics understandable and engaging.

Topic to Explain: {topic}

Context/Document Content: {context}

Teaching Level: {detail_level.title()}
{level_instructions.get(detail_level, level_instructions['medium'])}

Your teaching approach should include:

1. **Introduction**: Start with a warm greeting and overview
2. **Core Explanation**: Break down the topic systematically
3. **Examples**: Provide relevant, relatable examples
4. **Key Points**: Highlight the most important concepts
5. **Practical Applications**: Show how this applies in real life
6. **Common Misconceptions**: Address typical misunderstandings
7. **Summary**: Recap the main points
8. **Encouragement**: End with encouraging words and next steps

Remember to:
- Use analogies and metaphors to clarify difficult concepts
- Ask rhetorical questions to engage the learner
- Provide step-by-step breakdowns for processes
- Connect new information to familiar concepts
- Maintain an encouraging, patient tone throughout
- If the topic cannot be fully answered from the document, indicate that clearly

Begin your detailed explanation:"""
            
            response = model.generate_content(prompt)
            answer = response.text.strip()
            
            if "not in the document" in answer.lower() or "no information" in answer.lower() or not answer:
                answer = ""
        
        # If no answer from document or no context, fetch from external sources
        if not answer:
            source = "external"
            intermediate_response = {
                'answer': "<p class='note'>This topic is not fully covered in our database. I'll fetch a detailed explanation from external sources (Gemini, Google) in a few seconds...</p>",
                'success': True,
                'is_intermediate': True
            }
            logger.info("Fetching topic explanation from external sources")
            
            time.sleep(2)
            
            external_prompt = f"""You are a master teacher. Provide a detailed explanation of {topic} for an {detail_level} level learner.

Instructions:
1. Start with a warm greeting and overview
2. Break down the topic systematically
3. Provide relevant examples
4. Highlight key points
5. Show practical applications
6. Address common misconceptions
7. Summarize main points
8. End with encouraging words
9. Use analogies and a patient tone

Response:"""
            
            try:
                external_response = model.generate_content(external_prompt)
                answer = external_response.text.strip()
            except Exception as e:
                logger.warning(f"Gemini fetch failed: {str(e)}")
                answer = ""
            
            if not answer:
                search_results = fetch_google_search(topic)
                if search_results:
                    prompt_with_search = f"""You are a master teacher. Use the following web search results to provide a detailed explanation of {topic} for an {detail_level} level learner.

Search Results:
{search_results}

Instructions:
1. Start with a warm greeting and overview
2. Break down the topic systematically
3. Provide relevant examples
4. Highlight key points
5. Show practical applications
6. Address common misconceptions
7. Summarize main points
8. End with encouraging words
9. Use analogies and a patient tone

Response:"""
                    try:
                        search_response = model.generate_content(prompt_with_search)
                        answer = search_response.text.strip()
                    except Exception as e:
                        logger.warning(f"Search-based answer failed: {str(e)}")
            
            if answer:
                answer = format_teacher_response(answer)
                answer = f"<p class='note'>Note: This explanation was fetched from external sources (Gemini and web search) as the topic was not fully covered in the provided document.</p>{answer}"
            else:
                answer = "<p>I'm sorry, I couldn't find a detailed explanation from my external sources. Could you provide more details about what aspect of this topic you'd like to learn?</p>"

        if source == "document":
            answer = format_teacher_response(answer)
        
        if not answer:
            answer = "<p>I'd love to explain this topic for you! Could you provide a bit more detail about what specific aspect you'd like me to focus on?</p>"
        
        return jsonify({
            'answer': answer,
            'success': True,
            'topic': topic,
            'detail_level': detail_level,
            'response_type': 'detailed_explanation',
            'source': source
        })
        
    except Exception as e:
        logger.error(f"Error in explain_topic endpoint: {str(e)}")
        return jsonify({
            'answer': '<p>I apologize, but I encountered an issue while preparing your explanation. Please try again.</p>',
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Teacher Voice Assistant API is running',
        'features': {
            'custom_voice': custom_voice_data is not None,
            'elevenlabs': ELEVENLABS_API_KEY != "your_elevenlabs_api_key_here",
            'external_search': GOOGLE_API_KEY != "your_google_api_key_here"
        }
    })

if __name__ == '__main__':
    print("🎓 Teacher Voice Assistant Starting...")
    print("📚 Features available:")
    print("   - Detailed topic explanations")
    print("   - Custom voice cloning")
    print("   - Multiple TTS options")
    print("   - Educational response formatting")
    print("   - External knowledge fetching from Gemini and Google")
    app.run(debug=True, host='0.0.0.0', port=5000)