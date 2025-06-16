from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import google.generativeai as genai
import base64
import os
import logging
from gtts import gTTS
from io import BytesIO
import re
import PyPDF2
import tempfile
import json

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# IMPORTANT: Replace with your actual Gemini API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCr46nkrI0cmCpYybRg8uMtsnAHzQpb1VM')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Comprehensive language mapping with proper codes
LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish', 
    'fr': 'French',
    'de': 'German',
    'hi': 'Hindi',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'ru': 'Russian',
    'pt': 'Portuguese',
    'it': 'Italian',
    'ar': 'Arabic',
    'bn': 'Bengali',
    'ur': 'Urdu',
    'tr': 'Turkish',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'pl': 'Polish',
    'cs': 'Czech',
    'sk': 'Slovak',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'bg': 'Bulgarian',
    'hr': 'Croatian',
    'sr': 'Serbian',
    'sl': 'Slovenian',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'uk': 'Ukrainian',
    'be': 'Belarusian',
    'ka': 'Georgian',
    'hy': 'Armenian',
    'az': 'Azerbaijani',
    'kk': 'Kazakh',
    'ky': 'Kyrgyz',
    'uz': 'Uzbek',
    'tg': 'Tajik',
    'mn': 'Mongolian',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
    'ms': 'Malay',
    'tl': 'Filipino',
    'sw': 'Swahili',
    'am': 'Amharic',
    'he': 'Hebrew',
    'fa': 'Persian',
    'ps': 'Pashto',
    'sd': 'Sindhi',
    'ne': 'Nepali',
    'si': 'Sinhala',
    'my': 'Myanmar',
    'km': 'Khmer',
    'lo': 'Lao',
    'mt': 'Maltese',
    'is': 'Icelandic',
    'ga': 'Irish',
    'cy': 'Welsh',
    'eu': 'Basque',
    'ca': 'Catalan',
    'gl': 'Galician',
    'af': 'Afrikaans',
    'sq': 'Albanian',
    'mk': 'Macedonian'
}

# Updated gTTS language mapping with comprehensive support
GTTS_LANGUAGE_MAP = {
    'en': 'en',
    'es': 'es',
    'fr': 'fr', 
    'de': 'de',
    'hi': 'hi',
    'ja': 'ja',
    'ko': 'ko',
    'zh-cn': 'zh',
    'zh-tw': 'zh-tw',
    'ru': 'ru',
    'pt': 'pt',
    'it': 'it',
    'ar': 'ar',
    'bn': 'bn',
    'ur': 'ur',
    'tr': 'tr',
    'nl': 'nl',
    'sv': 'sv',
    'da': 'da',
    'no': 'no',
    'fi': 'fi',
    'pl': 'pl',
    'cs': 'cs',
    'sk': 'sk',
    'hu': 'hu',
    'ro': 'ro',
    'bg': 'bg',
    'hr': 'hr',
    'sr': 'sr',
    'sl': 'sl',
    'et': 'et',
    'lv': 'lv',
    'lt': 'lt',
    'uk': 'uk',
    'ka': 'ka',
    'hy': 'hy',
    'az': 'az',
    'kk': 'kk',
    'ky': 'ky',
    'uz': 'uz',
    'mn': 'mn',
    'th': 'th',
    'vi': 'vi',
    'id': 'id',
    'ms': 'ms',
    'tl': 'tl',
    'sw': 'sw',
    'am': 'am',
    'he': 'iw',  # Hebrew uses 'iw' in gTTS
    'fa': 'fa',
    'ne': 'ne',
    'si': 'si',
    'my': 'my',
    'km': 'km',
    'lo': 'lo',
    'mt': 'mt',
    'is': 'is',
    'ga': 'ga',
    'cy': 'cy',
    'eu': 'eu',
    'ca': 'ca',
    'gl': 'gl',
    'af': 'af',
    'sq': 'sq',
    'mk': 'mk'
}

# --- HELPER FUNCTIONS ---

def is_language_supported(language_code):
    """Check if a language is supported for translation and TTS."""
    return language_code in LANGUAGE_NAMES

def get_language_name(language_code):
    """Get the full name of a language from its code."""
    return LANGUAGE_NAMES.get(language_code, 'Unknown Language')

def translate_text_with_gemini(text, target_language):
    """Translate text using the Gemini API with improved language handling."""
    try:
        if not is_language_supported(target_language):
            logger.error(f"Unsupported language code: {target_language}")
            return text
            
        language_name = get_language_name(target_language)
        
        if target_language == 'en':
            return text  # No translation needed for English
            
        # Enhanced prompt for better translation quality
        prompt = f"""You are a professional translator. Your task is to translate the following text accurately to {language_name}.

CRITICAL INSTRUCTIONS:
1. Translate ONLY to {language_name} - do not include any English text in your response
2. Maintain the original meaning, context, and tone
3. Preserve paragraph structure and formatting
4. Use natural, fluent {language_name} that sounds native
5. Keep technical terms appropriate for the target language
6. If certain terms don't have direct translations, provide the closest cultural equivalent
7. Do NOT add any explanations, notes, or prefixes - respond ONLY with the translated text
8. Ensure the translation is complete and accurate

Text to translate:
---
{text}
---

Provide the complete translation in {language_name}:"""

        response = model.generate_content(prompt)
        if not response.text:
            raise ValueError("Empty response from Gemini API")
        
        translated_text = response.text.strip()
        
        # Clean up any unwanted prefixes that might be added by the AI
        prefixes_to_remove = [
            f"Translation in {language_name}:",
            f"Here is the translation in {language_name}:",
            f"The translation in {language_name} is:",
            f"{language_name} translation:",
            "Translation:",
            "Here is the translation:",
            "The translation is:"
        ]
        
        for prefix in prefixes_to_remove:
            if translated_text.startswith(prefix):
                translated_text = translated_text[len(prefix):].strip()
        
        logger.info(f"Successfully translated {len(text)} characters to {language_name}")
        return translated_text
        
    except Exception as e:
        logger.error(f"Translation error with Gemini for {target_language}: {str(e)}")
        return text  # Return original text as fallback

def extract_pdf_text(pdf_file):
    """Extract all text from an uploaded PDF file with improved error handling."""
    pdf_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            pdf_file.save(temp_file.name)
            pdf_path = temp_file.name
        
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                logger.error("PDF is encrypted and cannot be processed")
                return None
            
            total_pages = len(pdf_reader.pages)
            logger.info(f"Processing PDF with {total_pages} pages")
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                    logger.info(f"Processed page {page_num + 1}/{total_pages}")
                except Exception as page_error:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {str(page_error)}")
                    continue
        
        # Clean up temporary file
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        
        if not text.strip():
            logger.error("No text could be extracted from the PDF")
            return None
            
        logger.info(f"Successfully extracted {len(text)} characters from {pdf_file.filename}")
        return text.strip()
            
    except Exception as e:
        logger.error(f"PDF text extraction error: {str(e)}")
        # Clean up temporary file in case of error
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except:
                pass
        return None

def generate_audio_with_gemini(text, language_code):
    """Generate audio using Gemini API as primary method."""
    try:
        language_name = get_language_name(language_code)
        
        # Use Gemini to generate more natural speech-ready text
        speech_prompt = f"""Convert the following text into natural, speech-friendly {language_name} that sounds good when read aloud:

1. Break long sentences into shorter, more natural phrases
2. Replace complex punctuation with natural pauses
3. Ensure numbers and abbreviations are written out in words
4. Make the text flow naturally for speech synthesis
5. Keep the meaning and content exactly the same
6. Respond ONLY with the speech-ready text in {language_name}

Original text:
---
{text}
---

Speech-ready {language_name} text:"""

        response = model.generate_content(speech_prompt)
        if response.text:
            speech_ready_text = response.text.strip()
            logger.info(f"Generated speech-ready text for {language_name}")
            return generate_audio_for_text(speech_ready_text, language_code)
        else:
            # Fallback to original text
            return generate_audio_for_text(text, language_code)
            
    except Exception as e:
        logger.error(f"Gemini audio preparation error: {str(e)}")
        return generate_audio_for_text(text, language_code)

def generate_audio_for_text(text, language_code):
    """Generate audio for given text in specified language with improved language support."""
    try:
        if not text or not text.strip():
            logger.error("No text provided for audio generation")
            return None
            
        # Clean HTML tags and excessive whitespace
        clean_text = re.sub('<[^<]+?>', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Limit text length for TTS (gTTS has character limits)
        max_length = 4500  # Conservative limit
        if len(clean_text) > max_length:
            clean_text = clean_text[:max_length]
            # Try to cut at sentence boundary
            sentence_endings = ['.', '!', '?', '。', '！', '？']
            best_cut = max_length
            
            for ending in sentence_endings:
                last_ending = clean_text.rfind(ending)
                if last_ending > max_length * 0.8:  # If we can cut at a reasonable sentence boundary
                    best_cut = min(best_cut, last_ending + 1)
            
            if best_cut < max_length:
                clean_text = clean_text[:best_cut]
            else:
                clean_text += "..."
        
        # Get appropriate gTTS language code
        gtts_lang = GTTS_LANGUAGE_MAP.get(language_code)
        if not gtts_lang:
            logger.warning(f"Language code {language_code} not supported by gTTS, falling back to English")
            gtts_lang = 'en'
        
        # Generate TTS audio with error handling
        try:
            # Adjust speech parameters based on language
            slow_speech = language_code in ['ar', 'hi', 'bn', 'ur', 'th', 'my', 'km']  # Languages that benefit from slower speech
            
            tts = gTTS(text=clean_text, lang=gtts_lang, slow=slow_speech)
            audio_file = BytesIO()
            tts.write_to_fp(audio_file)
            audio_file.seek(0)
            audio_data = base64.b64encode(audio_file.getvalue()).decode('utf-8')
            
            logger.info(f"Generated audio for language: {gtts_lang} ({len(clean_text)} characters)")
            return audio_data
            
        except Exception as tts_error:
            logger.error(f"gTTS error for language {gtts_lang}: {str(tts_error)}")
            
            # Fallback to English if the specific language fails
            if gtts_lang != 'en':
                logger.info("Attempting fallback to English TTS")
                try:
                    # Translate to English first for fallback
                    if language_code != 'en':
                        english_text = translate_text_with_gemini(clean_text, 'en')
                    else:
                        english_text = clean_text
                        
                    tts = gTTS(text=english_text, lang='en', slow=False)
                    audio_file = BytesIO()
                    tts.write_to_fp(audio_file)
                    audio_file.seek(0)
                    audio_data = base64.b64encode(audio_file.getvalue()).decode('utf-8')
                    logger.info("Successfully generated English fallback audio")
                    return audio_data
                except Exception as fallback_error:
                    logger.error(f"English fallback TTS also failed: {str(fallback_error)}")
            
            return None
        
    except Exception as e:
        logger.error(f"Audio generation error for language {language_code}: {str(e)}")
        return None

def format_teacher_response(response):
    """Format the AI's response into clean HTML with improved formatting."""
    if not response:
        return "<p>No response available.</p>"
    
    # Split into paragraphs
    paragraphs = response.split('\n\n')
    formatted_response = []
    
    for para in paragraphs:
        para = para.strip()
        if not para: 
            continue
            
        # Handle headers
        if para.startswith('# '): 
            formatted_response.append(f"<h1>{para[2:].strip()}</h1>")
        elif para.startswith('## '): 
            formatted_response.append(f"<h2>{para[3:].strip()}</h2>")
        elif para.startswith('### '): 
            formatted_response.append(f"<h3>{para[4:].strip()}</h3>")
        elif para.startswith('#### '): 
            formatted_response.append(f"<h4>{para[5:].strip()}</h4>")
        
        # Handle lists
        elif para.startswith('- ') or para.startswith('* '):
            items = []
            for line in para.split('\n'):
                line = line.strip()
                if line.startswith('- ') or line.startswith('* '):
                    items.append(f"<li>{line[2:].strip()}</li>")
            if items:
                formatted_response.append(f"<ul>{''.join(items)}</ul>")
        
        # Handle numbered lists - FIXED THIS SECTION
        elif re.match(r'^\d+\.', para):
            items = []
            for line in para.split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    # Fixed: Move the regex operation outside the f-string
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    items.append(f"<li>{cleaned_line}</li>")
            if items:
                formatted_response.append(f"<ol>{''.join(items)}</ol>")
        
        # Handle regular paragraphs
        else:
            # Apply text formatting
            para = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', para)
            para = re.sub(r'\*(.*?)\*', r'<em>\1</em>', para)
            para = re.sub(r'`(.*?)`', r'<code>\1</code>', para)
            formatted_response.append(f"<p>{para}</p>")
    
    return ''.join(formatted_response) if formatted_response else "<p>No content available.</p>"

def generate_multilingual_response(question, context, target_language):
    """Generate a response in the target language with improved prompting."""
    try:
        language_name = get_language_name(target_language)
        
        # Enhanced prompt for better multilingual responses
        prompt = f"""You are an expert multilingual teacher and educational assistant. You MUST respond ENTIRELY in {language_name}.

Document Context:
---
{context}
---

Student's Question: "{question}"

CRITICAL INSTRUCTIONS:
1. Respond COMPLETELY in {language_name} - use NO English words or phrases whatsoever
2. If the answer is found in the document context, explain it thoroughly using that context
3. If the answer is NOT in the document, clearly state this in {language_name} and provide a helpful general explanation
4. Be encouraging, clear, patient, and educational in your tone
5. Use appropriate formatting and structure for better readability
6. Keep the response comprehensive but concise (aim for 2-4 paragraphs)
7. Use natural, fluent {language_name} that sounds like a native speaker
8. Include specific examples or details from the document when relevant
9. End with an encouraging note or invitation for follow-up questions

Your complete educational response in {language_name}:"""

        response = model.generate_content(prompt)
        if not response.text:
            # Fallback response in the target language
            if target_language == 'en':
                return "I apologize, but I couldn't generate a proper response to your question. Please try rephrasing your question or ask about a different aspect of the document."
            else:
                # Generate a simple fallback message
                fallback_prompt = f"Write this message in perfect {language_name}: 'I apologize, but I couldn't generate a proper response to your question. Please try rephrasing your question or ask about a different aspect of the document.'"
                fallback_response = model.generate_content(fallback_prompt)
                return fallback_response.text.strip() if fallback_response.text else "Response generation failed."
        
        answer_text = response.text.strip()
        
        # Clean up any unwanted prefixes
        prefixes_to_remove = [
            f"Response in {language_name}:",
            f"Answer in {language_name}:",
            f"Here is my response in {language_name}:",
            "Response:",
            "Answer:"
        ]
        
        for prefix in prefixes_to_remove:
            if answer_text.startswith(prefix):
                answer_text = answer_text[len(prefix):].strip()
        
        logger.info(f"Generated multilingual response in {language_name}")
        return answer_text
        
    except Exception as e:
        logger.error(f"Error generating multilingual response: {str(e)}")
        # Return error message in target language
        if target_language == 'en':
            return "I encountered a technical issue while processing your question. Please try again or rephrase your question."
        else:
            return "Technical error occurred. Please try again."

# --- API ENDPOINTS ---

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """Handle PDF upload, text extraction, and translation with improved language support."""
    try:
        if 'pdf_file' not in request.files:
            return jsonify({'success': False, 'error': 'No PDF file provided'}), 400
            
        pdf_file = request.files['pdf_file']
        target_language = request.form.get('language', 'en').lower().strip()
        
        if pdf_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Validate language support
        if not is_language_supported(target_language):
            return jsonify({
                'success': False, 
                'error': f'Language "{target_language}" is not supported. Supported languages: {", ".join(LANGUAGE_NAMES.keys())}'
            }), 400
            
        # Extract text from PDF
        original_text = extract_pdf_text(pdf_file)
        if not original_text:
            return jsonify({'success': False, 'error': 'Could not extract text from the PDF. Please ensure the PDF contains readable text and is not encrypted.'}), 500
            
        translated_text = original_text
        audio_data = None
        
        # Translate if not English
        if target_language != 'en':
            logger.info(f"Translating to {get_language_name(target_language)}")
            translated_text = translate_text_with_gemini(original_text, target_language)
            
            # Generate audio for translated text using enhanced method
            audio_data = generate_audio_with_gemini(translated_text, target_language)
            if not audio_data:
                logger.warning(f"Enhanced audio generation failed for {target_language}, trying standard method")
                audio_data = generate_audio_for_text(translated_text, target_language)
        else:
            # Generate English audio
            audio_data = generate_audio_for_text(original_text, 'en')
            
        return jsonify({
            'success': True,
            'original_text': original_text,
            'translated_text': translated_text,
            'language_name': get_language_name(target_language),
            'language_code': target_language,
            'filename': pdf_file.filename,
            'audio_data': audio_data,
            'target_language': target_language,
            'text_length': len(translated_text),
            'has_audio': audio_data is not None,
            'translation_performed': target_language != 'en'
        })
        
    except Exception as e:
        logger.error(f"Error in /upload-pdf endpoint: {str(e)}")
        return jsonify({'success': False, 'error': 'An internal server error occurred. Please try again.'}), 500

@app.route('/ask', methods=['POST'])
def ask():
    """Handle student questions with improved multilingual support."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        context = data.get('context', '').strip()
        target_language = data.get('language', 'en').lower().strip()
        
        if not question:
            error_msg = "Please ask a valid question."
            if target_language != 'en':
                # Try to translate error message
                try:
                    error_msg = translate_text_with_gemini(error_msg, target_language)
                except:
                    pass
            return jsonify({'answer': f'<p>{error_msg}</p>', 'success': False})

        # Validate language support
        if not is_language_supported(target_language):
            return jsonify({
                'answer': f'<p>Language "{target_language}" is not supported.</p>',
                'success': False
            }), 400

        # Generate response in target language
        answer_text = generate_multilingual_response(question, context, target_language)
        formatted_answer = format_teacher_response(answer_text)
        
        # Generate audio for the response using enhanced method
        audio_data = generate_audio_with_gemini(answer_text, target_language)
        if not audio_data:
            logger.warning(f"Enhanced audio generation failed for {target_language}, trying standard method")
            audio_data = generate_audio_for_text(answer_text, target_language)
        
        return jsonify({
            'answer': formatted_answer, 
            'success': True,
            'audio_data': audio_data,
            'language': target_language,
            'language_name': get_language_name(target_language),
            'has_audio': audio_data is not None,
            'question_language': target_language
        })
        
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {str(e)}")
        error_msg = "I apologize, but I encountered a technical issue. Please try again."
        return jsonify({
            'answer': f'<p>{error_msg}</p>',
            'success': False
        }), 500

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """Generate audio from text using gTTS with comprehensive language support."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        language = data.get('language', 'en').lower().strip()

        if not text:
            return jsonify({'success': False, 'error': 'No text provided for TTS'}), 400

        if not is_language_supported(language):
            return jsonify({
                'success': False, 
                'error': f'Language "{language}" is not supported for TTS'
            }), 400

        # Use enhanced audio generation
        audio_data = generate_audio_with_gemini(text, language)
        if not audio_data:
            audio_data = generate_audio_for_text(text, language)
        
        if audio_data:
            return jsonify({
                'success': True, 
                'audio_data': audio_data,
                'language': language,
                'language_name': get_language_name(language),
                'text_length': len(text)
            })
        else:
            return jsonify({
                'success': False, 
                'error': f'Failed to generate audio for {get_language_name(language)}'
            }), 500
            
    except Exception as e:
        logger.error(f"TTS endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate-speech', methods=['POST'])
def generate_speech():
    """Standalone endpoint to generate speech for any text with full language support."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        language = data.get('language', 'en').lower().strip()
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        
        if not is_language_supported(language):
            return jsonify({
                'success': False, 
                'error': f'Language "{language}" is not supported'
            }), 400
            
        # Use enhanced audio generation
        audio_data = generate_audio_with_gemini(text, language)
        if not audio_data:
            audio_data = generate_audio_for_text(text, language)
        
        if audio_data:
            return jsonify({
                'success': True,
                'audio_data': audio_data,
                'language': language,
                'language_name': get_language_name(language),
                'message': f'Speech generated for {get_language_name(language)}',
                'text_length': len(text)
            })
        else:
            return jsonify({
                'success': False, 
                'error': f'Failed to generate speech for {get_language_name(language)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Speech generation error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/supported-languages', methods=['GET'])
def get_supported_languages():
    """Get list of all supported languages."""
    try:
        languages = []
        for code, name in LANGUAGE_NAMES.items():
            languages.append({
                'code': code,
                'name': name,
                'tts_supported': code in GTTS_LANGUAGE_MAP,
                'translation_supported': True  # All languages support translation via Gemini
            })
        
        return jsonify({
            'success': True,
            'languages': languages,
            'total_count': len(languages),
            'tts_count': len(GTTS_LANGUAGE_MAP),
            'translation_count': len(LANGUAGE_NAMES)
        })
        
    except Exception as e:
        logger.error(f"Error getting supported languages: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    """Serve a placeholder favicon to avoid 404 errors."""
    try:
        return send_file(os.path.join(os.path.dirname(__file__), 'static/favicon.ico'))
    except FileNotFoundError:
        # Return a minimal 1x1 transparent PNG as fallback
        return send_file(
            BytesIO(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=')), 
            mimetype='image/png'
        )

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Smart Gurukul Multilingual Translator',
        'supported_languages': len(LANGUAGE_NAMES),
        'tts_languages': len(GTTS_LANGUAGE_MAP),
        'version': '2.0.0',
        'features': [
            'PDF text extraction',
            'Multilingual translation via Gemini',
            'Text-to-speech in 60+ languages',
            'Enhanced audio generation',
            'Intelligent question answering',
            'Responsive web interface'
        ]
    })

@app.route('/')
def index():
    """Serve the main application page."""
    return send_file('index.html')

if __name__ == '__main__':
    print("🚀 Starting Smart Gurukul Enhanced Multilingual Flask Server...")
    print("📚 Advanced Multilingual Document Translation & Text-to-Speech Service")
    print(f"🌍 Supporting {len(LANGUAGE_NAMES)} languages for translation")
    print(f"🔊 TTS available for {len(GTTS_LANGUAGE_MAP)} languages")
    print("🤖 Enhanced with Gemini AI for better translations and audio")
    print("\n📋 Available endpoints:")
    print("- GET  /: Main application interface")
    print("- POST /upload-pdf: Upload and translate PDF documents")
    print("- POST /ask: Ask questions about documents in any language")
    print("- POST /tts: Convert text to speech")
    print("- POST /generate-speech: Generate speech for any text")
    print("- GET  /supported-languages: Get list of supported languages")
    print("- GET  /health: Health check and service info")
    print("- GET  /favicon.ico: Serve favicon")
    print(f"\n⚠️  API Key Status: {'✅ Set' if GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY' else '❌ Please set your GEMINI_API_KEY!'}")
    print("\n🔧 To set your API key:")
    print("   export GEMINI_API_KEY='your-actual-api-key'")
    print("   or replace 'YOUR_GEMINI_API_KEY' in the code")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
