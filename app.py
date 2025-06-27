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
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Use environment variable for Gemini API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCr46nkrI0cmCpYybRg8uMtsnAHzQpb1VM')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Database directory for subjects and chapters
DATABASE_DIR = "database"
os.makedirs(DATABASE_DIR, exist_ok=True)

# Comprehensive language mapping
LANGUAGE_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'hi': 'Hindi',
    'ja': 'Japanese', 'ko': 'Korean', 'zh-cn': 'Chinese (Simplified)', 'zh-tw': 'Chinese (Traditional)',
    'ru': 'Russian', 'pt': 'Portuguese', 'it': 'Italian', 'ar': 'Arabic', 'bn': 'Bengali',
    'ur': 'Urdu', 'tr': 'Turkish', 'nl': 'Dutch', 'sv': 'Swedish', 'da': 'Danish',
    'no': 'Norwegian', 'fi': 'Finnish', 'pl': 'Polish', 'cs': 'Czech', 'sk': 'Slovak',
    'hu': 'Hungarian', 'ro': 'Romanian', 'bg': 'Bulgarian', 'hr': 'Croatian', 'sr': 'Serbian',
    'sl': 'Slovenian', 'et': 'Estonian', 'lv': 'Latvian', 'lt': 'Lithuanian', 'uk': 'Ukrainian',
    'be': 'Belarusian', 'ka': 'Georgian', 'hy': 'Armenian', 'az': 'Azerbaijani', 'kk': 'Kazakh',
    'ky': 'Kyrgyz', 'uz': 'Uzbek', 'tg': 'Tajik', 'mn': 'Mongolian', 'th': 'Thai',
    'vi': 'Vietnamese', 'id': 'Indonesian', 'ms': 'Malay', 'tl': 'Filipino', 'sw': 'Swahili',
    'am': 'Amharic', 'he': 'Hebrew', 'fa': 'Persian', 'ps': 'Pashto', 'sd': 'Sindhi',
    'ne': 'Nepali', 'si': 'Sinhala', 'my': 'Myanmar', 'km': 'Khmer', 'lo': 'Lao',
    'mt': 'Maltese', 'is': 'Icelandic', 'ga': 'Irish', 'cy': 'Welsh', 'eu': 'Basque',
    'ca': 'Catalan', 'gl': 'Galician', 'af': 'Afrikaans', 'sq': 'Albanian', 'mk': 'Macedonian'
}

GTTS_LANGUAGE_MAP = {
    'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'hi': 'hi', 'ja': 'ja', 'ko': 'ko',
    'zh-cn': 'zh', 'zh-tw': 'zh-tw', 'ru': 'ru', 'pt': 'pt', 'it': 'it', 'ar': 'ar',
    'bn': 'bn', 'ur': 'ur', 'tr': 'tr', 'nl': 'nl', 'sv': 'sv', 'da': 'da', 'no': 'no',
    'fi': 'fi', 'pl': 'pl', 'cs': 'cs', 'sk': 'sk', 'hu': 'hu', 'ro': 'ro', 'bg': 'bg',
    'hr': 'hr', 'sr': 'sr', 'sl': 'sl', 'et': 'et', 'lv': 'lv', 'lt': 'lt', 'uk': 'uk',
    'ka': 'ka', 'hy': 'hy', 'az': 'az', 'kk': 'kk', 'ky': 'ky', 'uz': 'uz', 'mn': 'mn',
    'th': 'th', 'vi': 'vi', 'id': 'id', 'ms': 'ms', 'tl': 'tl', 'sw': 'sw', 'am': 'am',
    'he': 'iw', 'fa': 'fa', 'ne': 'ne', 'si': 'si', 'my': 'my', 'km': 'km', 'lo': 'lo',
    'mt': 'mt', 'is': 'is', 'ga': 'ga', 'cy': 'cy', 'eu': 'eu', 'ca': 'ca', 'gl': 'gl',
    'af': 'af', 'sq': 'sq', 'mk': 'mk'
}

# --- HELPER FUNCTIONS ---

def is_language_supported(language_code):
    """Check if a language is supported for translation and TTS."""
    return language_code in LANGUAGE_NAMES

def get_language_name(language_code):
    """Get the full name of a language from its code."""
    return LANGUAGE_NAMES.get(language_code, 'Unknown Language')

def extract_pdf_text(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if pdf_reader.is_encrypted:
                logger.error(f"PDF {pdf_path} is encrypted")
                return None
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            if not text.strip():
                logger.error(f"No text extracted from {pdf_path}")
                return None
            logger.info(f"Extracted {len(text)} characters from {pdf_path}")
            return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error for {pdf_path}: {str(e)}")
        return None

def format_pdf_content(text):
    """Format PDF text content into clean HTML."""
    if not text:
        return "<p>No content available.</p>"
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    formatted_content = []
    
    # Define regex patterns outside f-strings
    number_pattern = r'^\d+\.?\s'
    digit_pattern = r'^\d+\.'
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # Check if it looks like a heading (short line, all caps, or starts with numbers)
        if len(para) < 100 and (para.isupper() or re.match(number_pattern, para) or para.endswith(':')):
            if para.isupper() or len(para.split()) <= 5:
                formatted_content.append(f"<h2>{para}</h2>")
            else:
                formatted_content.append(f"<h3>{para}</h3>")
        # Check for bullet points or numbered lists
        elif para.startswith(('•', '-', '*')) or re.match(digit_pattern, para):
            items = para.split('\n')
            list_items = []
            for item in items:
                item = item.strip()
                if item:
                    # Remove bullet points and numbers
                    clean_item = re.sub(r'^[•\-\*]\s*', '', item)
                    clean_item = re.sub(r'^\d+\.\s*', '', clean_item)
                    list_items.append(f"<li>{clean_item}</li>")
            if list_items:
                formatted_content.append(f"<ul>{''.join(list_items)}</ul>")
        else:
            # Regular paragraph
            # Split long paragraphs at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', para)
            if len(sentences) > 3:
                # Group sentences into smaller paragraphs
                current_para = []
                for sentence in sentences:
                    current_para.append(sentence)
                    if len(current_para) >= 3:
                        formatted_content.append(f"<p>{' '.join(current_para)}</p>")
                        current_para = []
                if current_para:
                    formatted_content.append(f"<p>{' '.join(current_para)}</p>")
            else:
                formatted_content.append(f"<p>{para}</p>")
    
    return ''.join(formatted_content) or "<p>No content available.</p>"

def generate_audio_for_text(text, language_code):
    """Generate audio for text in specified language."""
    try:
        if not text.strip():
            logger.error("No text provided for audio generation")
            return None
        
        # Clean HTML tags and format for TTS
        clean_text = re.sub('<[^<]+?>', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Limit text length for TTS
        max_length = 4500
        if len(clean_text) > max_length:
            clean_text = clean_text[:max_length] + "..."
        
        gtts_lang = GTTS_LANGUAGE_MAP.get(language_code, 'en')
        slow_speech = language_code in ['ar', 'hi', 'bn', 'ur', 'th', 'my', 'km']
        
        tts = gTTS(text=clean_text, lang=gtts_lang, slow=slow_speech)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        
        audio_data = base64.b64encode(audio_file.getvalue()).decode('utf-8')
        logger.info(f"Generated audio for {gtts_lang} ({len(clean_text)} chars)")
        return audio_data
    except Exception as e:
        logger.error(f"Audio generation error for {language_code}: {str(e)}")
        return None

def generate_multilingual_response(question, context, target_language):
    """Generate a response to a question in the target language."""
    try:
        language_name = get_language_name(target_language)
        prompt = f"""You are an expert multilingual teacher. Respond to the following question ENTIRELY in {language_name} based on the provided context.

Document Context:
---
{context[:10000]}  # Limit to avoid API token limits
---

Student's Question: {question}

CRITICAL INSTRUCTIONS:
1. Respond COMPLETELY in {language_name}
2. If the answer is found in the context, explain it thoroughly
3. If the answer is NOT in the context, state this clearly in {language_name} and provide a general explanation
4. Be encouraging, clear, patient, and educational
5. Use appropriate formatting and structure
6. Keep the response concise but comprehensive (2-4 paragraphs)
7. Use natural, fluent {language_name}
8. End with an encouraging note

Response:"""
        response = model.generate_content(prompt)
        if not response.text:
            raise ValueError("Empty response from Gemini API")
        answer_text = response.text.strip()
        prefixes_to_remove = [
            f"Response in {language_name}:",
            f"Answer in {language_name}:",
            "Response:"
        ]
        for prefix in prefixes_to_remove:
            if answer_text.startswith(prefix):
                answer_text = answer_text[len(prefix):].strip()
        logger.info(f"Generated response in {language_name}")
        return answer_text
    except Exception as e:
        logger.error(f"Response generation error: {str(e)}")
        return f"I apologize, but I couldn't generate a response in {language_name}. Please try again."

def format_teacher_response(response):
    """Format response into clean HTML."""
    if not response:
        return "<p>No content available.</p>"
    
    paragraphs = response.split('\n\n')
    formatted_response = []
    
    # Define regex patterns outside f-strings
    bullet_pattern = r'^[•\-\*]\s*'
    number_remove_pattern = r'^\d+\.\s*'
    number_match_pattern = r'^\d+\.'
    bold_pattern = r'\*\*(.*?)\*\*'
    italic_pattern = r'\*(.*?)\*'
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if para.startswith('# '):
            formatted_response.append(f"<h1>{para[2:].strip()}</h1>")
        elif para.startswith('## '):
            formatted_response.append(f"<h2>{para[3:].strip()}</h2>")
        elif para.startswith('### '):
            formatted_response.append(f"<h3>{para[4:].strip()}</h3>")
        elif para.startswith('- ') or para.startswith('* '):
            items = [f"<li>{line[2:].strip()}</li>" for line in para.split('\n') if line.startswith(('- ', '* '))]
            formatted_response.append(f"<ul>{''.join(items)}</ul>")
        elif re.match(number_match_pattern, para):
            items = []
            for line in para.split('\n'):
                if re.match(number_match_pattern, line):
                    clean_line = re.sub(number_remove_pattern, '', line.strip())
                    items.append(f"<li>{clean_line}</li>")
            formatted_response.append(f"<ol>{''.join(items)}</ol>")
        else:
            para = re.sub(bold_pattern, r'<strong>\1</strong>', para)
            para = re.sub(italic_pattern, r'<em>\1</em>', para)
            formatted_response.append(f"<p>{para}</p>")
    return ''.join(formatted_response) or "<p>No content available.</p>"

# --- API ENDPOINTS ---

@app.route('/subjects', methods=['GET'])
def get_subjects():
    """List all available subjects in the database."""
    try:
        logger.info(f"Checking database directory: {DATABASE_DIR}")
        logger.info(f"Directory exists: {os.path.exists(DATABASE_DIR)}")
        
        if not os.path.exists(DATABASE_DIR):
            logger.error(f"Database directory {DATABASE_DIR} does not exist")
            return jsonify({'success': False, 'error': f'Database directory {DATABASE_DIR} not found'}), 404
        
        subjects = []
        for item in os.listdir(DATABASE_DIR):
            item_path = os.path.join(DATABASE_DIR, item)
            if os.path.isdir(item_path):
                subjects.append(item)
                logger.info(f"Found subject: {item}")
        
        if not subjects:
            logger.warning("No subjects found in database")
            return jsonify({'success': False, 'error': 'No subjects found in the database'}), 404
        
        logger.info(f"Returning {len(subjects)} subjects: {subjects}")
        return jsonify({'success': True, 'subjects': subjects})
    except Exception as e:
        logger.error(f"Error listing subjects: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/chapters', methods=['POST'])
def get_chapters():
    """List chapters for a given subject."""
    try:
        data = request.get_json()
        subject = data.get('subject', '').strip()
        
        logger.info(f"Looking for chapters in subject: {subject}")
        
        subject_path = os.path.join(DATABASE_DIR, subject)
        logger.info(f"Subject path: {subject_path}")
        logger.info(f"Subject path exists: {os.path.exists(subject_path)}")
        
        if not os.path.isdir(subject_path):
            logger.error(f"Subject directory not found: {subject_path}")
            return jsonify({'success': False, 'error': f'Subject "{subject}" not found'}), 404
        
        chapters = []
        for file in os.listdir(subject_path):
            if file.endswith('.pdf'):
                chapter_name = file[:-4]  # Remove .pdf extension
                chapters.append(chapter_name)
                logger.info(f"Found chapter: {chapter_name}")
        
        if not chapters:
            logger.warning(f"No PDF chapters found in {subject_path}")
            return jsonify({'success': False, 'error': f'No chapters found for {subject}'}), 404
        
        logger.info(f"Returning {len(chapters)} chapters for {subject}: {chapters}")
        return jsonify({'success': True, 'subject': subject, 'chapters': chapters})
    except Exception as e:
        logger.error(f"Error listing chapters: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get-chapter-content', methods=['POST'])
def get_chapter_content():
    """Get the original PDF content for display."""
    try:
        data = request.get_json()
        subject = data.get('subject', '').strip()
        chapter = data.get('chapter', '').strip()

        if not subject or not chapter:
            return jsonify({'success': False, 'error': 'Subject and chapter are required'}), 400

        chapter_path = os.path.join(DATABASE_DIR, subject, f"{chapter}.pdf")
        logger.info(f"Looking for PDF at: {chapter_path}")
        
        if not os.path.exists(chapter_path):
            logger.error(f"Chapter file not found: {chapter_path}")
            return jsonify({'success': False, 'error': f'Chapter "{chapter}" not found for subject "{subject}"'}), 404

        # Extract text for chat functionality
        chapter_text = extract_pdf_text(chapter_path)
        if not chapter_text:
            return jsonify({'success': False, 'error': 'Could not extract text from the chapter PDF'}), 500

        # Provide PDF URL for embedding
        pdf_url = f"/get-pdf/{subject}/{chapter}"

        return jsonify({
            'success': True,
            'subject': subject,
            'chapter': chapter,
            'pdf_url': pdf_url,
            'raw_text': chapter_text,
            'has_pdf': True
        })
    except Exception as e:
        logger.error(f"Error in /get-chapter-content: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/teach-chapter', methods=['POST'])
def teach_chapter():
    """Get chapter content with audio generation."""
    try:
        data = request.get_json()
        subject = data.get('subject', '').strip()
        chapter = data.get('chapter', '').strip()
        target_language = data.get('language', 'en').lower().strip()

        if not subject or not chapter:
            return jsonify({'success': False, 'error': 'Subject and chapter are required'}), 400
        if not is_language_supported(target_language):
            return jsonify({'success': False, 'error': f'Language "{target_language}" is not supported'}), 400

        chapter_path = os.path.join(DATABASE_DIR, subject, f"{chapter}.pdf")
        if not os.path.exists(chapter_path):
            return jsonify({'success': False, 'error': f'Chapter "{chapter}" not found for subject "{subject}"'}), 404

        chapter_text = extract_pdf_text(chapter_path)
        if not chapter_text:
            return jsonify({'success': False, 'error': 'Could not extract text from the chapter PDF'}), 500

        formatted_content = format_pdf_content(chapter_text)
        audio_data = generate_audio_for_text(chapter_text, target_language)

        return jsonify({
            'success': True,
            'subject': subject,
            'chapter': chapter,
            'content': formatted_content,
            'audio_data': audio_data,
            'language': target_language,
            'language_name': get_language_name(target_language),
            'has_audio': audio_data is not None
        })
    except Exception as e:
        logger.error(f"Error in /teach-chapter: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    """Handle student questions about a chapter."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        subject = data.get('subject', '').strip()
        chapter = data.get('chapter', '').strip()
        target_language = data.get('language', 'en').lower().strip()

        if not question:
            error_msg = "Please ask a valid question."
            return jsonify({'success': False, 'answer': f'<p>{error_msg}</p>'}), 400
        if not is_language_supported(target_language):
            return jsonify({'success': False, 'answer': f'<p>Language "{target_language}" is not supported.</p>'}), 400
        if not subject or not chapter:
            return jsonify({'success': False, 'error': 'Subject and chapter are required'}), 400

        chapter_path = os.path.join(DATABASE_DIR, subject, f"{chapter}.pdf")
        if not os.path.exists(chapter_path):
            return jsonify({'success': False, 'error': f'Chapter "{chapter}" not found for subject "{subject}"'}), 404

        chapter_text = extract_pdf_text(chapter_path)
        if not chapter_text:
            return jsonify({'success': False, 'error': 'Could not extract text from the chapter PDF'}), 500

        answer_text = generate_multilingual_response(question, chapter_text, target_language)
        formatted_answer = format_teacher_response(answer_text)
        audio_data = generate_audio_for_text(answer_text, target_language)

        return jsonify({
            'success': True,
            'answer': formatted_answer,
            'audio_data': audio_data,
            'language': target_language,
            'language_name': get_language_name(target_language),
            'has_audio': audio_data is not None
        })
    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}")
        return jsonify({'success': False, 'answer': f'<p>Error: {str(e)}</p>'}), 500

@app.route('/supported-languages', methods=['GET'])
def get_supported_languages():
    """List all supported languages."""
    try:
        languages = [
            {'code': code, 'name': name, 'tts_supported': code in GTTS_LANGUAGE_MAP}
            for code, name in LANGUAGE_NAMES.items()
        ]
        return jsonify({
            'success': True,
            'languages': languages,
            'total_count': len(languages),
            'tts_count': len(GTTS_LANGUAGE_MAP)
        })
    except Exception as e:
        logger.error(f"Error in /supported-languages: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Smart Gurukul Teaching Assistant',
        'database_dir': DATABASE_DIR,
        'database_exists': os.path.exists(DATABASE_DIR),
        'supported_languages': len(LANGUAGE_NAMES),
        'tts_languages': len(GTTS_LANGUAGE_MAP),
        'version': '2.0.0'
    })

@app.route('/')
def index():
    """Serve the main UI or return JSON response."""
    try:
        # Try to serve the HTML file if it exists
        if os.path.exists('teacher.html'):
            return send_file('teacher.html')
        else:
            # Return a JSON response with API information
            return jsonify({
                'service': 'Smart Gurukul Teaching Assistant',
                'version': '2.0.0',
                'status': 'online',
                'message': 'API is running successfully',
                'endpoints': {
                    'GET /subjects': 'List all available subjects',
                    'POST /chapters': 'Get chapters for a subject',
                    'POST /get-chapter-content': 'Get PDF content for a chapter',
                    'POST /teach-chapter': 'Get chapter content with audio',
                    'POST /ask': 'Ask questions about a chapter',
                    'GET /supported-languages': 'List supported languages',
                    'GET /health': 'Health check'
                },
                'features': [
                    'Support for 75+ languages',
                    'Text-to-speech in 50+ languages',
                    'PDF content extraction',
                    'AI-powered question answering',
                    'Subject and chapter organization'
                ]
            })
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/get-pdf/<subject>/<chapter>')
def get_pdf(subject, chapter):
    """Serve PDF file directly."""
    try:
        pdf_path = os.path.join(DATABASE_DIR, subject, f"{chapter}.pdf")
        if not os.path.exists(pdf_path):
            return jsonify({'error': 'PDF not found'}), 404
        return send_file(pdf_path, mimetype='application/pdf')
    except Exception as e:
        logger.error(f"Error serving PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Smart Gurukul Teaching Assistant...")
    print(f"📚 Database directory: {os.path.abspath(DATABASE_DIR)}")
    print(f"📁 Database exists: {os.path.exists(DATABASE_DIR)}")
    
    if os.path.exists(DATABASE_DIR):
        subjects = [d for d in os.listdir(DATABASE_DIR) if os.path.isdir(os.path.join(DATABASE_DIR, d))]
        print(f"📖 Found {len(subjects)} subjects: {subjects}")
        
        for subject in subjects:
            subject_path = os.path.join(DATABASE_DIR, subject)
            pdfs = [f for f in os.listdir(subject_path) if f.endswith('.pdf')]
            print(f"   📄 {subject}: {len(pdfs)} chapters")
    else:
        print("⚠️  Database directory not found! Please create the 'database' folder.")
    
    print(f"🌍 Supporting {len(LANGUAGE_NAMES)} languages")
    print(f"🔊 TTS available for {len(GTTS_LANGUAGE_MAP)} languages")
    print("\n📋 Available endpoints:")
    print("- GET  /subjects: List available subjects")
    print("- POST /chapters: List chapters for a subject")
    print("- POST /get-chapter-content: Get original PDF content")
    print("- POST /teach-chapter: Get chapter content with audio")
    print("- POST /ask: Answer questions about a chapter")
    print("- GET  /supported-languages: List supported languages")
    print("- GET  /health: Health check")
    print("- GET  /: Main UI")
    print(f"\n⚠️ API Key: {'✅ Set' if GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY' else '❌ Please set GEMINI_API_KEY'}")
    app.run(host='0.0.0.0', port=5000, debug=True)
