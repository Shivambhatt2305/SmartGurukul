#4
from flask import Flask, request, jsonify, send_file, send_from_directory
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
import tempfile
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Use environment variable for Gemini API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyChFYnEka9jiBTHdTMK2jLH75X7K55ot4I')
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

def extract_pdf_pages(pdf_path):
    """Extract text from PDF file page by page."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if pdf_reader.is_encrypted:
                logger.error(f"PDF {pdf_path} is encrypted")
                return None
                
            pages = []
            total_pages = len(pdf_reader.pages)
                
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    pages.append({
                        'page_number': page_num + 1,
                        'content': page_text.strip(),
                        'total_pages': total_pages
                    })
                else:
                    pages.append({
                        'page_number': page_num + 1,
                        'content': f"[Page {page_num + 1} - Content could not be extracted]",
                        'total_pages': total_pages
                    })
                
            if not pages:
                logger.error(f"No pages extracted from {pdf_path}")
                return None
                
            logger.info(f"Extracted {len(pages)} pages from {pdf_path}")
            return pages
                
    except Exception as e:
        logger.error(f"PDF extraction error for {pdf_path}: {str(e)}")
        return None

def extract_pdf_text(pdf_file_or_path):
    """Extract all text from a PDF file (handles both file objects and paths)."""
    pdf_path = None
    try:
        # Handle file object (from upload)
        if hasattr(pdf_file_or_path, 'save'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                pdf_file_or_path.save(temp_file.name)
                pdf_path = temp_file.name
        else:
            # Handle file path
            pdf_path = pdf_file_or_path
            
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
        
        # Clean up temporary file if created
        if hasattr(pdf_file_or_path, 'save') and pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
            
        if not text.strip():
            logger.error("No text could be extracted from the PDF")
            return None
            
        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        return text.strip()
        
    except Exception as e:
        logger.error(f"PDF text extraction error: {str(e)}")
        # Clean up temporary file in case of error
        if hasattr(pdf_file_or_path, 'save') and pdf_path and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except:
                pass
        return None

def format_page_content(page_text, page_number, total_pages):
    """Format a single page content for teaching."""
    if not page_text:
        return f"<p>Page {page_number} of {total_pages} - No content available.</p>"
        
    # Add page header
    formatted_content = [f"<div class='page-header'><h2>Page {page_number} of {total_pages}</h2></div>"]
        
    # Split into paragraphs
    paragraphs = page_text.split('\n\n')
        
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
                
        # Check if it looks like a heading
        if len(para) < 100 and (para.isupper() or re.match(r'^\d+\.?\s', para) or para.endswith(':')):
            if para.isupper() or len(para.split()) <= 5:
                formatted_content.append(f"<h3>{para}</h3>")
            else:
                formatted_content.append(f"<h4>{para}</h4>")
        # Check for bullet points or numbered lists
        elif para.startswith(('•', '-', '*')) or re.match(r'^\d+\.', para):
            items = para.split('\n')
            list_items = []
            for item in items:
                item = item.strip()
                if item:
                    clean_item = re.sub(r'^[•\-\*]\s*', '', item)
                    clean_item = re.sub(r'^\d+\.\s*', '', clean_item)
                    list_items.append(f"<li>{clean_item}</li>")
            if list_items:
                formatted_content.append(f"<ul>{''.join(list_items)}</ul>")
        else:
            # Regular paragraph
            sentences = re.split(r'(?<=[.!?])\s+', para)
            if len(sentences) > 3:
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
        
    return ''.join(formatted_content) or f"<p>Page {page_number} of {total_pages} - No content available.</p>"

def generate_page_explanation(page_content, page_number, total_pages, language_code):
    """Generate an explanation for a specific page."""
    try:
        language_name = get_language_name(language_code)
        
        prompt = f"""You are an AI teacher explaining page {page_number} of {total_pages} from an educational document. Provide a clear, engaging explanation of this page content in {language_name}.

Page Content:
---
{page_content}
---

INSTRUCTIONS:
1. Respond ENTIRELY in {language_name}
2. Start by mentioning this is page {page_number} of {total_pages}
3. Provide a clear, structured explanation of the content
4. Break down complex concepts into simple terms
5. Use examples where helpful
6. Keep the explanation focused on this page only
7. End with a brief summary of key points from this page
8. Make it conversational and engaging for students
9. If this is the last page, mention that the chapter is complete

Response:"""

        response = model.generate_content(prompt)
        if not response.text:
            raise ValueError("Empty response from Gemini API")
            
        explanation = response.text.strip()
        
        # Clean up formatting
        explanation = re.sub(r'\*\*(.*?)\*\*', r'\1', explanation)
        explanation = re.sub(r'\*(.*?)\*', r'\1', explanation)
        explanation = re.sub(r'#{1,6}\s*', '', explanation)
        
        logger.info(f"Generated explanation for page {page_number} in {language_name}")
        return explanation
        
    except Exception as e:
        logger.error(f"Page explanation generation error: {str(e)}")
        return f"This is page {page_number} of {total_pages}. Let me explain the content on this page..."

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

def generate_voice_response(question, context, target_language):
    """Generate a conversational response optimized for voice interaction."""
    try:
        language_name = get_language_name(target_language)
        prompt = f"""You are a friendly AI teacher having a voice conversation with a student. The student just interrupted your explanation by saying "Hey Teacher" and asked a question. Respond ENTIRELY in {language_name}.

Document Context:
---
{context[:8000]}
---

Student's Question: {question}

INSTRUCTIONS for VOICE RESPONSE:
1. Respond COMPLETELY in {language_name}
2. Start with a friendly acknowledgment like "Yes, I'm here to help!" or "Great question!"
3. Keep the response conversational and concise (2-3 short paragraphs maximum)
4. If the answer is in the context, explain it clearly and simply
5. If not in the context, acknowledge this and provide helpful general guidance
6. End with encouragement like "Does that help?" or "Any other questions?"
7. Use natural speech patterns - this will be spoken aloud
8. Be warm, encouraging, and supportive like a real teacher
9. Avoid complex formatting or technical jargon

Response:"""

        response = model.generate_content(prompt)
        if not response.text:
            raise ValueError("Empty response from Gemini API")
            
        answer_text = response.text.strip()
        
        # Clean up formatting for voice
        answer_text = re.sub(r'\*\*(.*?)\*\*', r'\1', answer_text)
        answer_text = re.sub(r'\*(.*?)\*', r'\1', answer_text)
        answer_text = re.sub(r'#{1,6}\s*', '', answer_text)
        
        logger.info(f"Generated voice response in {language_name}")
        return answer_text
        
    except Exception as e:
        logger.error(f"Voice response generation error: {str(e)}")
        return "I'm here to help! Could you please repeat your question? I want to make sure I understand what you're asking about."

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

def format_teacher_response(response):
    """Format response into clean HTML."""
    if not response:
        return "<p>No content available.</p>"
        
    paragraphs = response.split('\n\n')
    formatted_response = []
    
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
        elif para.startswith('#### '):
            formatted_response.append(f"<h4>{para[5:].strip()}</h4>")
        elif para.startswith('- ') or para.startswith('* '):
            items = [f"<li>{line[2:].strip()}</li>" for line in para.split('\n') if line.startswith(('- ', '* '))]
            formatted_response.append(f"<ul>{''.join(items)}</ul>")
        elif re.match(r'^\d+\.', para):
            items = [f"<li>{re.sub(r'^\d+\.\s*', '', line).strip()}</li>" for line in para.split('\n') if re.match(r'^\d+\.', line)]
            formatted_response.append(f"<ol>{''.join(items)}</ol>")
        else:
            para = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', para)
            para = re.sub(r'\*(.*?)\*', r'<em>\1</em>', para)
            para = re.sub(r'`(.*?)`', r'<code>\1</code>', para)
            formatted_response.append(f"<p>{para}</p>")
    
    return ''.join(formatted_response) or "<p>No content available.</p>"

# --- API ENDPOINTS ---
@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 404 errors."""
    try:
        return send_file(os.path.join(os.path.dirname(__file__), 'static/favicon.ico'))
    except FileNotFoundError:
        # Return a minimal 1x1 transparent PNG as fallback
        return send_file(
            BytesIO(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=')),
            mimetype='image/png'
        )

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
    """Get the original PDF content for display with page information."""
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

        # Extract pages
        pages = extract_pdf_pages(chapter_path)
        if not pages:
            return jsonify({'success': False, 'error': 'Could not extract pages from the chapter PDF'}), 500

        # Extract full text for backward compatibility
        chapter_text = extract_pdf_text(chapter_path)
        
        # Provide PDF URL for embedding
        pdf_url = f"/get-pdf/{subject}/{chapter}"
        
        return jsonify({
            'success': True,
            'subject': subject,
            'chapter': chapter,
            'pdf_url': pdf_url,
            'raw_text': chapter_text,
            'pages': pages,
            'total_pages': len(pages),
            'has_pdf': True
        })
        
    except Exception as e:
        logger.error(f"Error in /get-chapter-content: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/teach-page', methods=['POST'])
def teach_page():
    """Get explanation for a specific page with audio generation."""
    try:
        data = request.get_json()
        subject = data.get('subject', '').strip()
        chapter = data.get('chapter', '').strip()
        page_number = data.get('page_number', 1)
        target_language = data.get('language', 'en').lower().strip()
        
        if not subject or not chapter:
            return jsonify({'success': False, 'error': 'Subject and chapter are required'}), 400
            
        if not is_language_supported(target_language):
            return jsonify({'success': False, 'error': f'Language "{target_language}" is not supported'}), 400

        chapter_path = os.path.join(DATABASE_DIR, subject, f"{chapter}.pdf")
        if not os.path.exists(chapter_path):
            return jsonify({'success': False, 'error': f'Chapter "{chapter}" not found for subject "{subject}"'}), 404

        # Extract pages
        pages = extract_pdf_pages(chapter_path)
        if not pages:
            return jsonify({'success': False, 'error': 'Could not extract pages from the chapter PDF'}), 500
            
        # Find the requested page
        target_page = None
        for page in pages:
            if page['page_number'] == page_number:
                target_page = page
                break
                
        if not target_page:
            return jsonify({'success': False, 'error': f'Page {page_number} not found'}), 404
            
        # Generate explanation for this page
        explanation = generate_page_explanation(
            target_page['content'], 
            page_number, 
            target_page['total_pages'], 
            target_language
        )
        
        # Format content
        formatted_content = format_page_content(
            target_page['content'], 
            page_number, 
            target_page['total_pages']
        )
        
        # Generate audio
        audio_data = generate_audio_for_text(explanation, target_language)
        
        # Check if this is the last page
        is_last_page = page_number >= target_page['total_pages']
        next_page = page_number + 1 if not is_last_page else None
        
        return jsonify({
            'success': True,
            'subject': subject,
            'chapter': chapter,
            'page_number': page_number,
            'total_pages': target_page['total_pages'],
            'content': formatted_content,
            'explanation': format_teacher_response(explanation),
            'audio_data': audio_data,
            'language': target_language,
            'language_name': get_language_name(target_language),
            'has_audio': audio_data is not None,
            'is_last_page': is_last_page,
            'next_page': next_page
        })
        
    except Exception as e:
        logger.error(f"Error in /teach-page: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/teach-chapter', methods=['POST'])
def teach_chapter():
    """Get chapter content with audio generation (starts from page 1)."""
    try:
        data = request.get_json()
        subject = data.get('subject', '').strip()
        chapter = data.get('chapter', '').strip()
        target_language = data.get('language', 'en').lower().strip()
        
        # Redirect to teach-page for page 1
        data['page_number'] = 1
        return teach_page()
        
    except Exception as e:
        logger.error(f"Error in /teach-chapter: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/voice-ask', methods=['POST'])
def voice_ask():
    """Handle voice questions from students - optimized for interruption flow."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        subject = data.get('subject', '').strip()
        chapter = data.get('chapter', '').strip()
        target_language = data.get('language', 'en').lower().strip()
        
        logger.info(f"Voice question received: {question}")
        
        if not question:
            error_msg = "I'm listening! Please ask your question."
            audio_data = generate_audio_for_text(error_msg, target_language)
            return jsonify({
                'success': True, 
                'answer': f'<p>{error_msg}</p>',
                'audio_data': audio_data,
                'has_audio': audio_data is not None,
                'is_voice_response': True
            })
        
        if not is_language_supported(target_language):
            error_msg = f'Sorry, I don\'t support {target_language} language yet.'
            audio_data = generate_audio_for_text(error_msg, target_language)
            return jsonify({
                'success': False, 
                'answer': f'<p>{error_msg}</p>',
                'audio_data': audio_data,
                'has_audio': audio_data is not None
            })
        
        if not subject or not chapter:
            error_msg = "Please select a chapter first, then I can answer your questions about it."
            audio_data = generate_audio_for_text(error_msg, target_language)
            return jsonify({
                'success': True, 
                'answer': f'<p>{error_msg}</p>',
                'audio_data': audio_data,
                'has_audio': audio_data is not None,
                'is_voice_response': True
            })
        
        chapter_path = os.path.join(DATABASE_DIR, subject, f"{chapter}.pdf")
        if not os.path.exists(chapter_path):
            error_msg = f'I don\'t have access to that chapter. Please check if the file exists.'
            audio_data = generate_audio_for_text(error_msg, target_language)
            return jsonify({
                'success': False, 
                'answer': f'<p>{error_msg}</p>',
                'audio_data': audio_data,
                'has_audio': audio_data is not None
            })
        
        chapter_text = extract_pdf_text(chapter_path)
        if not chapter_text:
            error_msg = 'I\'m having trouble reading that chapter. Please try again.'
            audio_data = generate_audio_for_text(error_msg, target_language)
            return jsonify({
                'success': False, 
                'answer': f'<p>{error_msg}</p>',
                'audio_data': audio_data,
                'has_audio': audio_data is not None
            })
        
        # Generate voice-optimized response
        answer_text = generate_voice_response(question, chapter_text, target_language)
        formatted_answer = format_teacher_response(answer_text)
        audio_data = generate_audio_for_text(answer_text, target_language)
        
        return jsonify({
            'success': True,
            'answer': formatted_answer,
            'audio_data': audio_data,
            'language': target_language,
            'language_name': get_language_name(target_language),
            'has_audio': audio_data is not None,
            'is_voice_response': True
        })
        
    except Exception as e:
        logger.error(f"Error in /voice-ask: {str(e)}")
        error_msg = "I'm having trouble processing your question. Could you please try again?"
        try:
            audio_data = generate_audio_for_text(error_msg, target_language)
        except:
            audio_data = None
        return jsonify({
            'success': False, 
            'answer': f'<p>{error_msg}</p>',
            'audio_data': audio_data,
            'has_audio': audio_data is not None,
            'is_voice_response': True
        })

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
    """Handle regular text questions about a chapter with multilingual support."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        context = data.get('context', '').strip()
        subject = data.get('subject', '').strip()
        chapter = data.get('chapter', '').strip()
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
            
        if not is_language_supported(target_language):
            return jsonify({'success': False, 'answer': f'<p>Language "{target_language}" is not supported.</p>'}), 400
            
        # If subject and chapter are provided, get context from database
        if subject and chapter and not context:
            chapter_path = os.path.join(DATABASE_DIR, subject, f"{chapter}.pdf")
            if os.path.exists(chapter_path):
                context = extract_pdf_text(chapter_path)
                if not context:
                    return jsonify({'success': False, 'error': 'Could not extract text from the chapter PDF'}), 500
        
        # Generate response in target language
        if context:
            answer_text = generate_multilingual_response(question, context, target_language)
        else:
            answer_text = generate_voice_response(question, "", target_language)
            
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
    """List all supported languages."""
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Smart Gurukul Teaching Assistant with Multilingual Support',
        'database_dir': DATABASE_DIR,
        'database_exists': os.path.exists(DATABASE_DIR),
        'supported_languages': len(LANGUAGE_NAMES),
        'tts_languages': len(GTTS_LANGUAGE_MAP),
        'version': '3.0.0',
        'voice_enabled': True,
        'wake_word': 'Hey Teacher',
        'features': [
            'auto_page_progression', 
            'page_by_page_teaching',
            'multilingual_translation',
            'enhanced_audio_generation',
            'pdf_upload_support',
            'voice_interaction'
        ]
    })

@app.route('/')
def index():
    """Serve the main UI."""
    return send_file('teacher.html')

@app.route('/get-pdf/<subject>/<chapter>')
def get_pdf(subject, chapter):
    """Serve PDF file directly with proper headers."""
    try:
        pdf_path = os.path.join(DATABASE_DIR, subject, f"{chapter}.pdf")
        if not os.path.exists(pdf_path):
            return jsonify({'error': 'PDF not found'}), 404
        
        return send_file(
            pdf_path, 
            mimetype='application/pdf',
            as_attachment=False,
            download_name=f"{chapter}.pdf"
        )
    except Exception as e:
        logger.error(f"Error serving PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Smart Gurukul Enhanced Teaching Assistant with Multilingual Support...")
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
    
    print(f"🌍 Supporting {len(LANGUAGE_NAMES)} languages for translation")
    print(f"🔊 TTS available for {len(GTTS_LANGUAGE_MAP)} languages")
    print("🎤 Enhanced voice interaction with 'Hey Teacher' wake word")
    print("⚡ Instant audio interruption and seamless Q&A flow")
    print("📖 Automatic page progression when explanation ends")
    print("📄 Page-by-page teaching with visual indicators")
    print("🌐 Multilingual PDF upload and translation support")
    print("🤖 Enhanced with Gemini AI for better translations and audio")
    
    print("\n📋 Available endpoints:")
    print("- GET  /subjects: List available subjects")
    print("- POST /chapters: List chapters for a subject")
    print("- POST /get-chapter-content: Get original PDF content with pages")
    print("- POST /teach-page: Get explanation for a specific page")
    print("- POST /teach-chapter: Get chapter content with audio (starts page 1)")
    print("- POST /ask: Answer text questions about a chapter")
    print("- POST /voice-ask: Answer voice questions with optimized responses")
    print("- POST /upload-pdf: Upload and translate PDF documents")
    print("- POST /tts: Convert text to speech")
    print("- POST /generate-speech: Generate speech for any text")
    print("- GET  /supported-languages: List supported languages")
    print("- GET  /health: Health check")
    print("- GET  /: Main UI")
    print("- GET  /favicon.ico: Favicon (prevents 404 errors)")
    print("- GET  /get-pdf/<subject>/<chapter>: Serve PDF files")
    
    print(f"\n⚠️ API Key: {'✅ Set' if GEMINI_API_KEY not in ['YOUR_GEMINI_API_KEY', 'AIzaSyCr46nkrI0cmCpYybRg8uMtsnAHzQpb1VM'] else '❌ Please set GEMINI_API_KEY'}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
