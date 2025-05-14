import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Suppress symlink warning for Windows

import torch
import librosa
import subprocess
import gradio as gr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from threading import Lock
import time
import mimetypes
import logging
from spellchecker import SpellChecker
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

spell = SpellChecker()


print("Initializing speech-to-text model...")
asr_model = "facebook/wav2vec2-large-960h-lv60-self"
processor = Wav2Vec2Processor.from_pretrained(asr_model)
speech_model = Wav2Vec2ForCTC.from_pretrained(asr_model)
speech_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
speech_model = speech_model.to(device)
print(f"Speech-to-text model loaded: {asr_model}, Using device: {device}")

# Initialize summarization model
print("Initializing summarization model...")
model_name = "facebook/bart-large-cnn"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer_model.to(device)
    summarizer_model.eval()  # Ensure evaluation mode
    print(f"Summarization model loaded: {model_name}")
except Exception as e:
    logger.error(f"Error loading summarization model: {e}")
    raise

# Thread lock for safe file operations
file_lock = Lock()

# Speech-to-Text Functions
def preprocess_audio(audio_data, sample_rate=16000):
    """Preprocess audio for better transcription accuracy."""
    audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))
    audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
    if len(audio_data) < sample_rate:
        audio_data = np.pad(audio_data, (0, sample_rate - len(audio_data)))
    return audio_data

def convert_to_wav(audio_path):
    """Convert audio to WAV with error handling."""
    output_wav = "temp_audio.wav"
    print(f"Converting {audio_path} to WAV format...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_wav, "-y"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")
        print("Audio conversion completed.")
        return output_wav
    except Exception as e:
        raise Exception(f"Audio conversion failed: {str(e)}")

def process_audio_segment(segment, sr=16000):
    """Process individual audio segment and return in lowercase."""
    input_values = processor(segment, return_tensors="pt", sampling_rate=sr).input_values
    input_values = input_values.to(device)
    with torch.no_grad():
        logits = speech_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.lower()  # Convert to lowercase

# Shared Summarization Function
def correct_typos_contextually(input_text):
    """Corrects typos based on dynamic context from the input text."""
    try:
        words = input_text.split()
        word_freq = Counter(words)
        corrected_words = []
        i = 0
        while i < len(words):
            word = words[i]
            if word in spell or word_freq[word] > 2:
                corrected_words.append(word)
                i += 1
                continue
            prev_word = words[i - 1] if i > 0 else ""
            next_word = words[i + 1] if i < len(words) - 1 else ""
            candidates = spell.candidates(word) or {word}
            best_candidate = word
            for candidate in candidates:
                if (prev_word and candidate.startswith(prev_word[-2:])) or \
                   (next_word and candidate.endswith(next_word[:2])) or \
                   candidate in word_freq:
                    best_candidate = candidate
                    break
                else:
                    best_candidate = spell.correction(word) or word
            corrected_words.append(best_candidate)
            i += 1
        corrected_text = " ".join(corrected_words)
        logger.info(f"Corrected text: {corrected_text}")
        return corrected_text
    except Exception as e:
        logger.error(f"Error in contextual typo correction: {e}")
        return input_text

def summarize_text(input_text, base_max_length=200, base_min_length=50):
    """Summarize text with dynamic length adjustment and chunking for long inputs."""
    try:
        if not input_text or input_text.strip() == "":
            return "no valid text provided"

        corrected_text = correct_typos_contextually(input_text)
        input_length = len(corrected_text.split())
        logger.info(f"Input length: {input_length} words")

        # Tokenize to check token count
        tokens = tokenizer.encode(corrected_text, truncation=False)
        logger.info(f"Input token count: {len(tokens)}")

        # Handle long inputs by chunking
        max_input_tokens = 1000  # Leave room for "summarize: " prefix
        summaries = []
        if len(tokens) > max_input_tokens:
            words = corrected_text.split()
            chunk_size = max_input_tokens // 2  # Approx words per chunk
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
            logger.info(f"Text split into {len(chunks)} chunks")
        else:
            chunks = [corrected_text]

        # Summarize each chunk
        for chunk in chunks:
            chunk_length = len(chunk.split())
            if chunk_length < 20:  # Too short to summarize meaningfully
                summaries.append(chunk)
                continue
            elif chunk_length < 100:
                max_length = max(50, int(chunk_length * 1.2))
                min_length = max(20, int(chunk_length * 0.4))
            elif chunk_length < 300:
                max_length = base_max_length
                min_length = base_min_length
            else:
                max_length = min(400, int(chunk_length * 0.7))  # Larger for long texts
                min_length = min(100, int(chunk_length * 0.25))

            logger.info(f"Chunk length: {chunk_length}, max_length: {max_length}, min_length: {min_length}")

            inputs = tokenizer("summarize: " + chunk, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                summary_ids = summarizer_model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=1.0,  # Encourage longer outputs
                    num_beams=6,  # Better quality
                    early_stopping=True,
                    no_repeat_ngram_size=3  # Avoid repetition
                )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary.lower())  # Convert to lowercase

        # Combine summaries
        final_summary = " ".join(summaries)
        logger.info(f"Final summary: {final_summary}")
        return final_summary
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        return f"failed to summarize: {str(e)}"

# Processing Functions
def process_audio_to_summary(audio_path, progress=gr.Progress()):
    """Convert audio to lowercase text and summarize in lowercase."""
    if not audio_path:
        return "no file uploaded", ""

    print(f"Received audio path: {audio_path}")
    start_time = time.time()
    print(f"Starting transcription for {audio_path}...")
    progress(0, desc="starting transcription...")
    
    try:
        # Validate file type
        mime_type, _ = mimetypes.guess_type(audio_path)
        print(f"Detected MIME type: {mime_type}")
        valid_mime_types = {'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/flac', 'audio/ogg', 'audio/aac'}
        valid_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.aac'}
        if (mime_type not in valid_mime_types) or not any(audio_path.lower().endswith(ext) for ext in valid_extensions):
            return f"invalid file type: {audio_path} (mime: {mime_type})", ""

        # Convert to WAV
        with file_lock:
            if not audio_path.lower().endswith(".wav"):
                audio_path = convert_to_wav(audio_path)
                progress(0.1, desc="audio converted to wav")

        # Load and preprocess audio
        print("Loading and preprocessing audio...")
        speech, sr = librosa.load(audio_path, sr=16000)
        speech = preprocess_audio(speech)
        progress(0.2, desc="audio loaded and preprocessed")

        # Split into chunks
        chunk_length = 10 * sr
        chunks = [speech[i:i + chunk_length] for i in range(0, len(speech), chunk_length)]
        print(f"Audio split into {len(chunks)} chunks.")
        progress(0.3, desc=f"split into {len(chunks)} chunks")

        # Transcribe in parallel
        transcriptions = []
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
            futures = [executor.submit(process_audio_segment, chunk) for chunk in chunks]
            for i, future in enumerate(futures, 1):
                transcriptions.append(future.result())
                print(f"Processed chunk {i}/{len(chunks)}")
                progress((0.3 + 0.6 * i / len(chunks)), desc=f"processing chunk {i}/{len(chunks)}")

        # Combine transcription in lowercase
        final_transcription = " ".join(transcriptions).strip()
        progress(0.95, desc="combining transcriptions")

        # Clean up
        with file_lock:
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
                print("Temporary file cleaned up.")

        duration = time.time() - start_time
        print(f"Transcription completed in {duration:.2f} seconds!")
        progress(1.0, desc="transcription complete!")

        # Summarize the transcription
        summary = summarize_text(final_transcription)
        return f"transcription completed in {duration:.2f} seconds:\n\n{final_transcription}", f"summary:\n\n{summary}"
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return f"error: {str(e)}", ""

def process_text_to_summary(input_text):
    """Summarize input text directly in lowercase."""
    if not input_text:
        return "no text provided"
    summary = summarize_text(input_text.lower())  # Convert input to lowercase
    return f"text summarized:\n\n{summary}"

# Gradio UI
with gr.Blocks(css="""
    body { background-color: #121212; color: white; }
    .gradio-container { background-color: #1e1e1e; padding: 30px; border-radius: 15px; }
    .title { color: #ffffff; font-size: 30px; font-weight: bold; text-align: center; margin-bottom: 20px; }
    .description { color: #b0b0b0; font-size: 16px; margin-bottom: 40px; text-align: center; }
    #audio-box, #text-box, #transcript-box, #summary-box { 
        background-color: #333333; color: white; border-radius: 8px; padding: 15px; 
        transition: all 0.3s ease; 
    }
    #audio-box:hover, #text-box:hover, #transcript-box:hover, #summary-box:hover { 
        background-color: #444444; box-shadow: 0 0 15px rgba(255, 255, 255, 0.1); 
    }
    .gradio-button { transition: all 0.3s ease; }
    .gradio-button:hover { background-color: #555555; transform: scale(1.05); }
""") as interface:
    gr.Markdown("# audio & text summarization tool")
    gr.Markdown("choose an option: upload an audio file to transcribe and summarize, or input text directly for summarization. all output in lowercase.")
    
    with gr.Tab("audio to summary"):
        audio_input = gr.Audio(
            sources=["upload"],
            type="filepath",
            label="upload audio file",
            elem_id="audio-box"
        )
        audio_submit_btn = gr.Button("summarize audio")
        with gr.Row():
            audio_transcript_output = gr.Textbox(label="transcription", lines=5, elem_id="transcript-box")
            audio_summary_output = gr.Textbox(label="summary", lines=5, elem_id="summary-box")
        
        audio_submit_btn.click(
            fn=process_audio_to_summary,
            inputs=audio_input,
            outputs=[audio_transcript_output, audio_summary_output]
        )
    
    with gr.Tab("text to summary"):
        text_input = gr.Textbox(
            label="enter text to summarize",
            lines=5,
            placeholder="paste your text here...",
            elem_id="text-box"
        )
        text_submit_btn = gr.Button("summarize text")
        text_summary_output = gr.Textbox(label="summary", lines=5, elem_id="summary-box")
        
        text_submit_btn.click(
            fn=process_text_to_summary,
            inputs=text_input,
            outputs=text_summary_output
        )

if __name__ == "__main__":
    print("Launching Gradio interface...")
    interface.launch(
        share=True, 
        server_name="127.0.0.1",
        server_port=7861,
        debug=True
    )
    print("Interface launched successfully! Access it via the provided URL.") 