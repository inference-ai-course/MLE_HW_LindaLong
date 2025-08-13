import os
import json
from unittest import result
import pytesseract
from pytube import YouTube
import yt_dlp
import whisper #install openai-whisper
from PIL import Image
import subprocess
import tempfile
#brew install ffmpeg


def collect_transcriptions_youtube_url(youtube_url):

    with tempfile.TemporaryDirectory() as temp_dir:
        # Download audio files
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
    
            #translate audio to text using Whisper
            model = whisper.load_model('base')
            result = model.transcribe(audio_path)

    return {
        "video_id": info['id'],
        "video_title": info['title'],
        "audio_transcription": result

    }




youtube_urls = [

        #"https://www.youtube.com/watch?v=..."
        #"https://www.youtube.com/watch?v=9bZkp7q19f0",  # Example URL
        "https://www.youtube.com/shorts/wQCRZlbLgY4"
    ]



with open('talks_transcripts.jsonl', 'w', encoding='utf-8') as f:
        for url in youtube_urls:
            try:
                result = collect_transcriptions_youtube_url(url)
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                print(f"Finished transcription: {result['video_title']}")
            except Exception as e:
                print(f"Failed to process {url}: {str(e)}")

print(f"Results saved to talks_transcripts.jsonl")

     