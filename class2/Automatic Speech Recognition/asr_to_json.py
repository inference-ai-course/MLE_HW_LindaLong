import os
import json
import yt_dlp
import whisper #install openai-whisper
import tempfile


def youtube_to_text(youtube_url):

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
            info = ydl.extract_info(youtube_url, download=True)
            audio_path = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
    
            #translate audio to text using Whisper
            model = whisper.load_model('base')
            result = model.transcribe(audio_path)

    return {
        "video_id": info['id'],
        "video_title": info['title'],
        "audio_transcription": result

    }



def asr_to_json(youtube_urls, output_file):

    with open(output_file, 'w', encoding='utf-8') as f:
            for youtube_url in youtube_urls:
                try:
                    result = youtube_to_text(youtube_url)
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"Failed to process {youtube_url}: {str(e)}")

    print(f"Results saved to {output_file}")


if __name__ == "__main__":

    youtube_urls = [
        #"https://youtu.be/ZDUOb42fS-Q?si=SpdsvmZ6BA5Gkviw",
        "https://youtu.be/PeMlggyqz0Y?si=U9OloR8VGchOR7IT",
        "https://www.youtube.com/shorts/wQCRZlbLgY4"
    ]

    asr_to_json(youtube_urls, "talks_transcripts.jsonl")
