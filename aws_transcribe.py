import os
import re

import boto3
import time
import requests
from pytube import YouTube
from moviepy.editor import VideoFileClip
import json
from datetime import datetime
from collections import defaultdict

import spacy

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Function to sanitize the job name
def sanitize_name(name):
    return re.sub(r'[^0-9a-zA-Z._-]', '_', name)

# Function to download video and extract audio
def download_video_and_extract_audio(youtube_url, video_file_path, audio_file_path):
    yt = YouTube(youtube_url)
    video = yt.streams.filter(progressive=True, file_extension='mp4').first()
    video.download(filename=video_file_path)

    video_clip = VideoFileClip(video_file_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_file_path)
    audio_clip.close()
    video_clip.close()

# Function to upload audio to Amazon S3
def upload_to_s3(bucket_name, source_file_name, destination_file_name):
    s3_client = boto3.client('s3')
    s3_client.upload_file(source_file_name, bucket_name, destination_file_name)
    return f"s3://{bucket_name}/{destination_file_name}"

# Function to transcribe audio with AWS Transcribe
def transcribe_audio(bucket_name, audio_file_uri, job_name, media_format):
    transcribe_client = boto3.client('transcribe')
    
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': audio_file_uri},
        MediaFormat=media_format,
        LanguageCode='en-US',
        Settings={
            'ShowSpeakerLabels': True,
            'MaxSpeakerLabels': 10  # Adjust based on the expected number of speakers
        }
    )

    # Wait for the transcription job to complete
    while True:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print("Waiting for transcription job to complete...")
        time.sleep(15)

    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        response = requests.get(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
        return response.json()
    else:
        raise Exception("Transcription job failed")

# Function to format the transcript into JSON
def format_transcript(transcript):
    formatted_transcript = []
    segments = transcript.get('results', {}).get('speaker_labels', {}).get('segments', [])
    items = transcript.get('results', {}).get('items', [])

    for segment in segments:
        speaker = segment.get('speaker_label', 'unknown')
        start_time = segment.get('start_time', '0.0')
        end_time = segment.get('end_time', '0.0')
        words = []

        for item in items:
            if item['type'] == 'pronunciation' and 'start_time' in item and 'end_time' in item:
                if float(segment['start_time']) <= float(item['start_time']) <= float(segment['end_time']):
                    word_content = item.get('alternatives', [{}])[0].get('content', item.get('content', ''))
                    words.append(word_content)

        transcript_text = " ".join(words)
        formatted_transcript.append({
            "speaker": speaker,
            "start_time": start_time,
            "end_time": end_time,
            "transcript": transcript_text
        })

        for i, entry in enumerate(formatted_transcript):
            curr_speaker = entry['speaker']
            if i < len(formatted_transcript)-1:
                next_speaker = formatted_transcript[i+1]['speaker']
                while(next_speaker == curr_speaker):
                    entry['transcript'] = entry['transcript'] + ' ' + formatted_transcript[i+1]['transcript']
                    entry['end_time'] = formatted_transcript[i+1]['end_time']
                    del formatted_transcript[i+1]
                    if i < len(formatted_transcript)-1:
                        next_speaker = formatted_transcript[i+1]['speaker']
                    else:
                        break

    return formatted_transcript

def extract_human_names(transcript_text):
    doc = nlp(transcript_text)
    human_names = set()

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            human_names.add(ent.text)
    
    return list(human_names)

def transcribe_video(youtube_url, task_id, bucket_name="press-hacks"):
    yt = YouTube(youtube_url)
    pre_title = yt.title
    title = sanitize_name(pre_title.replace(' ', '_'))
    audio_format = 'mp3'
    video_file_path = "videos/" + task_id + '.mp4'
    audio_file_path = "audios/" + task_id + '.' + audio_format
    final_audio_path = audio_file_path

    # Download video and extract audio
    download_video_and_extract_audio(youtube_url, video_file_path, audio_file_path)

    # Upload audio file to Amazon S3
    audio_file_uri = upload_to_s3(bucket_name, final_audio_path, final_audio_path)

    # Transcribe audio with AWS Transcribe
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    job_id = title+"_TranscriptionJob_"+str(timestamp)
    job_name = sanitize_name(job_id)
    transcript = transcribe_audio(bucket_name, audio_file_uri, job_name, audio_format)

    # Format transcript
    formatted_transcript = format_transcript(transcript)

    # Calculate average speaking time and identify interviewee speaker ids
    speaker_times = defaultdict(float)
    for entry in formatted_transcript:
        speaker_times[entry['speaker']] += float(entry['end_time']) - float(entry['start_time'])

    average_speaking_time = sum(speaker_times.values()) / len(speaker_times)
    interim_speaker_info = [(speaker,time) for speaker,time in speaker_times.items() if time > average_speaking_time]
    interim_speaker_info.sort(reverse=True,key=lambda x:(x[1]))
    interviewee_possible_ids = [item[0] for item in interim_speaker_info]

    # Extract list of human names spoken in the transcript
    full_transcript_text = " ".join([entry['transcript'] for entry in formatted_transcript])
    human_names = extract_human_names(full_transcript_text)

    # Add metadata
    metadata = {
        "interviewee_possible_ids": interviewee_possible_ids,
        "youtube_video_title": pre_title,
        "human_names": human_names
    }

    json_filepath = "raw_transcripts/" + task_id + '.json'

    # Save transcript to JSON file
    with open(json_filepath, "w") as json_file:
        json.dump({"metadata": metadata, "transcript": formatted_transcript}, json_file, indent=4)
        
    print(f"Transcript saved to {json_filepath}")
    print(f"Video saved to {video_file_path}")
