import os
import cv2
import json
import boto3
import tempfile
from collections import defaultdict


class CelebrityDetection:
    def __init__(self, aws_access_key, aws_secret_key, aws_region='us-west-2', confidence_threshold=85, frames_per_speaker=20, debug=False):
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
        os.environ['AWS_DEFAULT_REGION'] = aws_region
        self.client = boto3.client('rekognition')
        self.confidence_threshold = confidence_threshold
        self.frames_per_speaker = frames_per_speaker
        self.debug = debug

    def extract_frames(self, video_path, intervals, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_paths = {}
        
        for speaker, times in intervals.items():
            speaker_frames = []
            total_intervals = len(times)
            frames_needed = self.frames_per_speaker  
            
            if total_intervals > frames_needed:
                selected_intervals = [times[i] for i in range(0, total_intervals, total_intervals // frames_needed)]
                selected_intervals = selected_intervals[:frames_needed]  
            else:
                selected_intervals = times
            
            frames_per_interval = max(1, frames_needed // len(selected_intervals))  
            
            for time_range in selected_intervals:
                start, end = time_range if isinstance(time_range, list) else (time_range, time_range)
                start_frame = int(start * fps)
                end_frame = int(end * fps)
                frame_step = max(1, (end_frame - start_frame) // frames_per_interval)  
                
                for frame_num in range(start_frame, end_frame, frame_step):
                    if len(speaker_frames) >= frames_needed:
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if ret:
                        frame_filename = f"{output_dir}/{speaker}_frame_{frame_num}.jpg"
                        cv2.imwrite(frame_filename, frame)
                        speaker_frames.append(frame_filename)
            
            frame_paths[speaker] = speaker_frames[:frames_needed]  
        
        # cap.close()
        cap.release()
        return frame_paths

    def detect_celebrities(self, frame_paths):
        speaker_celebrities = {}
        placeholder_counter = 1

        for speaker, frames in frame_paths.items():
            celeb_counts = defaultdict(int)
            confidence_counts = defaultdict(list)
            
            for frame in frames:
                with open(frame, 'rb') as image:
                    response = self.client.recognize_celebrities(Image={'Bytes': image.read()})
                    if response['CelebrityFaces']:
                        for celebrity in response['CelebrityFaces']:
                            celeb_name = celebrity['Name']
                            celeb_confidence = celebrity['MatchConfidence']
                            if celeb_confidence >= self.confidence_threshold:
                                celeb_counts[celeb_name] += 1
                                confidence_counts[celeb_name].append(celeb_confidence)
                                if self.debug:
                                    print(f"Speaker: {speaker}, Frame: {frame}, Celebrity: {celeb_name}, Confidence: {celeb_confidence}")
                            else:
                                if self.debug:
                                    print(f"Speaker: {speaker}, Frame: {frame}, Celebrity: {celeb_name}, Confidence: {celeb_confidence} - Below Threshold")
                    else:
                        if self.debug:
                            print(f"Speaker: {speaker}, Frame: {frame}, Celebrity: None, Confidence: None")
            
            if celeb_counts:
                most_common_celeb = max(celeb_counts, key=celeb_counts.get)
                avg_confidence = sum(confidence_counts[most_common_celeb]) / len(confidence_counts[most_common_celeb])
                speaker_celebrities[speaker] = {
                    'name': most_common_celeb,
                }
            else:
                speaker_celebrities[speaker] = None
        
        # Assign placeholders to speakers with None
        identified_speakers = [speaker for speaker in speaker_celebrities if speaker_celebrities[speaker] is not None]
        for speaker in sorted(frame_paths.keys()):
            if speaker_celebrities[speaker] is None:
                while f"Speaker {placeholder_counter}" in [speaker_celebrities[s]['name'] for s in identified_speakers]:
                    placeholder_counter += 1
                speaker_celebrities[speaker] = {
                    'name': f"Speaker {placeholder_counter}",
                }
                placeholder_counter += 1
        
        return speaker_celebrities

    def save_output(self, speaker_celebrities, output_file):
        with open(output_file, 'w') as file:
            json.dump(speaker_celebrities, file, indent=4)

    def assign_speakers_save(self, video_path, intervals_path, output_file):
        with open(intervals_path, 'r') as file:
            intervals = json.load(file)

        with tempfile.TemporaryDirectory() as temp_dir:
            frame_paths = self.extract_frames(video_path, intervals, temp_dir)
            speaker_celebrities = self.detect_celebrities(frame_paths)
            if self.debug:
                print(speaker_celebrities)
            self.save_output(speaker_celebrities, output_file)
            if self.debug:
                print(f'Final output saved to {output_file}')

        return speaker_celebrities

    def assign_speakers_return(self, video_path, intervals):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_paths = self.extract_frames(video_path, intervals, temp_dir)
            speaker_celebrities = self.detect_celebrities(frame_paths)

        return speaker_celebrities

def test1():
    ## Enter your AWS credentials here
    aws_access_key = 'your_key'
    aws_secret_key = 'your_key'
    video_path = 'downloads/Post-Qualifying Drivers Press conference  2024 Monaco Grand Prix.mp4'
    intervals_path = 'speaker_intervals.json'
    output_file = 'final_output.json'

    detector = CelebrityDetection(aws_access_key, aws_secret_key, debug=True)
    detected_speakers = detector.assign_speakers_save(video_path, intervals_path, output_file)

def test2():
    # Enter your AWS credentials here
    aws_access_key = 'your_key'
    aws_secret_key = 'your_key'
    video_path = 'video/Tiger Woods returns to Augusta National  Press Conference  CBS Sports.mp4'
    intervals_path = 'speaker_intervals.json'
    output_file = 'final_output_tiger.json'

    with open(intervals_path, 'r') as file:
        intervals = json.load(file)

    detector = CelebrityDetection(aws_access_key, aws_secret_key, debug=False)
    detected_speakers = detector.assign_speakers_return(video_path, intervals)
    print(detected_speakers)
