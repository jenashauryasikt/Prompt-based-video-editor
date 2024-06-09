import json
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from celeb_detection import CelebrityDetection


class TranscriptSpeakerDetector():

    def __init__(self, transcript_path):

        self.transcript_path = transcript_path

        return
    
    
    def open_json(self):

        if os.path.exists(self.transcript_path):
            with open(self.transcript_path, 'r') as file:
                data = json.load(file)
                return data
        else:
            print(f"The file {self.transcript_path} does not exist.")
            return None


    def get_speaker_stats(self, json_data, plot = False):

        speakers = {}
        transcript = json_data["transcript"]
        for chunk in transcript:
            if chunk["speaker"] in speakers:
                speakers[chunk["speaker"]] += float(chunk["end_time"]) - float(chunk["start_time"])
            else:
                speakers[chunk["speaker"]] = float(chunk["end_time"]) - float(chunk["start_time"])

        if plot:
            plt.plot(speakers.values())
            plt.ylabel('time duration')
            plt.xlabel("speaker id")
            plt.show()    

        return speakers


    def predict_speaker(self, speaker_stats, z_threshold = 1, plot = False):

        z_score = np.abs(stats.zscore(list(speaker_stats.values())))

        if plot:
            plt.plot(z_score)
            plt.ylabel('z_score')
            plt.xlabel("speaker id")
            plt.show()  

        outlier_indices = np.where(z_score > z_threshold)[0]
        speakers = [list(speaker_stats.keys())[int(i)] for i in outlier_indices]

        return speakers


    def get_interval(self, speaker_id, json_data, min_dur = 2, total_dur = 20):

        intervals = dict.fromkeys(speaker_id)
        transcript = json_data["transcript"]
        for chunk in transcript:
            if chunk["speaker"] in intervals:
                if float(chunk["end_time"]) - float(chunk["start_time"]) > min_dur:
                    if intervals[chunk["speaker"]] == None:
                        intervals[chunk["speaker"]] = [float(chunk["start_time"]), float(chunk["end_time"])]
                    else:
                        val = intervals[chunk["speaker"]]
                        val.append([float(chunk["start_time"]), float(chunk["end_time"])])


        return intervals
        

    def assign_speaker_to_id(self, speaker_dict, json_data, default = "interviewer"):

        transcript = json_data["transcript"]
        for chunk in transcript:
            if chunk["speaker"] in speaker_dict:
                chunk["speaker"] = speaker_dict[chunk["speaker"]]["name"]
            else:
                chunk["speaker"] = default

        return json_data


    def write_json(self, data, file_path):
        
        with open(file_path, "w") as outfile: 
            json.dump(data, outfile, indent=4)

        return
    

    def get_speaker_intervals(self):

        json_data = self.open_json()
        speaker_stats = self.get_speaker_stats(json_data, plot=False)
        possible_speaker = self.predict_speaker(speaker_stats, z_threshold=1.5, plot=False)
        speaker_intervals = self.get_interval(possible_speaker, json_data)
        
        return speaker_intervals
    

    def refine_transcript(self, detected_speakers, save_file_path):
        
        json_data = self.open_json()
        refined_transcript = self.assign_speaker_to_id(detected_speakers, json_data)
        self.write_json(refined_transcript, save_file_path)

        return
   
        
def test():

    file_path = "transcripts/Erik_ten_Hag_embargoed_pre-match_press_conference___Chelsea_v_Manchester_United.json"
    video_path = 'downloads/football_vid.mp4'  
    save_file_path = "transcripts/refined_transcript.json"

    trans_detector = TranscriptSpeakerDetector(transcript_path=file_path)
    speaker_intervals = trans_detector.get_speaker_intervals()
    print(speaker_intervals)


    aws_access_key = 'AKIA6ODU75CFLURRGWGH'
    aws_secret_key = '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'
    celeb_detector = CelebrityDetection(aws_access_key, aws_secret_key)
    detected_speakers  = celeb_detector.assign_speakers_return(video_path, speaker_intervals)
    print(detected_speakers)

    trans_detector.refine_transcript(detected_speakers=detected_speakers, save_file_path=save_file_path)

    return


def test2():

    file_path = "transcripts/Erik_ten_Hag_embargoed_pre-match_press_conference___Chelsea_v_Manchester_United.json"
    video_path = 'downloads/Carlos Alcaraz  Jannik Sinner React To Their Thrilling Indian Wells Encounter.mp4'  
    save_file_path = "transcripts/refined_transcript.json"

    trans_detector = TranscriptSpeakerDetector(transcript_path=file_path)
    speaker_intervals = trans_detector.get_speaker_intervals()
    print(speaker_intervals)


    aws_access_key = 'AKIA6ODU75CFLURRGWGH'
    aws_secret_key = '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'
    celeb_detector = CelebrityDetection(aws_access_key, aws_secret_key)
    detected_speakers  = celeb_detector.assign_speakers_return(video_path, speaker_intervals)
    print(detected_speakers)

    trans_detector.refine_transcript(detected_speakers=detected_speakers, save_file_path=save_file_path)

    return


def refine_trascript(transcript_file_path, video_file_path, save_file_path):

    trans_detector = TranscriptSpeakerDetector(transcript_path=transcript_file_path)
    speaker_intervals = trans_detector.get_speaker_intervals()

    aws_access_key = 'AKIAR57MVXEGMXIW5VUA'
    aws_secret_key = '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'
    celeb_detector = CelebrityDetection(aws_access_key, aws_secret_key)
    detected_speakers  = celeb_detector.assign_speakers_return(video_file_path, speaker_intervals)

    trans_detector.refine_transcript(detected_speakers=detected_speakers, save_file_path=save_file_path)

    return


# refine_trascript("transcripts/Erik_ten_Hag_embargoed_pre-match_press_conference___Chelsea_v_Manchester_United.json", "downloads/football_vid.mp4", "results/refined_transcript.json")
