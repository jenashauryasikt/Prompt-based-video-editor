from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
from copy import deepcopy
import os
import json

class VideoClipper():

    def __init__(self, video_file_path, json_file_path):
        
        self.video_file_path = video_file_path
        self.json_file_path = json_file_path

        return
    

    def open_json(self):

        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'r') as file:
                data = json.load(file)
                return data
        else:
            print(f"The file {self.json_file_path} does not exist.")
            return None
    

    def get_intervals(self, json_data):

        intervals = []
        transcript = json_data["transcript"]
        for chunk in transcript:
            if "start_time" in chunk and "end_time" in chunk:
                if chunk["start_time"] != None and chunk["end_time"] != None:
                    intervals.append([float(chunk["start_time"]) , float(chunk["end_time"])])
                else:
                    print("Undefined start or end time, skipping this chunk")
            else:
                print("Could not find start or end time for the chunk, skipping this chunk")

        return intervals
    

    def clip_and_join_video(self,intervals, output_video_path):
    
        clips = []
        
        for start, end in intervals:
            # Load the video and create a subclip
            clip = VideoFileClip(self.video_file_path).subclip(start, end)
            
            # Check if the clip has a valid duration
            if clip.duration > 0:
                # Apply a transition (fadein) to each clip
                clip = clip.fx(vfx.fadein, duration=0.5)
                clip = clip.fx(vfx.fadeout, duration=0.5)
                clips.append(clip)
                pass
            else:
                print(f"Clip from {start} to {end} is invalid and will be skipped.")

        if not clips:
            raise ValueError("No valid clips found.")
        
        # Concatenate all clips
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Write the result to a file
        final_clip.write_videofile(output_video_path, codec="libx264")

        return
    

    def refine_intervals(self, intervals):

        if len(intervals) == 0:
            print("No intervals founds")
            return None

        intervals =  sorted(intervals, key=lambda x: x[0])

        parameters = {
            "min_dur": 1,
            "merge_gap": 2,
            "start_margin": 0,
            "end_margin" : 0,
        }

        #Remove min dur
        temp = []
        for interval in intervals:
            if interval[1] - interval[0] > parameters["min_dur"]:
                temp.append(interval)
        intervals = deepcopy(temp)
        

        # Merge intervals
        merged_intervals = [intervals[0]]
        end_time = merged_intervals[-1][1]
        for idx in range(1, len(intervals)):
            if intervals[idx][0] <= end_time:
                if intervals[idx][1] > end_time:
                    end_time = intervals[idx][1]
            else:
                merged_intervals[-1][1] = end_time
                merged_intervals.append(intervals[idx])
                end_time = merged_intervals[-1][1]
        
        merged_intervals[-1][1] = end_time
        intervals = deepcopy(merged_intervals)


        # Resolve merge_gap
        merged_intervals = [intervals[0]]
        end_time = merged_intervals[-1][1]
        for idx in range(1, len(intervals)):
            if intervals[idx][0] <= end_time + parameters["merge_gap"]:
                if intervals[idx][1] > end_time:
                    end_time = intervals[idx][1]
            else:
                merged_intervals[-1][1] = end_time
                merged_intervals.append(intervals[idx])
                end_time = merged_intervals[-1][1]

        merged_intervals[-1][1] = end_time
        intervals = deepcopy(merged_intervals)

        return intervals
    

    def generate_clipped_video(self, output_video_path):

        json_data = self.open_json()
        intervals = self.get_intervals(json_data)
        refined_intervals = self.refine_intervals(intervals)
        self.clip_and_join_video(refined_intervals, output_video_path)

        return
    

def test():

    video_file_path = "downloads/football_vid.mp4"
    json_file_path = "transcripts/fitness.json"
    output_video_path = "results/clipped_video.mp4"

    clipper = VideoClipper(video_file_path=video_file_path, json_file_path=json_file_path)
    clipper.generate_clipped_video(output_video_path=output_video_path)

    pass


def generate_clipped_video(video_file_path, json_file_path, output_video_path):

    clipper = VideoClipper(video_file_path=video_file_path, json_file_path=json_file_path)
    clipper.generate_clipped_video(output_video_path=output_video_path)

    return







        
