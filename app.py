from flask import Flask, render_template, jsonify, request
import threading
import time

from aws_transcribe import *
from transcript_speaker_detect import *
from retriever import *
from video_clipper import *

app = Flask(__name__)
random_generate_reel_string = "ABCDE"
total_gen_tasks = 0
gen_tasks = dict()



# Function to run when giving the first prompt
def first_prompt_function(task_id):
    # Transcribe the video
    video_path = f"videos/{task_id}.mp4"
    output_video_path = f"static/output_videos/{task_id}_{str(gen_tasks[task_id]['num_videos'])}.mp4"
    transcribe_video(gen_tasks[task_id]["youtube-url"], task_id)
    refine_trascript("raw_transcripts/" + task_id + ".json", video_path, "refine_transcripts/" + task_id + ".json")
    gen_tasks[task_id]["llm-data"] = gpt4o_conv_qas(f"refine_transcripts/{task_id}.json", task_id)
    gen_tasks[task_id]["summary"] = gpt4o_conv_chain(gen_tasks[task_id]["prompt"], gen_tasks[task_id]["llm-data"])
    generate_clipped_video(video_path, gen_tasks[task_id]["llm-data"]["output_file"], output_video_path)
    gen_tasks[task_id]["video-url"] = f"static/output_videos/{task_id}_{gen_tasks[task_id]['num_videos']}.mp4"
    gen_tasks[task_id]["num_videos"] += 1
    gen_tasks[task_id]["status"] = "complete"
    return

# Function to run when giving the first prompt
def revise_prompt_function(task_id):
    video_path = f"videos/{task_id}.mp4"
    output_video_path = f"static/output_videos/{task_id}_{str(gen_tasks[task_id]['num_videos'])}.mp4"
    gen_tasks[task_id]["summary"] = gpt4o_conv_chain(gen_tasks[task_id]["feedback-chain"], gen_tasks[task_id]["llm-data"])
    generate_clipped_video(video_path, gen_tasks[task_id]["llm-data"]["output_file"], output_video_path)
    gen_tasks[task_id]["video-url"] = f"static/output_videos/{task_id}_{gen_tasks[task_id]['num_videos']}.mp4"
    gen_tasks[task_id]["num_videos"] += 1
    gen_tasks[task_id]["status"] = "complete"
    return

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route to serve the HTML page
@app.route('/about')
def about():
    return render_template('about.html')

# Generate a response
@app.route('/generate-reel', methods=['POST'])
def generate_reel():
    global total_gen_tasks,gen_tasks
    data = request.json
    task_id = random_generate_reel_string + str(total_gen_tasks)
    total_gen_tasks += 1
    gen_tasks[task_id] = {
        "youtube-url": data["youtube-url"],
        "prompt": data["prompt"],
        "llm-data": None,
        "num_videos": 0,
        "feedback-chain": "", 
        "status": "running"
    }
    thread = threading.Thread(target=first_prompt_function, args=(task_id,))
    thread.start()
    return {"task-id":task_id,"status":"running"}

# feedback 
@app.route('/feedback', methods=['POST'])
def feedback():
    global total_gen_tasks,gen_tasks
    data = request.json
    if "task-id" not in data or "feedback-prompt" not in data:
        return {"status":"Invalid Format"}
    if data["task-id"] not in gen_tasks:
        return {"status":"Invalid task-id"}
    gen_tasks[data["task-id"]]["feedback-chain"] = data["feedback-prompt"]
    gen_tasks[data["task-id"]]["status"] = "running"
    thread = threading.Thread(target=revise_prompt_function, args=(data["task-id"],))
    thread.start()    
    return {"task-id":data["task-id"],"status":"running"}

# check up on task
@app.route("/check-status", methods=["POST"])
def check_status():
    global total_gen_tasks,gen_tasks
    data = request.json
    if "task-id" not in data:
        return {"status":"Invalid Format"}
    if data["task-id"] in gen_tasks:
        return {"status":gen_tasks[data["task-id"]]["status"]}
    return {"status":"Invalid task-id"}

# get data after task completes
@app.route("/get-response", methods=["POST"])
def get_response():
    global total_gen_tasks,gen_tasks
    data = request.json
    if "task-id" not in data:
        return {"status":"Invalid task-id"}
    if gen_tasks[data["task-id"]]["status"]=="running":
        return {"task-id":data["task-id"],"status":"running"}
    video_url = gen_tasks[data["task-id"]].get("video-url",None)
    summary = gen_tasks[data["task-id"]].get("summary",None)
    if not video_url or not summary:
        return {"task-id":data["task-id"],"status":"error"}
    return {
        "task-id":data["task-id"],
        "status":"complete",
        "video-url":video_url,
        "summary":summary
    }

if __name__ == '__main__':
    app.run(debug=True)