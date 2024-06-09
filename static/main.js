document.getElementById("chatbox").style.display="none";
var task_id = "";

function add_user_chat(chat_to_add) {
    var new_ele = document.createElement("div");
    new_ele.classList.add("mb-2");
    new_ele.classList.add("row");
    new_ele.classList.add("justify-content-end");
    new_ele.innerHTML = `<div class="chat-user col-auto me-1" style="max-width: 70%; text-align: center;">
    ` + chat_to_add + `
</div>
<div class="col-sm-1 text-start ps-0">
    <div class="chat-user-icon"><i class="bi bi-person"></i></div>
</div>`;
    document.getElementById("chat-store").appendChild(new_ele);
    setTimeout(load_chat_top,200);
}

function add_chat_response(video_url, summary) {
    //TODO: Update share and copy links
    var new_ele = document.createElement("div");
    new_ele.classList.add("mb-2");
    new_ele.classList.add("row");
    new_ele.classList.add("justify-content-start");
    new_ele.innerHTML = `<div class="col-sm-1 text-end pe-0">
    <div class="chat-response-icon">AI</div>
</div>
<div class="chat-response col-auto ms-1" style="max-width: 70%; text-align: center;">
    <video width="100%" controls>
        <source src="`+ video_url +`" type="video/mp4">
        Its not working
    </video>
    <div class="video-share-btns">
        <button class="btn">
            <i class="bi bi-clipboard"></i>
        </button>
        <button class="btn">
            <i class="bi bi-twitter"></i>
        </button>
        <button class="btn">
            <i class="bi bi-instagram"></i>
        </button>
        <button class="btn">
            <i class="bi bi-facebook"></i>
        </button>
    </div>
    `+summary+`
</div>`;
    document.getElementById("chat-store").appendChild(new_ele);
    setTimeout(load_chat_top,200);
}

function load_chat_top() {
    var contentDiv = document.getElementsByClassName("chats")[0];
    contentDiv.scrollTo({
        top: contentDiv.scrollHeight,
        left: 0,
        behavior: "smooth"
    });
}

async function get_response_data() {
    console.log(" in get_response_data");
    try{
        const response = await fetch("/get-response",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body: JSON.stringify({
                "task-id":task_id
            })
        });
        const result = await response.json();
        console.log("Success",result);
        if(result["status"]=="running"){
            wait_for_server();
        } else if(result["status"]=="complete"){
            add_chat_response(result["video-url"],result["summary"]);
        } else {
            console.log("Error",result);
        }
    } catch (error) {
        console.log("Error",error);
    }
}

async function wait_for_server() {
    //TODO: Add gif
    document.getElementById("feedback-text").disabled=true;
    document.getElementById("feedback-text").placeholder="Patience is the key...";
    console.log("in wait_for_server")
    const intervalId = setInterval(async () => {
        try{
            const response = await fetch("/check-status", {
                method:"POST",
                headers:{"Content-Type":"application/json"},
                body: JSON.stringify({
                    "task-id":task_id
                })
            });
            const result = await response.json();
            console.log("wait_for_server result",result);
            if(result["status"]=="complete"){
                document.getElementById("feedback-text").placeholder="Feedback Prompt";
                document.getElementById("feedback-text").disabled=false;
                clearInterval(intervalId);
                get_response_data();
            } else if(result["status"]!="running"){
                console.log("Error",result);
                clearInterval(intervalId);
            }
        } catch (error) {
            console.log("Error",error);
            clearInterval(intervalId);
        }

    }, 5000);
}

async function call_server_generate_reel(youtube_url, prompt) {
    try{
        const response = await fetch("/generate-reel", {
            method:"POST",
            body: JSON.stringify({
                "youtube-url":youtube_url,
                "prompt":prompt
            }),
            headers: {
                'Content-Type':"application/json"
            }
        });
        const result = await response.json();
        console.log("Success",result);
        task_id = result["task-id"];
        if(result["status"]=="complete") {
            get_response_data();
        } else {
            wait_for_server();
        }
    } catch (error) {
        console.log("Error",error);
    }
}

function generate_reel() {
    var youtube_url = document.getElementById("youtube-url-text").value.trim();
    var prompt = document.getElementById("prompt-text").value.trim();
    console.log(youtube_url + ":" + prompt)
    if(youtube_url=="" || prompt==""){
        document.getElementsByClassName("invalid-input")[0].style.display="block";
        setTimeout(()=>{
            document.getElementsByClassName("invalid-input")[0].style.display="none";
        },1500);
        return;
    }
    document.getElementById("starting-content").classList.add("fadeout");
    setTimeout(() => {
        document.getElementById("starting-content").style.display="none";
        document.getElementById("starting-content").style.opacity="0";
        document.getElementById("starting-content").classList.remove("fadeout");
        document.getElementById("chatbox").style.display="block";
        
        var chat_to_add = "<a href="+youtube_url+' class="og-video-link">Video URL</a>'
        add_user_chat(chat_to_add+": "+prompt);
        
        call_server_generate_reel(youtube_url, prompt);
    },500);
}

async function feedback_prompt() {
    var feedback_text = document.getElementById("feedback-text").value.trim();
    console.log("Im feedback_prompt:",feedback_text);
    if(feedback_text=="") {
        get_response_data();
        return;
    }
    document.getElementById("feedback-text").value = "";
    add_user_chat(feedback_text);
    try{
        const response = await fetch("feedback", {
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({
                "task-id":task_id,
                "feedback-prompt":feedback_text
            })
        });
        const result = await response.json();
        console.log("Success",result);
        if(result["status"]=="running") {
            wait_for_server();
        } else {
            console.log("Error",result)
        }
    } catch (error) {
        console.log("Error:",error)
    }
}