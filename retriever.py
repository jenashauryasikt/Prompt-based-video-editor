import os
import re
import json
from datetime import datetime
from collections import defaultdict
import uuid

import spacy
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory

from huggingface_hub import login

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

os.environ['OPENAI_API_KEY'] = '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'

hf_access_token = "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"
os.environ["HF_ACCESS_TOKEN"] = hf_access_token

login(hf_access_token)

# Load the JSON file and extract metadata and transcript
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    metadata = data.get('metadata', {})
    transcript = data.get('transcript', [])
    return metadata, transcript

# Function to sanitize the job name
def sanitize_name(name):
    return re.sub(r'[^0-9a-zA-Z._-]', '_', name)

# Process the transcript and store as vector store
def store_transcript_as_vectorstore(transcript):
    # Create individual chunks from each transcript entry pair of Q-A
    documents = []
    for i, item in enumerate(transcript):
        content = None
        if item['speaker'] == "interviewer":
            if i < len(transcript)-1:
                content = item['transcript'] + ' ' + transcript[i+1]['transcript']
            else:
                content = item['transcript']
        else:
            if i==0:
                content = item['transcript']
            else:
                pass
        if content is not None:
            documents.append(Document(page_content=content, metadata=transcript[i+1]))

    # documents = [Document(page_content=item['transcript'], metadata=item) for item in transcript]
    embeddings = OpenAIEmbeddings()
    
    store = Chroma.from_documents(
        documents,
        embeddings,
        ids=[f"{index}" for index, item in enumerate(documents)],
        collection_name="Press-conf-embeds",
        persist_directory='db',
    )
    store.persist()

    return store

# Decide which kind of q-a pipeline to enter
def decide_long_short(question, model="gpt-4o"):
    template = """You are a bot that decides whether to give short outputs or long outputs. 
    If the user question demands for a collection of short portions that may not have much relevance between them but follow a theme, like highlights or quotes collection, reply the word 'short'.
    If the user question demands for parts of a video that require question-answer pairs that should have relevance with the context of the prompt, reply the word 'long'.
    You have to answer from the two reply choices without paraphrasing any of the replies given.
    
    Question: {question}"""
    PROMPT = PromptTemplate(
        template=template, input_variables=["question"]
    )
    llm = ChatOpenAI(temperature=0, model=model)
    decider = LLMChain(
        llm=llm,
        prompt=PROMPT
        )
    reply = decider.run(question)
    return reply

# Process user prompts and retrieve relevant portions of the transcript
def process_prompt(reply, store, memory, model="gpt-4o"):
    store.persist() #check
    if reply=='long':
        system_template = r"""You are a bot that finds the most relevant portions of the transcript by focusing on the objectivity of the context provided by the user prompt.
        Define a question as a spoken line with the speaker id as interviewer. Define an answer as a spoken line by the person being interviewed that is not the interviewer. 
        Each document is a question-answer pair. If the document is relevant, retrieve the whole document.
        Retrieve complete question-answer pairs that are objectively most relevant to the context of the prompt. Limit such pairs to maximum 2 pairs, can be 1 but not 0.
        You do not paraphrase the portions that you extract and present them as they are without any extra description provided by you.
        
        {context}
        """
    if reply=='short':
        system_template = r"""You are a bot that finds the most relevant portions of the transcript by focusing on the subjectivity of the theme of the user prompt.
        You can return multiple relevant portions of the transcript that fit the user prompt.
        Do not retrieve portions that are full of filler words or do not make coherent sense as a phrase, sentence or sentences.
        The portions that you retrieve from the transcript do not need to fully cover page content of retrieved documents, they can and likely will be a part of the page content.
        Prompt demands that you need to look out for include, but not limited to: 
        (a) Highlights: Key moments or statements that stand out or encapsulate the main themes of the interview
        (b) Quotes: Verbatim quotes from the person being interviewed. This person is not the interviewer.
        (c) Emotional moments, Funny moments etc.
        The retrieved documents should span between 15% o 20% of the total time of the original video as detailed in the transcript. The span of retrieved documents should not exceed 3 minutes.
        You do not paraphrase the portions that you extract and present them as they are without any extra description provided by you.


        {context}

        Question: {question}
        """

    user_template = "Question: ```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)

    llm = ChatOpenAI(temperature=0, model=model)
    chain_qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=store.as_retriever(),
        memory=memory,
        return_source_documents=True,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        output_key='source_documents'
    )

    return chain_qa


# Process user prompts and retrieve relevant portions of the transcript
# def short_process_prompt(store, question, model="gpt-4o"):
#     store.persist() #check
#     system_template = r"""You are a bot that finds the most relevant portions of the transcript following the theme of the user prompt.
#     You can return multiple relevant portions of the transcript that fit the user prompt.
#     The portions that you retrieve from the transcript do not need to fully cover page content of retrieved documents, they can and likely will be a part of the page content.
#     You do not paraphrase the portions that you extract and present them as they are without any extra description provided by you.

#     {context}

#     Question: {question}
#     """

#     PROMPT = PromptTemplate(
#         template=system_template, input_variables=["context", "question"]
#     )

#     llm = ChatOpenAI(temperature=0, model=model)
#     # chain_qa = RetrievalQA.from_chain_type(
#     #     llm=llm,
#     #     chain_type="stuff",
#     #     retriever=store.as_retriever(),
#     #     chain_type_kwargs={"prompt": PROMPT, },
#     #     return_source_documents=True,
#     # )
#     responses = {'source_documents': [], 'results': []}
#     source_documents = []
#     results = []
#     for doc in store.as_retriever().get_relevant_documents(question):
#         context = doc.page_content
#         print(doc.page_content)
#         prompt = PROMPT.format(context=context, question=question) # CHECK
#         # print("Prompt in short is:",prompt)
#         response = llm.invoke(prompt) # CHECK
#         # print("Respones in short is:",response)
#         source_documents.append(doc)
#         results.append(response)
#     responses['source_documents'] = source_documents
#     responses['results'] = results

#     return responses

def conversational_prompt(chain_qa, prompt):
    response = chain_qa({"question": prompt})
    return response

def remove_duplicate_dicts(dict_list):
    seen = set()
    unique_list = []
    for d in dict_list:
        dict_tuple = frozenset(d.items())
        if dict_tuple not in seen:
            seen.add(dict_tuple)
            unique_list.append(d)
    return unique_list

def is_coherent(transcript):
    # Process the transcript using spaCy
    doc = nlp(transcript)
    # Count the number of meaningful tokens
    entity_tokens = set()
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE"}:
            entity_tokens.update(ent)
    
    filtered_tokens = [token for token in doc if token not in entity_tokens]
    meaningful_tokens = [token for token in filtered_tokens if token.is_alpha and not token.is_stop]
    # Check if the transcript contains at least three meaningful token
    return len(meaningful_tokens) > 2

# Generate output JSON with metadata and selected transcript lines
def generate_output_json(reply, response, metadata, original_transcript, output_file):
    selected_lines = []
    for doc in response['source_documents']:
        for item in original_transcript:
            # long
            if reply=='long':
                if item['transcript'] in doc.page_content:
                    selected_lines.append(item)
            # short
            if reply=='short':
                if is_coherent(doc.page_content):
                    selected_lines.append(doc.metadata)

    unique_lines = remove_duplicate_dicts(selected_lines)
    output_metadata = {
        "interviewee_possible_ids": metadata.get('interviewee_possible_ids', []),
        "original_youtube_video_title": metadata.get('youtube_video_title', 'Unknown Title'),
    }
    
    output_data = {
        "metadata": output_metadata,
        "transcript": unique_lines
    }
    
    with open(output_file, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)
    print(f"Output saved to {output_file}")



def gpt4o_conv_qas(json_path, task_id):
    # session_id = str(uuid.uuid4())
    metadata, transcript = load_json(json_path)
    store = store_transcript_as_vectorstore(transcript)

    # SQL for chat memory 
    long_db_uri = 'sqlite:///chat_memory_long.db'
    long_session_id = str(uuid.uuid4())
    long_memory = ConversationBufferMemory(
        chat_memory=SQLChatMessageHistory(
            session_id=long_session_id,
            connection_string=long_db_uri,
            table_name="chat_messages"
        ),
        memory_key="chat_history",
        return_messages=True
    )

    short_db_uri = 'sqlite:///chat_memory_short.db'
    short_session_id = str(uuid.uuid4())
    short_memory = ConversationBufferMemory(
        chat_memory=SQLChatMessageHistory(
            session_id=short_session_id,
            connection_string=short_db_uri,
            table_name="chat_messages"
        ),
        memory_key="chat_history",
        return_messages=True
    )

    long_chain_qa = process_prompt(reply='long', store=store, memory=long_memory)
    short_chain_qa = process_prompt(reply='short', store=store, memory=short_memory)

    output_file = f"clips/clipped_{task_id}.json"

    return {
        "long_chain_qa": long_chain_qa,
        "short_chain_qa": short_chain_qa,
        "metadata": metadata,
        "transcript": transcript,
        "output_file": output_file
    }

def gpt4o_conv_chain(question, llm_data):
    reply = decide_long_short(question)
    if reply == 'long':
        response = conversational_prompt(llm_data["long_chain_qa"], question)
    elif reply == 'short':
        response = conversational_prompt(llm_data["short_chain_qa"], question)
    generate_output_json(reply, response, llm_data["metadata"], llm_data["transcript"], llm_data["output_file"])



# def gpt4o_conv_chain(json_path):
#     # session_id = str(uuid.uuid4())
#     metadata, transcript = load_json(json_path)
#     store = store_transcript_as_vectorstore(transcript)
    
#     # SQL for chat memory 
#     long_db_uri = 'sqlite:///chat_memory_long.db'
#     long_session_id = str(uuid.uuid4())
#     long_memory = ConversationBufferMemory(
#         chat_memory=SQLChatMessageHistory(
#             session_id=long_session_id,
#             connection_string=long_db_uri,
#             table_name="chat_messages"
#         ),
#         memory_key="chat_history",
#         return_messages=True
#     )

#     short_db_uri = 'sqlite:///chat_memory_short.db'
#     short_session_id = str(uuid.uuid4())
#     short_memory = ConversationBufferMemory(
#         chat_memory=SQLChatMessageHistory(
#             session_id=short_session_id,
#             connection_string=short_db_uri,
#             table_name="chat_messages"
#         ),
#         memory_key="chat_history",
#         return_messages=True
#     )

#     question = input("Enter your prompt (type 'end' to quit the process): ")
#     reply = decide_long_short(question)

#     long_chain_qa = process_prompt(reply='long', store=store, memory=long_memory)
#     short_chain_qa = process_prompt(reply='short', store=store, memory=short_memory)

#     pre_title = metadata.get('youtube_video_title', 'output')
#     title = sanitize_name(pre_title.replace(' ', '_'))
#     output_file = f"clipped_{title}.json"

#     while True:
#         if question.lower() == "end":
#             break
#         if reply == 'long':
#             response = conversational_prompt(long_chain_qa, question)
#         elif reply == 'short':
#             response = conversational_prompt(short_chain_qa, question)
#         generate_output_json(reply, response, metadata, transcript, output_file)
#         # prepare next query
#         question = input("Enter your desired modification (type 'end' to quit the process): ")
#         reply = decide_long_short(question)
            
# Example usage
# if __name__ == "__main__":
#     json_path = 'files/Erik_ten_Hag_embargoed_pre-match_press_conference___Chelsea_v_Manchester_United.json'  # Update this path to your input JSON file
#     gpt4o_conv_chain(json_path)