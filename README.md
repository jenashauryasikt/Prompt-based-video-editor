# README

## Environment setup

1. Conda version >= 24.4.0
2. Create environment: ``conda create -n <myenv> python==3.9.19``
3. Activate environment: ``conda activate <myenv>``
4. Install dependencies: ``conda install -c conda-forge boto3 pytube moviepy huggingface_hub tiktoken spacy``
5. Install Spacy: ``python -m spacy download en_core_web_sm``
6. Set AWS CLI:

   1. Refer to installation for Windows, macOS, Linux. ``conda install -c conda-forge awscli`` should work.
   2. Check installation: ``aws --version``
   3. Give these programmatic permissions to IAM user:
      1. AmazonTranscribeFullAccess
      2. AmazonS3FullAccess
      3. AmazonRekognitionFullAccess
   4. Create access keys in the IAM users -> security credentials -> create access keys -> CLI -> add tag. Download access key and secret access key.
   5. In CLI, type ``aws configure`` and add necessary details.
7. Install dependencies: ``conda install -c conda-forge openai chromadb langchain langchain-core langchain-community langchain-text-splitters``
8. Install more dependencies: ``conda install -c conda-forge opencv``
9. Install more dependencies: ``conda install -c conda-forge numpy matplotlib scipy flask imageio imageio-ffmpeg``
10. Before execution, remember to set your Hugging Face access token and OpenAI access token as environment varibles:

    1. -``export OPENAI_API_KEY=your_key``
    2. -``export HF_ACCESS_TOKEN=your_key``

11. Sequence of python scripts in execution when the tool is run via ``app.py`` in the activated environment, open the link that pops up in the terminal for UI:

    1. -``aws_transcribe.py``
    2. -``transcript_speaker_detect.py`` calls from ``celeb_detection.py``
    3. -``retriever.py``
    4. -``video_clipper.py``
