# README

## Environment setup

1. Conda version >= 24.4.0
2. Create environment: ``conda create -n <myenv> python==3.9.19``
3. Activate environment: ``conda activate <myenv>``
4. Install dependencies: ``conda install -c conda-forge boto3 pytube moviepy huggingface_hub tiktoken spacy``
5. Install Spacy: ``python -m spacy download en_core_web_sm``
6. Set AWS CLI:
   1. Refer to installation for Windows, macOS, Linux
   2. Check installation: ``aws --version``
   3. Give these programmatic permissions to IAM user:
      1. AmazonTranscribeFullAccess
      2. AmazonS3FullAccess
      3. AmazonRekognitionFullAccess
   4. Create access keys in the IAM users -> security credentials -> create access keys -> CLI -> add tag. Download access key and secret access key.
   5. In CLI, type ``aws configure`` and add necessary details.
7. Install dependencies: ``conda install -c conda-forge openai chromadb langchain langchain-core langchain-community langchain-text-splitters``
8. Install more dependencies: ``pip install opencv-python``
9. Install more dependencies: ``conda install -c conda-forge numpy matplotlib scipy flask imageio imageio-ffmpeg``
10.
