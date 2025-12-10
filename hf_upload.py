from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()


LOCAL_MODEL_PATH = ""  
HF_REPO_ID =os.getenv("HF_REPO")

print(f"working")
api = HfApi()

print(f"Uploading folder '{LOCAL_MODEL_PATH}' ")

api.upload_folder(
    folder_path=LOCAL_MODEL_PATH,
    repo_id=HF_REPO_ID,
    commit_message="Initial upload "
)

print(f"upload successful")