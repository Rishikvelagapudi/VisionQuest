import os
import sys
from huggingface_hub import HfApi

def upload():
    token = input("Paste your Hugging Face Token (hf_...): ").strip()
    if not token:
        print("Error: Token is required.")
        return
    
    print("\nUploading full repository (including all subfolders) to Hugging Face Space...")
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=".",
        repo_id="rishikvelagapudi/visionquest",
        repo_type="space",
        ignore_patterns=[".git*", "*.pyc", "__pycache__", ".env", "venv", ".gemini", ".agents"]
    )
    print("\nSUCCESS! All subdirectories and code files are uploaded to Hugging Face Spaces!")

if __name__ == "__main__":
    upload()
