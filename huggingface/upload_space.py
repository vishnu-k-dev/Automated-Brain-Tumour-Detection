"""
Upload the Brain Tumor Detection Gradio Space to Hugging Face.

Usage:
    1. pip install huggingface_hub
    2. huggingface-cli login   (paste your HF token)
    3. python upload_space.py

This will create the Space 'ZorroJurro/brain-tumor-detection' and upload:
    - README.md (Space metadata)
    - app.py (Gradio application)
    - requirements.txt (dependencies)
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ─── Configuration ───────────────────────────────────────────────────────────
REPO_ID = "Zorrojurro/brain-tumor-detection"
SPACE_DIR = Path(__file__).parent / "space"

def main():
    api = HfApi()

    # 1. Create the Space
    print(f"🚀 Creating Space: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False,
        )
        print(f"   ✅ Space ready: https://huggingface.co/spaces/{REPO_ID}")
    except Exception as e:
        print(f"   ⚠️  Space may already exist: {e}")

    # 2. Upload all Space files
    print("\n📄 Uploading Space files...")
    files_to_upload = [
        (SPACE_DIR / "README.md", "README.md"),
        (SPACE_DIR / "app.py", "app.py"),
        (SPACE_DIR / "requirements.txt", "requirements.txt"),
    ]

    for local_path, repo_path in files_to_upload:
        if local_path.exists():
            print(f"   📤 {repo_path}...")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="space",
            )
            print(f"   ✅ {repo_path} uploaded")
        else:
            print(f"   ❌ {local_path} not found, skipping")

    print(f"\n🎉 Done! View your Space at: https://huggingface.co/spaces/{REPO_ID}")
    print("   ⏳ The Space may take 2-5 minutes to build on first deploy.")


if __name__ == "__main__":
    main()
