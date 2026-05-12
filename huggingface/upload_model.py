"""
Upload the Brain Tumor CNN-ViT model to Hugging Face Hub.

Usage:
    1. pip install huggingface_hub
    2. huggingface-cli login   (paste your HF token)
    3. python upload_model.py

This will create the repo 'ZorroJurro/brain-tumor-cnn-vit' and upload:
    - config.json
    - README.md (model card)
    - model.py (self-contained architecture)
    - best_model.pth (~540MB checkpoint)
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ─── Configuration ───────────────────────────────────────────────────────────
REPO_ID = "Zorrojurro/brain-tumor-cnn-vit"
MODEL_REPO_DIR = Path(__file__).parent / "model_repo"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

def main():
    api = HfApi()

    # 1. Create the repository (if it doesn't exist)
    print(f"📦 Creating repository: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False,
        )
        print(f"   ✅ Repository ready: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"   ⚠️  Repo may already exist: {e}")

    # 2. Upload model card, config, and architecture
    print("\n📄 Uploading model files...")
    files_to_upload = [
        (MODEL_REPO_DIR / "README.md", "README.md"),
        (MODEL_REPO_DIR / "config.json", "config.json"),
        (MODEL_REPO_DIR / "model.py", "model.py"),
    ]

    for local_path, repo_path in files_to_upload:
        if local_path.exists():
            print(f"   📤 {repo_path}...")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"   ✅ {repo_path} uploaded")
        else:
            print(f"   ❌ {local_path} not found, skipping")

    # 3. Upload checkpoint (large file)
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"\n🧠 Uploading checkpoint ({size_mb:.0f} MB) — this may take a while...")
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo="best_model.pth",
            repo_id=REPO_ID,
            repo_type="model",
        )
        print("   ✅ Checkpoint uploaded!")
    else:
        print(f"\n❌ Checkpoint not found at {checkpoint_path}")
        print("   Please ensure checkpoints/best_model.pth exists.")

    print(f"\n🎉 Done! View your model at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
