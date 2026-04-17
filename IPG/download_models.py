import os
from huggingface_hub import snapshot_download

# ====== 路径配置（强烈建议用数据盘） ======
BASE_DIR = "/root/autodl-tmp"   # ⭐ 改成你的大盘

SD_PATH = os.path.join(BASE_DIR, "stable-diffusion-v1-5")
VAE_PATH = os.path.join(BASE_DIR, "sd-vae-ft-mse")

# HuggingFace repo
SD_REPO = "runwayml/stable-diffusion-v1-5"
VAE_REPO = "stabilityai/sd-vae-ft-mse"


def download_sd():
    print(f"\n📥 Downloading Stable Diffusion → {SD_PATH}")

    snapshot_download(
        repo_id=SD_REPO,
        local_dir=SD_PATH,
        local_dir_use_symlinks=False,
        resume_download=True,

        # ⭐⭐⭐ 只保留 diffusers 必要文件
        allow_patterns=[
            "unet/*",
            "vae/*",
            "text_encoder/*",
            "tokenizer/*",
            "scheduler/*",
            "model_index.json"
        ]
    )

    print("✅ Stable Diffusion 下载完成")


def download_vae():
    print(f"\n📥 Downloading VAE → {VAE_PATH}")

    snapshot_download(
        repo_id=VAE_REPO,
        local_dir=VAE_PATH,
        local_dir_use_symlinks=False,
        resume_download=True
    )

    print("✅ VAE 下载完成")


if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)

    download_sd()
    download_vae()

    print("\n🎉 所有模型已下载完成！")