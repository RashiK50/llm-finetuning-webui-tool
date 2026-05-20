"""
gguf_exporter.py
────────────────
Converts a saved merged model directory to GGUF format using llama.cpp's
convert_hf_to_gguf.py script, then pushes the resulting .gguf file to a
Hugging Face repo.

Requirements (add to requirements.txt):
  - llama-cpp-python   (pulls in llama.cpp headers; not strictly needed but handy)
  - huggingface_hub    (already used elsewhere)

The actual conversion is done by running:
    python <llama.cpp>/convert_hf_to_gguf.py <merged_model_dir> --outfile out.gguf
We auto-download the llama.cpp repo if it is not already present.
"""

import os
import subprocess
import traceback
from pathlib import Path

import shared_state as ss

# Where we store the cloned llama.cpp repo locally
LLAMA_CPP_DIR = "./llama_cpp_repo"
# Where GGUF output files land
GGUF_OUTPUT_DIR = "./gguf_outputs"


def _ensure_llama_cpp():
    """
    Clone llama.cpp if not already present.
    Returns the path to convert_hf_to_gguf.py.
    """
    convert_script = Path(LLAMA_CPP_DIR) / "convert_hf_to_gguf.py"
    if convert_script.exists():
        return str(convert_script)

    ss.state["gguf_push_message"] = "Cloning llama.cpp for GGUF conversion..."
    print("[GGUF] Cloning llama.cpp...")

    result = subprocess.run(
        [
            "git", "clone",
            "--depth", "1",
            "https://github.com/ggerganov/llama.cpp.git",
            LLAMA_CPP_DIR,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to clone llama.cpp:\n{result.stderr}"
        )

    # Install Python deps required by the conversion script
    req_file = Path(LLAMA_CPP_DIR) / "requirements.txt"
    if req_file.exists():
        subprocess.run(
            ["pip", "install", "-q", "-r", str(req_file)],
            check=False,
        )

    print("[GGUF] llama.cpp ready.")
    return str(convert_script)


def convert_and_push(merged_model_dir: str, hf_repo_id: str, quantisation: str = "f16") -> dict:
    """
    1. Convert a saved merged-model directory to GGUF.
    2. Push the .gguf file to a Hugging Face repo.

    Args:
        merged_model_dir : local path that contains the merged HF model
        hf_repo_id       : destination HF repo, e.g. "username/my-model-gguf"
        quantisation     : GGUF output type — "f16" (default) or "q8_0", "q4_k_m" etc.
                           Note: quantised types need the llama-quantize binary;
                           f16 works with convert_hf_to_gguf.py alone.

    Returns dict with keys: success (bool), message (str), gguf_path (str)
    """

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set. "
            "Add it to your .env file to enable Hugging Face push."
        )

    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)

    ss.state["gguf_push_status"]  = "converting"
    ss.state["gguf_push_message"] = "Starting GGUF conversion..."
    ss.state["gguf_pushed_repo"]  = ""

    try:
        # ── 1. Ensure llama.cpp is available ─────────────────────────────
        convert_script = _ensure_llama_cpp()

        # ── 2. Build output path ──────────────────────────────────────────
        model_slug = Path(merged_model_dir).name or "model"
        gguf_filename = f"{model_slug}_{quantisation}.gguf"
        gguf_path = str(Path(GGUF_OUTPUT_DIR) / gguf_filename)

        ss.state["gguf_push_message"] = f"Converting to GGUF ({quantisation})..."
        print(f"[GGUF] Converting {merged_model_dir} → {gguf_path}")

        # ── 3. Run conversion script ──────────────────────────────────────
        cmd = [
            "python", convert_script,
            merged_model_dir,
            "--outfile", gguf_path,
            "--outtype", quantisation,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            error_detail = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(
                f"GGUF conversion failed (exit {result.returncode}):\n{error_detail}"
            )

        print(f"[GGUF] Conversion complete: {gguf_path}")

        # ── 4. Push .gguf file to Hugging Face ────────────────────────────
        ss.state["gguf_push_status"]  = "pushing"
        ss.state["gguf_push_message"] = f"Uploading {gguf_filename} to {hf_repo_id}..."
        print(f"[GGUF] Uploading to HF repo: {hf_repo_id}")

        from huggingface_hub import HfApi
        api = HfApi()

        # Create repo if it doesn't exist yet
        api.create_repo(
            repo_id=hf_repo_id,
            token=hf_token,
            private=False,
            exist_ok=True,
        )

        api.upload_file(
            path_or_fileobj=gguf_path,
            path_in_repo=gguf_filename,
            repo_id=hf_repo_id,
            token=hf_token,
            commit_message=f"Add GGUF model ({quantisation})",
        )

        repo_url = f"https://huggingface.co/{hf_repo_id}"
        ss.state["gguf_push_status"]  = "pushed"
        ss.state["gguf_push_message"] = f"GGUF pushed successfully → {repo_url}"
        ss.state["gguf_pushed_repo"]  = repo_url
        print(f"[GGUF] Done: {repo_url}")

        return {
            "success":   True,
            "message":   f"GGUF converted and pushed to {repo_url}",
            "gguf_path": gguf_path,
        }

    except Exception as e:
        ss.state["gguf_push_status"]  = "error"
        ss.state["gguf_push_message"] = f"GGUF export failed: {str(e)}"
        traceback.print_exc()
        return {
            "success":   False,
            "message":   str(e),
            "gguf_path": "",
        }
