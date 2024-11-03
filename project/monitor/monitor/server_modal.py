from pathlib import Path

import modal


def find_project_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        if (current_dir / ".root").exists():
            return current_dir
        current_dir = current_dir.parent
    raise RuntimeError("Could not find project root; no .root file found in parent directories")


REPO_BRANCH = "main"
REMOTE_ROOT = Path("/root/observatory")
LOCAL_ROOT = find_project_root() if modal.is_local() else REMOTE_ROOT


stub = modal.App("monitor-server")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        f"git clone -b {REPO_BRANCH} https://github.com/TransluceAI/observatory.git {REMOTE_ROOT}",
        "apt update && apt install -y curl && curl -LsSf https://astral.sh/uv/install.sh | sh",
        f". ~/.bashrc && cd {REMOTE_ROOT}/project/monitor && uv pip install --system -e .",
    )
    .copy_local_file(LOCAL_ROOT / ".env", REMOTE_ROOT / ".env")
    .env({"HF_HOME": "/root/.cache/huggingface"})
)

# Load the llama3.1-8b-instruct model
volume = modal.Volume.from_name("llama3.1-8b-instruct")
model_store_path = "/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/"

# Define the GPU configuration
gpu = modal.gpu.A100(count=1)  # type: ignore


@stub.function(
    image=image,
    gpu=gpu,  # type: ignore
    concurrency_limit=10,
    allow_concurrent_inputs=1,
    volumes={model_store_path: volume},
    container_idle_timeout=300,
    keep_warm=2,
)
@modal.asgi_app(custom_domains=["monitor-backend.transluce.org"])  # type: ignore
def app():
    from monitor.server import asgi_app

    return asgi_app
