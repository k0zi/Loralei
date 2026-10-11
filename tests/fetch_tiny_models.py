"""Download the tiny random-weight models `test_families.py` runs against.

A few MB each, structurally the real architectures. This is the one thing in the worker's
test suite that touches the network, which is why it is a script and not a fixture.

    .venv/bin/python tests/fetch_tiny_models.py
"""

from pathlib import Path

from huggingface_hub import snapshot_download

REPOS = {
    "qwen": "optimum-intel-internal-testing/tiny-random-qwen-image",
    "krea": "CedricPerauer/tiny-krea2-modular-pipe",
    "ltx": "optimum-intel-internal-testing/tiny-random-ltx2.3",
}

if __name__ == "__main__":
    target = Path(__file__).parent / "tiny-models"
    for name, repo in REPOS.items():
        print(f"{repo} -> {target / name}")
        snapshot_download(repo, local_dir=target / name)
