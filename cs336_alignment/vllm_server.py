# cs336_alignment/start_vllm_server.py

import signal
import time

from cs336_alignment.vllm_utils import VLLMServer
import os
import subprocess

MODEL_PATH = (
    "/root/.cache/huggingface/hub/"
    "models--allenai--OLMo-2-0425-1B/"
    "snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf"
)


server = None


def shutdown(signum, frame):
    print("Stopping vLLM...")

    if server is not None:
        process = server.process

        if process is not None:
            os.killpg(
                os.getpgid(process.pid),
                signal.SIGTERM,
            )

            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(
                    os.getpgid(process.pid),
                    signal.SIGKILL,
                )

    print("vLLM stopped")
    exit(0)


def main():
    global server

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGHUP, shutdown)


    server = VLLMServer(
        model_id=MODEL_PATH,
        gpu=0,
    )

    print("Starting vLLM server...")
    server.start()

    print("vLLM server is ready.")
    print("Press Ctrl+C to stop.")

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()