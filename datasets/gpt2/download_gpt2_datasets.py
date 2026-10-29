import argparse
import json
import os
import sys

import numpy as np
import requests

ALLOWED_SIZES = ("124M", "355M", "774M", "1558M")
BASE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models"
FILENAMES = [
    "checkpoint", "encoder.json", "hparams.json",
    "model.ckpt.data-00000-of-00001", "model.ckpt.index",
    "model.ckpt.meta", "vocab.bpe",
]
CHUNK_SIZE = 1024 * 1024


def human_readable(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


def remote_size(url):
    # A one-byte range request is enough to read the total size without downloading the file.
    response = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
    response.raise_for_status()
    content_range = response.headers.get("Content-Range", "")
    response.close()
    if "/" not in content_range:
        return None
    return int(content_range.rsplit("/", 1)[1])


def download_file(file_url, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    total_size = remote_size(file_url)

    if os.path.exists(file_path):
        on_disk = os.path.getsize(file_path)
        if total_size is not None and on_disk == total_size:
            print(f"  [skip] {os.path.basename(file_path)} already complete ({human_readable(on_disk)})")
            return file_path
        if total_size is not None and on_disk > total_size:
            os.remove(file_path)
            on_disk = 0
    else:
        on_disk = 0

    headers = {"Range": f"bytes={on_disk}-"} if on_disk else {}
    mode = "ab" if on_disk else "wb"
    if on_disk:
        print(f"  [resume] {os.path.basename(file_path)} from {human_readable(on_disk)}")

    with requests.get(file_url, headers=headers, stream=True, timeout=120) as response:
        if on_disk and response.status_code == 200:
            # Server ignored the range request, start over.
            on_disk = 0
            mode = "wb"
        response.raise_for_status()
        expected = total_size if total_size is not None else on_disk
        written = on_disk
        last_reported = -1
        with open(file_path, mode) as destination:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                destination.write(chunk)
                written += len(chunk)
                percent = int(written * 100 / expected) if expected else 0
                if percent != last_reported and percent % 5 == 0:
                    last_reported = percent
                    print(f"  {os.path.basename(file_path)}: {percent}% "
                          f"({human_readable(written)} / {human_readable(expected)})", flush=True)
        if expected and written != expected:
            raise IOError(f"{file_path}: expected {expected} bytes, got {written}")

    if total_size is not None and os.path.getsize(file_path) != total_size:
        raise IOError(f"{file_path}: size mismatch after download")
    return file_path


def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings):
    import tensorflow as tf

    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        variable_name_parts = name.split("/")[1:]
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})
        target_dict[variable_name_parts[-1]] = variable_array
    return params


def download_and_load_gpt2(model_size, models_dir, load_params=True):
    # Validate model size
    if model_size not in ALLOWED_SIZES:
        raise ValueError(f"Model size not in {ALLOWED_SIZES}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in FILENAMES:
        file_url = f"{BASE_URL}/{model_size}/{filename}"
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # Load settings and params
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = None
    if load_params:
        tf_ckpt_path = os.path.join(model_dir, "model.ckpt")
        params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def main():
    parser = argparse.ArgumentParser(description="Download the original GPT-2 weights published by OpenAI.")
    parser.add_argument("--model_size", default="124M", choices=ALLOWED_SIZES,
                        help="GPT-2 variant to download (default: 124M)")
    parser.add_argument("--models_dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory that receives a <model_size> sub folder (default: this script's folder)")
    parser.add_argument("--load_params", action="store_true",
                        help="Load the checkpoint into numpy arrays (requires tensorflow)")
    args = parser.parse_args()

    print(f"Downloading GPT-2 {args.model_size} into {args.models_dir}")
    settings, params = download_and_load_gpt2(args.model_size, args.models_dir,
                                              load_params=args.load_params)
    print(f"Settings: {settings}")
    if params is not None:
        print(f"Loaded params for {len(params['blocks'])} transformer blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
