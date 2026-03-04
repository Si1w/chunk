import io
import os
import argparse
import requests
import zipfile

_BASE_DIR = os.path.dirname(__file__)


def download_data(directory=None):
    if directory is None:
        directory = _BASE_DIR
    os.makedirs(directory, exist_ok=True)

    datasets_dir = os.path.join(directory, "datasets")
    repo_dir = os.path.join(directory, "repositories")

    print("Start downloading the necessary `datasets` and `repositories` files.")

    if not os.path.exists(datasets_dir):
        print("Start downloading the `datasets`.")
        url = "https://github.com/microsoft/CodeT/raw/main/RepoCoder/datasets/datasets.zip"
        r = requests.get(url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(datasets_dir)
        print("Finished downloading the `datasets` files.")

    if not os.path.exists(repo_dir):
        print("Start downloading the `repositories` (line_and_api).")
        url = "https://github.com/microsoft/CodeT/raw/main/RepoCoder/repositories/line_and_api_level.zip"
        r = requests.get(url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(repo_dir)
        print("Finished downloading the `repositories` files.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory", type=str, default=None,
        help="The directory to save the downloaded datasets and repositories.",
    )
    args = parser.parse_args()
    download_data(directory=args.directory)


if __name__ == "__main__":
    main()
