"""Fetch CrossCodeEval dataset and clone required repositories.

Downloads the cceval dataset, filters to Python, and clones the
required repositories at their specified commits from GitHub.
"""

import argparse
import json
import os
import shutil
import subprocess
import tarfile

_BASE_DIR = os.path.dirname(__file__)

CCEVAL_REPO = "https://github.com/amazon-science/cceval.git"


def download_data(directory=None, lang="python"):
    """Clone official cceval repo and extract dataset for the specified language."""
    if directory is None:
        directory = _BASE_DIR
    datasets_dir = os.path.join(directory, "datasets")

    if os.path.exists(os.path.join(datasets_dir, lang)):
        print(f"Dataset already exists at {datasets_dir}/{lang}, skipping download.")
        return

    print("Cloning official cceval repository...")
    os.makedirs(datasets_dir, exist_ok=True)

    clone_dir = os.path.join(datasets_dir, "_cceval_repo")
    if not os.path.exists(clone_dir):
        subprocess.run(
            ["git", "clone", "--depth", "1", CCEVAL_REPO, clone_dir],
            check=True,
        )

    tar_path = os.path.join(clone_dir, "data", "crosscodeeval_data.tar.xz")
    if not os.path.exists(tar_path):
        raise FileNotFoundError(
            f"Dataset archive not found at {tar_path}. "
            "The file may require Git LFS. Try: cd {clone_dir} && git lfs pull"
        )

    print("Extracting dataset...")
    with tarfile.open(tar_path, "r:xz") as tar:
        members = [
            m for m in tar.getmembers()
            if lang in m.name and not m.name.startswith(".")
        ]
        tar.extractall(path=datasets_dir, members=members)

    shutil.rmtree(clone_dir)
    print(f"Downloaded cceval dataset to {datasets_dir}")


def _clone_and_checkout(repo_url, commit, dest_dir):
    """Clone a repository and checkout a specific commit.

    Returns True on success, False on failure. Cleans up on failure.
    """
    if os.path.exists(dest_dir):
        print(f"  Already exists: {dest_dir}")
        return True

    try:
        subprocess.run(
            ["git", "clone", "--no-checkout", repo_url, dest_dir],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", commit],
            cwd=dest_dir, check=True, capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        return False


def _parse_repo_field(repo_field):
    """Parse cceval repository field '{owner}-{repo}-{commit_hash}'.

    The commit hash is always the last 7 characters after the final '-'.
    The owner/repo split is at the first '-' boundary that yields a valid
    GitHub owner (no '-' ambiguity is resolved by trying owner as the first segment).

    Returns:
        (owner, repo_name, commit) tuple, e.g. ('turboderp', 'exllama', 'a544085')
    """
    # commit hash is the last segment after final '-'
    last_dash = repo_field.rfind("-")
    commit = repo_field[last_dash + 1:]
    owner_repo = repo_field[:last_dash]

    # owner is the first segment before the first '-'
    first_dash = owner_repo.find("-")
    owner = owner_repo[:first_dash]
    repo_name = owner_repo[first_dash + 1:]

    return owner, repo_name, commit


def resolve_and_clone(directory=None, lang="python"):
    """Clone repositories referenced in the cceval dataset."""
    if directory is None:
        directory = _BASE_DIR

    datasets_dir = os.path.join(directory, "datasets")
    repo_dir = os.path.join(directory, "repositories")
    os.makedirs(repo_dir, exist_ok=True)

    dataset_path = os.path.join(datasets_dir, lang, "line_completion.jsonl")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run download first.")

    # Collect unique repositories: keyed by the raw field value
    repos = {}
    with open(dataset_path, "r") as f:
        for line in f:
            item = json.loads(line)
            repo_field = item["metadata"]["repository"]
            if repo_field not in repos:
                repos[repo_field] = _parse_repo_field(repo_field)

    print(f"Found {len(repos)} repositories to clone.")
    failed_repos = set()
    for repo_field, (owner, repo_name, commit) in repos.items():
        dest = os.path.join(repo_dir, repo_field)
        repo_url = f"https://github.com/{owner}/{repo_name}.git"
        print(f"Cloning {owner}/{repo_name} @ {commit}...")
        if not _clone_and_checkout(repo_url, commit, dest):
            print(f"  Failed: {owner}/{repo_name} (repo missing or commit invalid)")
            failed_repos.add(repo_field)

    if failed_repos:
        print(f"Failed to clone {len(failed_repos)} repositories: {failed_repos}")

    return failed_repos


def clone_and_curate(directory=None, lang="python"):
    """Clone repos and curate dataset by filtering to files that exist in cloned repos."""
    if directory is None:
        directory = _BASE_DIR

    failed_repos = resolve_and_clone(directory, lang)

    datasets_dir = os.path.join(directory, "datasets")
    raw_path = os.path.join(datasets_dir, lang, "line_completion.jsonl")
    curated_path = os.path.join(datasets_dir, lang, "line_completion_curated.jsonl")

    if os.path.exists(curated_path):
        print(f"Curated dataset already exists: {curated_path}")
        return

    repo_dir = os.path.join(directory, "repositories")
    curated = []
    skipped_repo = 0
    skipped_file = 0

    with open(raw_path, "r") as f:
        for line in f:
            item = json.loads(line)
            meta = item["metadata"]
            repo = meta["repository"]
            if repo in failed_repos:
                skipped_repo += 1
                continue
            fpath = meta["file"]
            full_path = os.path.join(repo_dir, repo, fpath)
            if os.path.exists(full_path):
                curated.append(item)
            else:
                skipped_file += 1

    with open(curated_path, "w") as f:
        for item in curated:
            f.write(json.dumps(item) + "\n")

    print(f"Curated: {len(curated)} kept, "
          f"{skipped_repo} skipped (repo unavailable), "
          f"{skipped_file} skipped (file missing)")


def main():
    parser = argparse.ArgumentParser(description="Fetch CrossCodeEval dataset and repositories.")
    parser.add_argument("--directory", type=str, default=None,
                        help="Base directory for datasets and repositories.")
    parser.add_argument("--lang", type=str, default="python",
                        help="Language to keep from the dataset.")
    parser.add_argument("--download_only", action="store_true",
                        help="Only download the dataset, skip repo cloning.")
    args = parser.parse_args()

    download_data(directory=args.directory, lang=args.lang)
    if not args.download_only:
        clone_and_curate(directory=args.directory, lang=args.lang)


if __name__ == "__main__":
    main()
