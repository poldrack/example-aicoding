"""Download the most recently committed Python files from GitHub.

Uses the GitHub Search API to find recently updated Python repositories,
then downloads Python files from their file trees via raw content URLs.
"""

import os

import requests


GITHUB_API = "https://api.github.com"
RAW_CONTENT_BASE = "https://raw.githubusercontent.com"


def download_recent_python_files(max_files=100):
    """Download the most recently committed Python files from GitHub.

    Finds recently updated Python repositories via the Search API,
    retrieves their file trees, and downloads Python file content
    from raw.githubusercontent.com.

    Args:
        max_files: Maximum number of files to download (default 100).

    Returns:
        List of dicts, each with keys: name, url, content.
    """
    results = []
    headers = {"Accept": "application/vnd.github.v3+json"}

    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"token {token}"

    seen = set()

    # Find recently updated Python repositories
    search_url = f"{GITHUB_API}/search/repositories"
    params = {"q": "language:python", "sort": "pushed", "per_page": 30}
    resp = requests.get(search_url, headers=headers, params=params)
    if resp.status_code != 200:
        return results

    repos = resp.json().get("items", [])

    for repo_info in repos:
        if len(results) >= max_files:
            break

        repo_name = repo_info.get("full_name", "")
        branch = repo_info.get("default_branch", "main")
        if not repo_name:
            continue

        # Get the repo's file tree
        tree_url = f"{GITHUB_API}/repos/{repo_name}/git/trees/{branch}"
        tree_resp = requests.get(
            tree_url, headers=headers, params={"recursive": "1"}
        )
        if tree_resp.status_code != 200:
            continue

        tree = tree_resp.json().get("tree", [])
        py_files = [
            f for f in tree
            if f.get("path", "").endswith(".py") and f.get("type") == "blob"
        ]

        for file_info in py_files:
            if len(results) >= max_files:
                break

            path = file_info["path"]
            key = f"{repo_name}/{path}"
            if key in seen:
                continue
            seen.add(key)

            # Download raw content (doesn't count against API rate limit)
            raw_url = f"{RAW_CONTENT_BASE}/{repo_name}/{branch}/{path}"
            try:
                content_resp = requests.get(raw_url, timeout=10)
                content = content_resp.text if content_resp.status_code == 200 else ""
            except requests.RequestException:
                content = ""

            basename = path.split("/")[-1]
            blob_url = f"https://github.com/{repo_name}/blob/{branch}/{path}"
            results.append({
                "name": basename,
                "url": blob_url,
                "content": content,
            })

    return results


if __name__ == "__main__":
    print("Downloading 100 most recently committed Python files from GitHub...")
    files = download_recent_python_files(max_files=100)
    print(f"Downloaded {len(files)} files:\n")
    for f in files:
        preview = f["content"][:100].replace("\n", " ") if f["content"] else "(empty)"
        print(f"  {f['name']}: {preview}...")
