# download_github_code

Downloads the 100 most recently committed Python files from GitHub.

## Approach

Uses the public GitHub Events API to find recent `PushEvent` entries, extracts Python file paths from commit details, and downloads content via the GitHub Contents API (base64 decoded).

## Usage

```bash
python -m aicoding.download_github_code.solution
```
