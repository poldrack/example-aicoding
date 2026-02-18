"""Evaluation harness — runs each solution script and generates an HTML report.

Executes every solution.py __main__ block, captures stdout output and any
generated image files, and produces an HTML report.
"""

import csv
import html
import os
import subprocess
import sys
import glob
import base64
import time
from datetime import datetime

# All modules in the project, ordered by problem number
MODULES = [
    ("linear_regression", 1),
    ("pubmed", 2),
    ("LDA", 3),
    ("ddm", 4),
    ("lmm", 5),
    ("pubmed_emails", 6),
    ("download_github_code", 7),
    ("code_complexity", 8),
    ("bad_cv", 9),
    ("mk_classification_data", 10),
    ("mk_animation", 11),
    ("transformer", 12),
    ("logisticmap", 13),
    ("GES", 14),
    ("DAG", 15),
    ("CLT", 16),
    ("ddm_collapsing_plot", 17),
    ("hurdle", 18),
    ("corridor", 19),
    ("imgsmooth", 20),
    ("PC", 21),
    ("textanalysis", 22),
    ("twostep", 23),
    ("balanced_cv", 24),
    ("overfitting", 25),
    ("factoranalysis", 26),
    ("PAweather", 27),
    ("ascii_art", 28),
    ("pdf_generation", 29),
    ("PCA_poweriteration", 30),
    ("ttest_wrapper", 31),
    ("randomization", 32),
]

# No modules are skipped — all external-API modules should be exercised
# against the real APIs.
SKIP_MODULES = set()


def load_prompts(project_root):
    """Load problem prompts from coding_problems.tsv.

    Returns a dict mapping problem number (int) to prompt string.
    """
    tsv_path = os.path.join(project_root, "data", "coding_problems.tsv")
    prompts = {}
    if not os.path.exists(tsv_path):
        return prompts
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            filename = row.get("File", "")
            # Extract number from filename like "2_pubmed.py"
            parts = filename.split("_", 1)
            if parts[0].isdigit():
                prompts[int(parts[0])] = row.get("Prompt", "").strip().strip('"')
    return prompts


LONG_TIMEOUT_MODULES = {"PAweather", "download_github_code"}


def run_solution(module_name, project_root):
    """Run a solution module's __main__ block and capture output.

    Parameters
    ----------
    module_name : str
        Module name under src/aicoding/.
    project_root : str
        Path to the project root.

    Returns
    -------
    dict
        'stdout': str, 'stderr': str, 'returncode': int, 'images': list of str.
    """
    solution_path = os.path.join(project_root, "src", "aicoding", module_name, "solution.py")

    if not os.path.exists(solution_path):
        return {
            "stdout": "",
            "stderr": f"solution.py not found at {solution_path}",
            "returncode": -1,
            "images": [],
        }

    image_exts = ("*.png", "*.jpg", "*.jpeg", "*.svg", "*.gif")

    start_time = time.time()

    timeout = 300 if module_name in LONG_TIMEOUT_MODULES else 120

    try:
        result = subprocess.run(
            [sys.executable, solution_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_root,
        )
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = f"TIMEOUT: Script exceeded {timeout} seconds"
        returncode = -2
    except Exception as e:
        stdout = ""
        stderr = str(e)
        returncode = -3

    # Detect images created or modified during the run (check root and outputs/)
    new_images = []
    search_dirs = [project_root, os.path.join(project_root, "outputs")]
    for search_dir in search_dirs:
        for ext in image_exts:
            for img_path in glob.glob(os.path.join(search_dir, ext)):
                if os.path.getmtime(img_path) >= start_time:
                    new_images.append(img_path)
    new_images.sort()

    return {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "images": new_images,
    }


def image_to_base64(image_path):
    """Convert an image file to a base64 data URI."""
    ext = os.path.splitext(image_path)[1].lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".gif": "image/gif",
    }.get(ext, "image/png")

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


def generate_html_report(results, output_path="evaluation_report.html", prompts=None):
    """Generate an HTML report from evaluation results.

    Parameters
    ----------
    results : list of dict
        Each dict has 'module', 'number', 'stdout', 'stderr', 'returncode', 'images'.
    output_path : str
        Path to write the HTML file.
    prompts : dict, optional
        Mapping of problem number to prompt string.
    """
    if prompts is None:
        prompts = {}
    n_total = len(results)
    n_success = sum(1 for r in results if r["returncode"] == 0)
    n_skipped = sum(1 for r in results if r["returncode"] == -99)
    n_failed = n_total - n_success - n_skipped

    report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
  .summary {{ background: #f0f0f0; padding: 15px; border-radius: 8px; margin: 20px 0; }}
  .module {{ border: 1px solid #ddd; border-radius: 8px; margin: 20px 0; padding: 20px; }}
  .module h2 {{ margin-top: 0; }}
  .success {{ border-left: 4px solid #28a745; }}
  .failure {{ border-left: 4px solid #dc3545; }}
  .skipped {{ border-left: 4px solid #ffc107; }}
  pre {{ background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; }}
  .stderr {{ background: #3a1a1a; color: #ff6b6b; }}
  .prompt {{ background: #eef6ff; border-left: 3px solid #4a90d9; padding: 10px 15px; margin: 10px 0; font-style: italic; white-space: pre-wrap; }}
  img {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; color: white; }}
  .badge-success {{ background: #28a745; }}
  .badge-failure {{ background: #dc3545; }}
  .badge-skipped {{ background: #ffc107; color: #333; }}
</style>
</head>
<body>
<h1>Evaluation Report</h1>
<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<div class="summary">
  <strong>Summary:</strong> {n_success} passed, {n_failed} failed, {n_skipped} skipped / {n_total} total
</div>
"""

    for r in results:
        if r["returncode"] == 0:
            css_class = "success"
            badge = '<span class="badge badge-success">PASS</span>'
        elif r["returncode"] == -99:
            css_class = "skipped"
            badge = '<span class="badge badge-skipped">SKIP</span>'
        else:
            css_class = "failure"
            badge = '<span class="badge badge-failure">FAIL</span>'

        report += f'<div class="module {css_class}">\n'
        report += f'<h2>#{r["number"]} — {r["module"]} {badge}</h2>\n'

        prompt_text = prompts.get(r["number"], "")
        if prompt_text:
            report += f'<h3>Prompt</h3>\n<div class="prompt">{html.escape(prompt_text)}</div>\n'

        if r["returncode"] == -99:
            report += "<p><em>Skipped (requires external API or data)</em></p>\n"
        else:
            if r["stdout"]:
                report += f'<h3>Output</h3>\n<pre>{html.escape(r["stdout"])}</pre>\n'
            if r["stderr"] and r["returncode"] != 0:
                report += f'<h3>Errors</h3>\n<pre class="stderr">{html.escape(r["stderr"])}</pre>\n'
            for img_path in r.get("images", []):
                if os.path.exists(img_path):
                    data_uri = image_to_base64(img_path)
                    report += f'<h3>Image: {os.path.basename(img_path)}</h3>\n'
                    report += f'<img src="{data_uri}" alt="{os.path.basename(img_path)}">\n'

        report += "</div>\n"

    report += "</body>\n</html>"

    with open(output_path, "w") as f:
        f.write(report)

    return output_path


def main():
    """Run all solution scripts and generate the evaluation report."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("Evaluation Harness")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Running {len(MODULES)} solution modules...\n")

    results = []
    for module_name, number in MODULES:
        if module_name in SKIP_MODULES:
            print(f"  [{number:2d}] {module_name:25s} ... SKIPPED (external dependency)")
            results.append({
                "module": module_name,
                "number": number,
                "stdout": "",
                "stderr": "Skipped: requires external API or data",
                "returncode": -99,
                "images": [],
            })
            continue

        print(f"  [{number:2d}] {module_name:25s} ... ", end="", flush=True)
        r = run_solution(module_name, project_root)
        r["module"] = module_name
        r["number"] = number
        results.append(r)

        if r["returncode"] == 0:
            print("OK")
        else:
            print(f"FAILED (exit code {r['returncode']})")

    prompts = load_prompts(project_root)
    output = generate_html_report(results, os.path.join(project_root, "evaluation_report.html"), prompts=prompts)
    print(f"\nReport saved to: {output}")

    n_success = sum(1 for r in results if r["returncode"] == 0)
    n_skipped = sum(1 for r in results if r["returncode"] == -99)
    n_failed = len(results) - n_success - n_skipped
    print(f"Results: {n_success} passed, {n_failed} failed, {n_skipped} skipped")


if __name__ == "__main__":
    main()
