"""Code complexity metrics using radon.

Computes cyclomatic complexity, maintainability index, and Halstead metrics
for Python files in a directory.
"""

import os

from radon.complexity import cc_visit, cc_rank
from radon.metrics import mi_visit, h_visit


def compute_cyclomatic_complexity(source_code):
    """Compute cyclomatic complexity for all functions/classes in source code.

    Parameters
    ----------
    source_code : str
        Python source code.

    Returns
    -------
    list of dict
        Each dict has 'name', 'type', 'complexity', and 'rank'.
    """
    if not source_code.strip():
        return []
    results = cc_visit(source_code)
    return [
        {
            "name": block.name,
            "type": block.letter,
            "complexity": block.complexity,
            "rank": cc_rank(block.complexity),
        }
        for block in results
    ]


def compute_maintainability_index(source_code):
    """Compute the maintainability index for source code.

    Parameters
    ----------
    source_code : str
        Python source code.

    Returns
    -------
    float
        Maintainability index (0–100 scale).
    """
    return mi_visit(source_code, multi=True)


def compute_halstead_metrics(source_code):
    """Compute Halstead complexity metrics for source code.

    Parameters
    ----------
    source_code : str
        Python source code.

    Returns
    -------
    dict
        Halstead metrics including volume, difficulty, effort, etc.
    """
    report = h_visit(source_code)
    if not report:
        return {"volume": 0, "difficulty": 0, "effort": 0, "bugs": 0}

    # h_visit returns a list of HalsteadReport objects; take the total
    total = report.total if hasattr(report, "total") else report[0] if report else None
    if total is None:
        return {"volume": 0, "difficulty": 0, "effort": 0, "bugs": 0}

    return {
        "h1": total.h1,
        "h2": total.h2,
        "N1": total.N1,
        "N2": total.N2,
        "vocabulary": total.vocabulary,
        "length": total.length,
        "volume": total.volume,
        "difficulty": total.difficulty,
        "effort": total.effort,
        "bugs": total.bugs,
        "time": total.time,
    }


def analyze_directory(directory_path):
    """Analyze all Python files in a directory for complexity metrics.

    Parameters
    ----------
    directory_path : str
        Path to directory containing Python files.

    Returns
    -------
    dict
        Mapping of filename to metrics dict with 'cyclomatic_complexity',
        'maintainability_index', and 'halstead' keys.
    """
    results = {}

    if not os.path.isdir(directory_path):
        return results

    for filename in sorted(os.listdir(directory_path)):
        if not filename.endswith(".py"):
            continue

        filepath = os.path.join(directory_path, filename)
        with open(filepath, "r") as f:
            source = f.read()

        results[filename] = {
            "cyclomatic_complexity": compute_cyclomatic_complexity(source),
            "maintainability_index": compute_maintainability_index(source),
            "halstead": compute_halstead_metrics(source),
        }

    return results


if __name__ == "__main__":
    import sys

    target_dir = sys.argv[1] if len(sys.argv) > 1 else "data/code_complexity"

    print("Code Complexity Analysis")
    print("=" * 60)

    results = analyze_directory(target_dir)

    if not results:
        print(f"No Python files found in '{target_dir}'")
        sys.exit(0)

    for filename, metrics in results.items():
        print(f"\n--- {filename} ---")
        print(f"  Maintainability Index: {metrics['maintainability_index']:.2f}")

        if metrics["cyclomatic_complexity"]:
            print("  Cyclomatic Complexity:")
            for block in metrics["cyclomatic_complexity"]:
                print(f"    {block['name']}: {block['complexity']} (rank {block['rank']})")

        h = metrics["halstead"]
        if h.get("volume", 0) > 0:
            print(f"  Halstead Volume: {h['volume']:.2f}")
            print(f"  Halstead Difficulty: {h['difficulty']:.2f}")
            print(f"  Halstead Effort: {h['effort']:.2f}")
