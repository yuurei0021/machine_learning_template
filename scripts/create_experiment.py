"""
Create a new experiment from template.

Usage:
    uv run python scripts/create_experiment.py <experiment_name> [--template classification|regression]

Examples:
    uv run python scripts/create_experiment.py baseline_lgbm
    uv run python scripts/create_experiment.py baseline_lgbm --template regression
    uv run python scripts/create_experiment.py eda --date 20251201
"""

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"
TEMPLATES_DIR = BASE_DIR / "templates"

TEMPLATE_MAP = {
    "classification": "classification_lgbm.py",
    "regression": "regression_lgbm.py",
}


def get_next_sequence(date_str: str) -> int:
    """Get the next sequence number for the given date."""
    existing = list(EXPERIMENTS_DIR.glob(f"{date_str}_*"))
    if not existing:
        return 1

    max_seq = 0
    for d in existing:
        match = re.match(rf"{date_str}_(\d+)_", d.name)
        if match:
            max_seq = max(max_seq, int(match.group(1)))
    return max_seq + 1


def create_experiment(name: str, template: str, date_str: str | None = None) -> Path:
    """Create a new experiment directory from template."""
    date_str = date_str or datetime.now().strftime("%Y%m%d")
    seq = get_next_sequence(date_str)

    exp_name = f"{date_str}_{seq:02d}_{name}"
    exp_dir = EXPERIMENTS_DIR / exp_name

    if exp_dir.exists():
        raise FileExistsError(f"Experiment directory already exists: {exp_dir}")

    # Create directory structure
    exp_dir.mkdir(parents=True)
    (exp_dir / "predictions").mkdir()
    (exp_dir / "model").mkdir()

    # Copy template
    template_file = TEMPLATES_DIR / TEMPLATE_MAP[template]
    if template_file.exists():
        shutil.copy2(template_file, exp_dir / "main.py")
    else:
        print(f"Warning: Template not found: {template_file}")
        (exp_dir / "main.py").write_text(
            f'"""\\n{exp_name}\\n"""\\n',
            encoding="utf-8",
        )

    # Create README
    readme = f"""# {exp_name}

## 目的
TODO

## アプローチ
TODO

## 結果
TODO

## 次のステップ
TODO
"""
    (exp_dir / "README.md").write_text(readme, encoding="utf-8")

    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="Create a new experiment from template")
    parser.add_argument("name", help="Experiment name (e.g., baseline_lgbm, eda)")
    parser.add_argument(
        "--template", "-t",
        default="classification",
        choices=list(TEMPLATE_MAP.keys()),
        help="Template type (default: classification)",
    )
    parser.add_argument(
        "--date", "-d",
        default=None,
        help="Date override (YYYYMMDD format, default: today)",
    )
    args = parser.parse_args()

    exp_dir = create_experiment(args.name, args.template, args.date)

    print(f"Created: {exp_dir}")
    print(f"  Template: {args.template}")
    print(f"  Run: uv run python experiments/{exp_dir.name}/main.py")


if __name__ == "__main__":
    main()
