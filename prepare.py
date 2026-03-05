#!/usr/bin/env python3
"""Standalone data preparation script for the Converfix Harbor task set.

Usage:
    python prepare.py --all              # Prepare all problems
    python prepare.py <problem_id>       # Prepare a specific problem

Requirements: pip install pandas scikit-learn kaggle pyyaml
"""
import argparse
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import webbrowser
import zipfile
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
PROBLEMS_DIR = SCRIPT_DIR / "problems"


# ---------------------------------------------------------------------------
# Git LFS
# ---------------------------------------------------------------------------

def is_git_lfs_available() -> bool:
    try:
        result = subprocess.run(
            ["git", "lfs", "version"], capture_output=True, text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def is_lfs_pointer(file_path: Path) -> bool:
    if not file_path.exists():
        return False
    try:
        with open(file_path, "rb") as f:
            header = f.read(100)
        return header.startswith(b"version https://git-lfs.github.com/spec/")
    except Exception:
        return False


def fetch_lfs_file(file_path: Path, repo_root: Path) -> bool:
    try:
        rel_path = file_path.relative_to(repo_root)
    except ValueError:
        print(f"WARNING: {file_path} is not under repo root {repo_root}")
        return False

    print(f"  Fetching {rel_path} from Git LFS...")
    result = subprocess.run(
        ["git", "lfs", "fetch", "--include", str(rel_path)],
        cwd=repo_root, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: git lfs fetch failed: {result.stderr}", file=sys.stderr)
        return False

    result = subprocess.run(
        ["git", "lfs", "checkout", str(rel_path)],
        cwd=repo_root, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: git lfs checkout failed: {result.stderr}", file=sys.stderr)
        return False

    print(f"  Successfully fetched {rel_path}")
    return True


def ensure_lfs_fetched(dataset_zip: Path) -> None:
    if not is_lfs_pointer(dataset_zip):
        return

    print(f"  Detected LFS pointer: {dataset_zip.name}")

    if not is_git_lfs_available():
        raise RuntimeError(
            f"{dataset_zip.name} is a Git LFS pointer but git-lfs is not installed.\n"
            "Please install Git LFS:\n"
            "  Ubuntu/Debian: sudo apt-get install git-lfs && git lfs install\n"
            "  macOS:         brew install git-lfs && git lfs install\n"
            "Then run this script again."
        )

    if not fetch_lfs_file(dataset_zip, SCRIPT_DIR):
        raise RuntimeError(
            f"Failed to fetch {dataset_zip.name} from Git LFS.\n"
            "Make sure you have network access and try again."
        )


# ---------------------------------------------------------------------------
# Kaggle
# ---------------------------------------------------------------------------

def download_kaggle_dataset(competition_id: str, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading from Kaggle: {competition_id}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise RuntimeError(
            "Kaggle API not installed. Run: pip install kaggle\n"
            "Then configure credentials: https://github.com/Kaggle/kaggle-api#api-credentials"
        )

    api = KaggleApi()
    api.authenticate()

    try:
        api.competition_download_files(
            competition=competition_id, path=download_dir, quiet=False, force=False,
        )
    except Exception as e:
        if "You must accept this competition" in str(e):
            print("  You must accept the competition rules before downloading.")
            _prompt_user_to_accept_rules(competition_id)
            return download_kaggle_dataset(competition_id, download_dir)
        raise

    zip_files = list(download_dir.glob("*.zip"))
    if len(zip_files) != 1:
        raise RuntimeError(
            f"Expected exactly 1 zip file after Kaggle download, found {len(zip_files)} in {download_dir}"
        )
    return zip_files[0]


def _prompt_user_to_accept_rules(competition_id: str) -> None:
    url = f"https://www.kaggle.com/c/{competition_id}/rules"
    response = input(f"  Open competition page in browser? ({url}) [y/n]: ")
    if response.strip().lower() != "y":
        raise RuntimeError("You must accept the competition rules before downloading.")
    webbrowser.open(url)
    input("  Press Enter after you have accepted the rules...")


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------

def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_zip_checksum(zipfile_path: Path, checksums_file: Path) -> None:
    if not checksums_file.exists():
        print("  No checksums.yaml found, skipping zip verification")
        return
    expected = yaml.safe_load(checksums_file.read_text())
    if "zip" not in expected:
        return
    actual = md5(zipfile_path)
    if actual != expected["zip"]:
        raise ValueError(
            f"ZIP CHECKSUM MISMATCH for {zipfile_path.name}: "
            f"expected={expected['zip']} actual={actual}"
        )
    print(f"  Zip checksum verified")


def verify_prepared_checksums(data_dir: Path, checksums_file: Path) -> None:
    if not checksums_file.exists():
        print("  No checksums.yaml found, skipping prepared data verification")
        return
    expected = yaml.safe_load(checksums_file.read_text())
    for subdir in ["public", "private"]:
        if subdir not in expected:
            continue
        for fname, expected_hash in expected[subdir].items():
            fpath = data_dir / "prepared" / subdir / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Expected prepared file not found: {fpath}")
            actual = md5(fpath)
            if actual != expected_hash:
                raise ValueError(
                    f"CHECKSUM MISMATCH: {fpath.name} ({subdir}) "
                    f"expected={expected_hash} actual={actual}"
                )
    print(f"  Prepared data checksums verified")


# ---------------------------------------------------------------------------
# Core preparation
# ---------------------------------------------------------------------------

def is_dir_nonempty(d: Path) -> bool:
    return d.is_dir() and any(d.iterdir())


def prepare_problem(problem_id: str) -> None:
    problem_dir = PROBLEMS_DIR / problem_id
    config_path = problem_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml found for problem: {problem_id}")

    config = yaml.safe_load(config_path.read_text())
    problem_type = config.get("problem_type", "community")
    checksums_file = problem_dir / "checksums.yaml"

    data_dir = DATA_DIR / problem_id
    raw_dir = data_dir / "raw"
    public_dir = data_dir / "prepared" / "public"
    private_dir = data_dir / "prepared" / "private"

    print(f"=== Preparing {problem_id} ({problem_type}) ===")
    raw_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire dataset
    if problem_type == "community":
        dataset_zip = problem_dir / "dataset.zip"
        if not dataset_zip.exists():
            raise FileNotFoundError(
                f"dataset.zip not found for community problem '{problem_id}' at {dataset_zip}"
            )
        ensure_lfs_fetched(dataset_zip)
        dst_zip = data_dir / "dataset.zip"
        if not dst_zip.exists() or is_lfs_pointer(dst_zip):
            shutil.copy2(dataset_zip, dst_zip)
            print(f"  Copied dataset.zip to {dst_zip}")
        zipfile_path = dst_zip
    else:
        # Check if already downloaded
        existing_zips = list(data_dir.glob("*.zip"))
        if existing_zips:
            zipfile_path = existing_zips[0]
            print(f"  Using existing zip: {zipfile_path.name}")
        else:
            zipfile_path = download_kaggle_dataset(problem_id, data_dir)

    # 2. Verify zip checksum
    verify_zip_checksum(zipfile_path, checksums_file)

    # 3. Extract
    if not is_dir_nonempty(raw_dir):
        print(f"  Extracting {zipfile_path.name}...")
        with zipfile.ZipFile(zipfile_path, "r") as zf:
            zf.extractall(raw_dir)
        print(f"  Extracted to {raw_dir}")

    # 4. Run prepare.py if not already prepared
    if not is_dir_nonempty(public_dir) or not is_dir_nonempty(private_dir):
        prepare_py = problem_dir / "prepare.py"
        if not prepare_py.exists():
            raise FileNotFoundError(f"prepare.py not found for {problem_id}")

        print(f"  Running prepare.py...")
        # Import and call prepare() from the problem's prepare.py
        spec = importlib.util.spec_from_file_location(f"prepare_{problem_id}", prepare_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.prepare(raw=raw_dir, public=public_dir, private=private_dir)
        print(f"  prepare() completed")
    else:
        print(f"  Prepared data already exists, skipping prepare step")

    # 5. Copy description.md to public
    desc_src = problem_dir / "description.md"
    if desc_src.exists():
        shutil.copy2(desc_src, public_dir / "description.md")

    # 6. Verify prepared data checksums
    verify_prepared_checksums(data_dir, checksums_file)

    # 7. Verify non-empty
    assert is_dir_nonempty(public_dir), f"Public dir is empty after preparation: {public_dir}"
    assert is_dir_nonempty(private_dir), f"Private dir is empty after preparation: {private_dir}"

    print(f"=== {problem_id} ready ===\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for the Converfix Harbor task set",
    )
    parser.add_argument(
        "problem_id",
        nargs="?",
        help="Problem ID to prepare (or use --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prepare all problems",
    )
    args = parser.parse_args()

    if args.all:
        problem_ids = sorted([
            d.name for d in PROBLEMS_DIR.iterdir()
            if d.is_dir() and (d / "config.yaml").exists()
        ])
        if not problem_ids:
            print("No problems found in problems/ directory.")
            sys.exit(1)
        print(f"Preparing {len(problem_ids)} problems: {', '.join(problem_ids)}\n")
        for pid in problem_ids:
            try:
                prepare_problem(pid)
            except Exception as e:
                print(f"ERROR preparing {pid}: {e}", file=sys.stderr)
                sys.exit(1)
        print(f"All {len(problem_ids)} problems prepared successfully.")
    elif args.problem_id:
        prepare_problem(args.problem_id)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
