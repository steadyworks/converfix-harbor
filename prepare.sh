#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
PROBLEMS_DIR="${SCRIPT_DIR}/problems"

prepare_problem() {
    local problem_id="$1"
    local problem_dir="${PROBLEMS_DIR}/${problem_id}"
    local config="${problem_dir}/config.yaml"
    local data_dir="${DATA_DIR}/${problem_id}"
    local raw_dir="${data_dir}/raw"
    local public_dir="${data_dir}/prepared/public"
    local private_dir="${data_dir}/prepared/private"

    echo "=== Preparing ${problem_id} ==="
    mkdir -p "$raw_dir" "$public_dir" "$private_dir"

    # 1. Acquire dataset
    local problem_type=$(python3 -c "import yaml; print(yaml.safe_load(open('${config}'))['problem_type'])")
    if [ "$problem_type" = "community" ]; then
        # Ensure LFS file is fetched
        local zip="${problem_dir}/dataset.zip"
        if head -c 50 "$zip" | grep -q "git-lfs"; then
            echo "Fetching ${zip} from Git LFS..."
            git lfs pull --include "problems/${problem_id}/dataset.zip"
        fi
        cp "$zip" "${data_dir}/dataset.zip"
    else
        # Kaggle download
        echo "Downloading from Kaggle: ${problem_id}"
        kaggle competitions download -c "$problem_id" -p "$data_dir"
    fi

    # 2. Find the zip file (community -> dataset.zip, kaggle -> {competition_id}.zip)
    local zipfile
    zipfile=$(find "$data_dir" -maxdepth 1 -name "*.zip" | head -1)
    if [ -z "$zipfile" ]; then
        echo "ERROR: No zip file found in ${data_dir}" >&2
        exit 1
    fi

    # 3. Extract
    if [ -z "$(ls -A "$raw_dir")" ]; then
        unzip -o "$zipfile" -d "$raw_dir"
    fi

    # 4. Verify zip checksum
    python3 -c "
import yaml, hashlib, sys
from pathlib import Path
def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()
checksums_file = Path('${problem_dir}/checksums.yaml')
if checksums_file.exists():
    expected = yaml.safe_load(checksums_file.read_text())
    if 'zip' in expected:
        actual = md5(Path('${zipfile}'))
        if actual != expected['zip']:
            print(f'ZIP CHECKSUM MISMATCH: expected={expected[\"zip\"]} actual={actual}', file=sys.stderr)
            sys.exit(1)
        print('Zip checksum verified for ${problem_id}')
"

    # 5. Run prepare.py
    if [ -z "$(ls -A "$public_dir")" ] || [ -z "$(ls -A "$private_dir")" ]; then
        python3 "${problem_dir}/prepare.py" \
            --raw "$raw_dir" --public "$public_dir" --private "$private_dir"
    fi

    # 6. Copy description.md to public
    cp "${problem_dir}/description.md" "${public_dir}/description.md"

    # 7. Verify prepared data checksums
    python3 -c "
import yaml, hashlib, sys
from pathlib import Path
def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()
checksums_file = Path('${problem_dir}/checksums.yaml')
if checksums_file.exists():
    expected = yaml.safe_load(checksums_file.read_text())
    for subdir in ['public', 'private']:
        if subdir in expected:
            for fname, expected_hash in expected[subdir].items():
                fpath = Path('${data_dir}/prepared') / subdir / fname
                actual = md5(fpath)
                if actual != expected_hash:
                    print(f'CHECKSUM MISMATCH: {fpath} expected={expected_hash} actual={actual}', file=sys.stderr)
                    sys.exit(1)
    print('Prepared data checksums verified for ${problem_id}')
else:
    print('No checksums.yaml found, skipping verification')
"
    echo "=== ${problem_id} ready ==="
}

# Main
if [ "${1:-}" = "--all" ]; then
    for dir in "${PROBLEMS_DIR}"/*/; do
        problem_id=$(basename "$dir")
        prepare_problem "$problem_id"
    done
elif [ -n "${1:-}" ]; then
    prepare_problem "$1"
else
    echo "Usage: ./prepare.sh --all | ./prepare.sh <problem_id>"
    exit 1
fi
