#!/bin/bash
# validate_submission.sh — Extensible submission format checker
# Usage: /home/validate_submission.sh /home/submission/submission.csv

SUBMISSION_FILE="${1:-/home/submission/submission.csv}"

if [ ! -f "$SUBMISSION_FILE" ]; then
    echo "ERROR: Submission file not found: ${SUBMISSION_FILE}"
    exit 1
fi

# Check it's valid CSV with at least a header + 1 row
LINE_COUNT=$(wc -l < "$SUBMISSION_FILE")
if [ "$LINE_COUNT" -lt 2 ]; then
    echo "ERROR: Submission has fewer than 2 lines (need header + data)"
    exit 1
fi

# Check for 'id' column in header
HEADER=$(head -1 "$SUBMISSION_FILE")
if ! echo "$HEADER" | grep -q "id"; then
    echo "ERROR: Submission missing 'id' column. Header: ${HEADER}"
    exit 1
fi

echo "Submission format OK: ${LINE_COUNT} lines, header: ${HEADER}"
