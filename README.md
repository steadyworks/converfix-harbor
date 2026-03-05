# Converfix Harbor Task Set

A Harbor-compatible benchmark for debugging buggy ML implementations.

**22 tasks** across **6 problems**.

## Prerequisites

- [Harbor](https://github.com/harbor-framework/harbor) installed
- Docker
- Git LFS (`git lfs install`)
- Python 3.11+ with: `pip install pandas scikit-learn kaggle pyyaml`
- Kaggle API credentials (if using Kaggle problems): `~/.kaggle/kaggle.json`

## Quick Start

```bash
# 1. Clone and fetch LFS data
git clone <repo-url> converfix-harbor
cd converfix-harbor
git lfs install && git lfs pull

# 2. Prepare data
pip install pandas scikit-learn kaggle pyyaml
./prepare.sh --all

# 3. Build/pull base Docker image
./build.sh

# 4. Run with Harbor
harbor run -p tasks/ -a claude-code -m anthropic/claude-opus-4-6
```

## Directory Structure

```
├── prepare.sh          # Data preparation script
├── build.sh            # Docker image pull/build
├── problems/           # Problem definitions (data + grading)
├── environment/        # Base Docker image
├── data/               # Prepared data (created by prepare.sh, gitignored)
└── tasks/              # Harbor task set (22 tasks)
```

## Problems

| Problem | Type | Tasks | GPU |
|---------|------|-------|-----|
| Creditcard Fraud Detection 2023 | community | 1 | Yes |
| Simple Classification Problem | community | 2 | No |
| Social Network User Engagement Prediction | community | 2 | No |
| Tomato diseases | community | 9 | Yes |
| Tweet Sentiment Extraction | kaggle | 2 | No |
| Unsolicited Messages | community | 6 | No |


## Data Preparation

Run `./prepare.sh --all` to prepare all datasets, or `./prepare.sh <problem_id>` for a specific problem.

- **Community problems**: Data is included via Git LFS (`dataset.zip` files)
- **Kaggle problems**: Requires Kaggle API credentials; downloads automatically

## Base Docker Image

Run `./build.sh` to get the base image. It first tries to pull `steadyworks/converfix-base:latest` from DockerHub; if that fails, it builds locally (~30 min).

## Running Tasks

```bash
# Run all tasks
harbor run -p tasks/ -a claude-code -m anthropic/claude-opus-4-6

# Run a specific task
harbor run -t tasks/converfix-scp-hardcoded-v0 -a claude-code -m anthropic/claude-opus-4-6

# Run with oracle (reference solution)
harbor run -p tasks/ -a oracle
```

## Reward Format

Each task produces a `reward.json` with:

| Key | Type | Description |
|-----|------|-------------|
| `reward` | float [0,1] | Primary reward (normalized performance) |
| `metric` | float | Raw grading metric score |
| `valid_submission` | int 0/1 | Whether a valid submission was produced |
| `beats_buggy` | int 0/1/-1 | Whether score exceeds buggy baseline |
| `matches_golden` | int 0/1 | Whether score matches golden reference |
