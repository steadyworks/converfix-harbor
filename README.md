# Converfix Harbor Preview Set

A Harbor-compatible benchmark for debugging buggy ML implementations.

**12 tasks** across **5 problems**.

## Prerequisites (~5 mins)

1. Ensure:
   - Kaggle API credentials stored under: `~/.kaggle/kaggle.json`
2. Ensure following are installed: Run `./bootstrap_prerequisite.sh` if not sure (Tested working on Ubuntu 22.04)
   - NVIDIA Container Toolkit
   - Docker 
   - Git LFS installed
   - Python 3.12

## Quick Start (~10 mins)

```bash
# 1. Clone and fetch LFS data
cd converfix-harbor

# 2. Build/pull base Docker image
sudo usermod -aG docker "$USER"                                
newgrp docker
./build_or_pull_docker_image.sh # (~ 5 mins)

# 3. Install Python packages
python3.12 -m venv .venv # or python3 -m venv .venv  (Needs >= Python 3.12)
source .venv/bin/activate
pip install -r requirements.txt
git lfs install && git lfs pull

# 4. Prepare data
# Before you run, make sure you have filled out `~/.kaggle/kaggle.json` (as specified in Prerequisite)
# And go to https://www.kaggle.com/competitions/tweet-sentiment-extraction to accept the Terms
python prepare_data.py --all

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

# 5. Run with Harbor
# 5.a. Run oracle
harbor run -c configs/job-oracle.yaml
# Expected to see perfect or near perfect score:
#⠸ 11/11 Mean: 1.000 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸━━━ 0:35:44 -:--:--

# 5.b. Run Claude Code agent
# Option 1
export ANTHROPIC_API_KEY="..."    

# Option 2
export CLAUDE_CODE_OAUTH_TOKEN="..."

harbor run -c configs/job-claude-code.yaml
# Expected to see non-perfect score:
#⠸ 11/11 Mean: 0.867 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸━━━ 0:35:44 -:--:--
```

## Directory Structure

```
├── prepare.py          # Data preparation script
├── build.sh            # Docker image pull/build
├── problems/           # Problem definitions (data + grading)
├── environment/        # Base Docker image
├── data/               # Prepared data (created by prepare.py, gitignored)
└── tasks/              # Harbor task set (12 tasks)
```

## Problems

| Problem | Type | Tasks | GPU |
|---------|------|-------|-----|
| Creditcard Fraud Detection 2023 | community | 1 | Yes |
| Simple Classification Problem | community | 2 | No |
| Tomato diseases | community | 1 | Yes |
| Tweet Sentiment Extraction | kaggle | 2 | No |
| Unsolicited Messages | community | 6 | No |


## Data Preparation

Run `python prepare.py --all` to prepare all datasets, or `python prepare.py <problem_id>` for a specific problem.

- **Community problems**: Data is included via Git LFS (`dataset.zip` files)
- **Kaggle problems**: Requires Kaggle API credentials; downloads automatically

## Base Docker Image

Run `./build.sh` to get the base image. It first tries to pull `steadyworks/converfix-base:latest` from DockerHub; if that fails, it builds locally (~30 min).


## Reward Format

Each task produces a `reward.json` with:

| Key | Type | Description |
|-----|------|-------------|
| `reward` | float [0,1] | Primary reward (normalized performance) |
| `metric` | float | Raw grading metric score |
| `valid_submission` | int 0/1 | Whether a valid submission was produced |
| `beats_buggy` | int 0/1/-1 | Whether score exceeds buggy baseline |
| `matches_golden` | int 0/1 | Whether score matches golden reference |


## How to Obtain kaggle.json

You can generate and download this file directly from the Kaggle website: 
- Sign in to your Kaggle account.
- Navigate to your Account settings page (profile picture -> "My Account").
- Scroll down to the API section.
- Click the "Create New API Token" button. This action will automatically generate and download the kaggle.json file to your computer.

Store `~/.kaggle/kaggle.json`: 

```
{"username":"<yourkaggleusername>","key":"<kagglekey>"}
```