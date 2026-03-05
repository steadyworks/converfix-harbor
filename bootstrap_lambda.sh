#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fresh Ubuntu Lambda/cloud instance for converfix-harbor.
# Installs: Python 3.11, Docker, NVIDIA Container Toolkit, Harbor, and prepares data.

# ── Remove conflicting Docker packages ───────────────────────────────────────
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
    sudo apt-get remove -y "$pkg" 2>/dev/null || true
done

# ── System packages ──────────────────────────────────────────────────────────
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg2 python3.11 python3.11-venv python3.11-dev git-lfs

git lfs install

# ── Git config ───────────────────────────────────────────────────────────────
git config --global user.email "tedwxli@gmail.com"
git config --global user.name "Ted Li"
git config --global pull.rebase true

# ── Docker Engine ────────────────────────────────────────────────────────────
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker "$USER"

# ── NVIDIA Container Toolkit ─────────────────────────────────────────────────
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# ── Python venv + Harbor + data prep deps ────────────────────────────────────
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas scikit-learn kaggle pyyaml
pip install harbor-bench

# ── Prepare data & build base image ──────────────────────────────────────────
newgrp docker <<'NEWGRP'
docker --version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "=== Bootstrap complete ==="
echo "Run: source .venv/bin/activate"
echo "Then: python prepare.py --all"
echo "Then: ./build.sh"
echo "Then: harbor run -c configs/job-all.yaml -a oracle"
NEWGRP
