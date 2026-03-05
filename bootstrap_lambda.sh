#!/usr/bin/env bash
set -euo pipefail

# Install Docker Engine on a fresh Ubuntu system using the official apt repository.
# Reference: https://docs.docker.com/engine/install/ubuntu/

# Remove any conflicting packages
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
    sudo apt-get remove -y "$pkg" 2>/dev/null || true
done


echo "installing 3.11"
sudo apt install -y python3.11 python3.11-venv python3.11-dev

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .

git config --global user.email "tedwxli@gmail.com"
git config --global user.name "Ted Li"
git config pull.rebase true


# Install prerequisites
sudo apt-get update
sudo apt-get install -y ca-certificates curl

sudo apt-get install git-lfs && git lfs install

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the Docker apt repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Enable and start Docker
sudo systemctl enable docker
sudo systemctl start docker

# Add current user to the docker group so sudo isn't needed
sudo usermod -aG docker "$USER"

echo ""
echo "Docker installed successfully."
echo "Log out and back in (or run 'newgrp docker') for group changes to take effect."

newgrp docker <<'EOF'
docker --version
docker build -t converfix-env -f environment/Dockerfile .
EOF
