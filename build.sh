#!/usr/bin/env bash
IMAGE="steadyworks/converfix-base:latest"
echo "Pulling ${IMAGE} from DockerHub..."
if docker pull "${IMAGE}" 2>/dev/null; then
    docker tag "${IMAGE}" converfix-base
    echo "Successfully pulled converfix-base"
else
    echo "Pull failed, building locally (this will take ~30 minutes)..."
    docker build -t converfix-base -f environment/Dockerfile .
fi
