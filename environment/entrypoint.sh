#!/bin/bash
set -x
{
  LOGS_DIR=/home/logs
  mkdir -p $LOGS_DIR /home/code /home/submission

  # Copy scaffold to agent working directory
  if [ -d "/home/task_instance/scaffold" ]; then
    echo "Copying scaffold to /home/code/"
    cp -r /home/task_instance/scaffold/* /home/code/
  fi

  # Install scaffold requirements if present
  if [ -f "/home/code/requirements.txt" ]; then
    echo "Installing scaffold requirements"
    /opt/conda/bin/conda run -n agent pip install -r /home/code/requirements.txt
  fi

  # Set permissions for nonroot user (exclude read-only mounts)
  find /home -path /home/data -prune -o -exec chmod a+rw {} \;

  # Execute the command passed by harbor ("sleep infinity")
  exec "$@"
} 2>&1 | tee $LOGS_DIR/entrypoint.log
