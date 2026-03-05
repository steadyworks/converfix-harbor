#!/bin/bash
cp /solution/main.py /home/code/main.py
cd /home/code
/opt/conda/bin/conda run -n agent python main.py
