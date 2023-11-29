#!/bin/bash
# Bash Script for Updating Repository and Re-executing main.py with Venv Activation

# Activate venv
source venv/bin/activate

cd LLaVA
# Updating the repository
cd ./sharegpt4v-synthesize-caption
git stash
git reset --hard
git pull

# Copy updated main.py and re-execute
cd ..
cp sharegpt4v-synthesize-caption/main.py main.py
export CUDA_VISIBLE_DEVICES=0
python main.py --port 9051 --launch-option gradio --cuda-devices 0

# Deactivate venv
cd ..
deactivate

read -p "Press enter to continue"