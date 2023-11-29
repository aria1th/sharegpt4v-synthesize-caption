#!/bin/bash
# Setup venv
if [ -d "venv" ]; then
    echo "venv already exists"
else
    python3 -m venv venv
fi
source venv/bin/activate

# check venv was successfully activated
if [ -z "$VIRTUAL_ENV" ]; then
    exit 1
fi

# Environment Setup and Dependency Installation
# skip if already installed
if [ -d "LLaVA" ]; then
    cd LLaVA
else
    git clone https://github.com/haotian-liu/LLaVA
    cd LLaVA
    pip install --upgrade pip
    pip install -e .
    # re-install torch with cuda 118
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
    # now it should work
    pip install gradio
fi
cd ..

# update git repository if already cloned
if [ -d "sharegpt4v-synthesize-caption" ]; then
    cd sharegpt4v-synthesize-caption
    git stash
    git reset --hard
    git pull
    cd ..
else
    git clone https://github.com/aria1th/sharegpt4v-synthesize-caption
fi

cp sharegpt4v-synthesize-caption/main.py main.py
cp sharegpt4v-synthesize-caption/patch.py patch.py

#call patch.py
python3 patch.py

# Execution of main.py, test with --test-api
python3 main.py --port 9051 --launch-option gradio --test-api --cuda-devices 0

# Deactivate venv
deactivate