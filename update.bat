:: Batch Script for Updating Repository and Re-executing main.py with Venv Activation

@echo off
SETLOCAL EnableDelayedExpansion

call venv\Scripts\activate.bat

:: Updating the repository
cd ./sharegpt4v-synthesize-caption
git stash
git reset --hard
git pull

:: Copy updated main.py and re-execute
cd ..
copy sharegpt4v-synthesize-caption\main.py main.py
python main.py --port 9051 --launch-option gradio

:: Deactivate venv
call venv\Scripts\deactivate.bat
