SETLOCAL EnableDelayedExpansion
:: Setup venv
if exist venv (
    echo "venv already exists"
) else (
    echo "Creating venv"
    python -m venv venv
)
call venv\Scripts\activate.bat

:: check venv was successfully activated
if "%VIRTUAL_ENV%" == "" (
    echo "Failed to activate venv"
    exit /b 1
)

:: Environment Setup and Dependency Installation
:: skip if already installed
if exist LLaVA (
    echo "LLaVA already installed"
) else (
    git clone https://github.com/haotian-liu/LLaVA
    cd LLaVA
    pip install --upgrade pip
    pip install -e .

    cd ..
)
:: update git repository if already cloned
if exist sharegpt4v-synthesize-caption (
    echo "Updating the repository"
    cd sharegpt4v-synthesize-caption
    git stash
    git reset --hard
    git pull
    cd ..
) else (
    git clone https://github.com/aria1th/sharegpt4v-synthesize-caption
)

copy sharegpt4v-synthesize-caption\main.py main.py
copy sharegpt4v-synthesize-caption\patch.py patch.py
::patch_file('./llava/model/multimodal_encoder/builder.py', 7, '    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:')
::patch_file('./llava/model/builder.py', 130, '    if "llava" in model_name.lower() or "sharegpt4v" in model_name.lower():\n')

::call patch.py
python patch.py

:: Execution of main.py, test with --test-api
python main.py --port 9051 --launch-option gradio --test-api

:: Deactivate venv
call venv\Scripts\deactivate.bat