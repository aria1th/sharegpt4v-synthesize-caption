::activate venv
call venv\Scripts\activate.bat

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade

::exit
call venv\Scripts\deactivate.bat

pause