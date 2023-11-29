::activate venv
call venv\Scripts\activate.bat

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip3 install bitsandbytes-windows
::exit
call venv\Scripts\deactivate.bat

pause