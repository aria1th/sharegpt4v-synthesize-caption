"""
Executable patch script.
"""

import argparse
import os

def patch_file(file_path:str, line_num:int, content:str):
    """
    Patches file in-place.
    """
    lines = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines[line_num] = content
    with open(file_path, 'w') as f:
        f.writelines(lines)

def main():
    """
    Main entry point.
    """
    patch_file('./llava/model/multimodal_encoder/builder.py', 7, '    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:')
    patch_file('./llava/model/builder.py', 130, '    if "llava" in model_name.lower() or "sharegpt4v" in model_name.lower():\n')

if __name__ == '__main__':
    main()        
