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
    parser = argparse.ArgumentParser(description='Patch a file.')
    parser.add_argument('--patch-file', type=str, help='path to file')
    parser.add_argument('--line-num', type=int, help='line number')
    parser.add_argument('--content', type=str, help='content')
    args = parser.parse_args()
    patch_file(args.file_path, args.line_num, args.content)

if __name__ == '__main__':
    main()        
