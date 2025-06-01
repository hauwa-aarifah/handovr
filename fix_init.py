# fix_init.py
import os

# List of directories that need __init__.py files
directories = [
    'prediction',
    'components',
    'config',
    'data_generation',
    'demo_output',
    'notebooks',
    'optimization',
    'models',
    'results',
    'tests',
    'utils',
    'visualization'
]

for directory in directories:
    if os.path.exists(directory):
        init_file = os.path.join(directory, '__init__.py')
        # Create or overwrite with empty file
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write('# -*- coding: utf-8 -*-\n')
        print(f"Created/Fixed: {init_file}")