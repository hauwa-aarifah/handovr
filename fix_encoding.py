# Create a script to fix encoding issues: fix_encoding.py
import os
import codecs

def fix_init_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__init__.py' in filenames:
            filepath = os.path.join(dirpath, '__init__.py')
            print(f"Checking {filepath}")
            
            # Try to read the file
            try:
                with open(filepath, 'rb') as f:
                    content = f.read()
                
                # Check for BOM
                if content.startswith(b'\xff\xfe') or content.startswith(b'\xfe\xff') or content.startswith(b'\xef\xbb\xbf'):
                    print(f"  - Found BOM, removing...")
                    # Remove BOM
                    if content.startswith(b'\xef\xbb\xbf'):
                        content = content[3:]
                    elif content.startswith(b'\xff\xfe'):
                        content = content[2:]
                    elif content.startswith(b'\xfe\xff'):
                        content = content[2:]
                    
                    # Write back without BOM
                    with open(filepath, 'wb') as f:
                        f.write(content)
                    print(f"  - Fixed!")
                else:
                    print(f"  - OK")
                    
            except Exception as e:
                print(f"  - Error: {e}")

# Run from your HANDOVR directory
fix_init_files('.')