import pickle
import sys
import pprint

if len(sys.argv) < 2:
    print("Usage: python view_pickle.py <path_to_pickle_file>")
    sys.exit(1)

file_path = sys.argv[1]

# Load the pickle file
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Print the structure and content
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print("\nKeys:")
        for key in data.keys():
            print(f"  - {key}: {type(data[key])}")
        
        # Ask which key to explore
        print("\nData preview:")
        pprint.pprint(data, depth=2)
    else:
        pprint.pprint(data)
except Exception as e:
    print(f"Error: {e}")