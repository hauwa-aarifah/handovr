# File: fix_selection_details.py
import os

def fix_selection_methods():
    """Fix the bug in analyze_selection_reasons method"""
    filepath = os.path.join("prediction", "hospital_selection.py")
    
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}")
        return False
    
    # Read the file
    with open(filepath, "r") as f:
        content = f.readlines()
    
    # Find the analyze_selection_reasons method and modify it
    method_found = False
    in_method = False
    modified = False
    
    for i, line in enumerate(content):
        if "def analyze_selection_reasons" in line:
            method_found = True
            in_method = True
            
        if in_method and "'Hospital_Type': hospital['Hospital_Type']" in line:
            # Replace with a version that checks if the column exists
            content[i] = "                'Hospital_Type': hospital.get('Hospital_Type', 'Unknown')\n"
            modified = True
            
        if in_method and line.strip() == "return analysis":
            in_method = False
    
    if not method_found or not modified:
        print("Error: Could not find or modify analyze_selection_reasons method")
        return False
    
    # Write the fixed content back to the file
    with open(filepath, "w") as f:
        f.writelines(content)
    
    print(f"Fixed bug in analyze_selection_reasons method in {filepath}")
    
    # Now fix the get_hospital_selection_details method
    method_found = False
    in_method = False
    modified = False
    
    for i, line in enumerate(content):
        if "def get_hospital_selection_details" in line:
            method_found = True
            in_method = True
            
        if in_method and "'Hospital_Type': top_hospital['Hospital_Type']," in line:
            # Replace with a version that checks if the column exists
            content[i] = "                'Hospital_Type': top_hospital.get('Hospital_Type', 'Unknown'),\n"
            modified = True
            
        if in_method and line.strip() == "return {":
            in_method = False
    
    if not method_found or not modified:
        print("Warning: Could not find or modify get_hospital_selection_details method")
    else:
        # Write the fixed content back to the file
        with open(filepath, "w") as f:
            f.writelines(content)
        print(f"Fixed bug in get_hospital_selection_details method in {filepath}")
    
    return True

if __name__ == "__main__":
    if fix_selection_methods():
        print("Fix applied successfully!")
    else:
        print("Failed to apply fix.")