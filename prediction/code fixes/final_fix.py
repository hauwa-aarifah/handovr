# File: final_fix.py
import os
import re

def fix_hospital_selection_file():
    """Apply all necessary fixes to hospital_selection.py"""
    filepath = os.path.join("prediction", "hospital_selection.py")
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False
        
    # Read the file
    with open(filepath, "r") as f:
        content = f.read()
    
    # 1. Fix generate_selection_explanation method
    pattern = r"def generate_selection_explanation\(.*?return .*?explanation"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        method_text = match.group(0)
        # Replace direct access with .get()
        fixed_method = method_text.replace(
            "if \"Type 1\" in top_hospital['Hospital_Type']:", 
            "if \"Type 1\" in top_hospital.get('Hospital_Type', 'Unknown'):"
        )
        content = content.replace(method_text, fixed_method)
        print("Fixed generate_selection_explanation method")
    
    # 2. Fix analyze_selection_reasons method
    pattern = r"def analyze_selection_reasons\(.*?return analysis"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        method_text = match.group(0)
        # Replace direct access with .get()
        fixed_method = method_text.replace(
            "'Hospital_Type': hospital['Hospital_Type']", 
            "'Hospital_Type': hospital.get('Hospital_Type', 'Unknown')"
        )
        content = content.replace(method_text, fixed_method)
        print("Fixed analyze_selection_reasons method")
    
    # 3. Fix get_hospital_selection_details method
    pattern = r"def get_hospital_selection_details\(.*?return \{"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        method_text = match.group(0)
        # Replace direct access with .get()
        fixed_method = method_text.replace(
            "'Hospital_Type': top_hospital['Hospital_Type'],", 
            "'Hospital_Type': top_hospital.get('Hospital_Type', 'Unknown'),"
        )
        content = content.replace(method_text, fixed_method)
        print("Fixed get_hospital_selection_details method")
    
    # 4. Add Hospital_Type to ranked_hospitals in select_optimal_hospital
    pattern = r"def select_optimal_hospital\(.*?ranked_hospitals = hospital_data\.sort_values\('Final_Score', ascending=False\)"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        method_text = match.group(0)
        replacement = method_text + """
        
        # Ensure Hospital_Type is included
        if 'Hospital_Type' not in ranked_hospitals.columns:
            logger.warning("Hospital_Type column missing from ranked_hospitals. Adding from hospital_locations.")
            # Create a mapping from Hospital_ID to Hospital_Type
            hospital_type_map = {}
            for _, row in self.hospital_locations.iterrows():
                if 'Hospital_ID' in row and 'Hospital_Type' in row:
                    hospital_type_map[row['Hospital_ID']] = row['Hospital_Type']
            
            # Add Hospital_Type column
            ranked_hospitals['Hospital_Type'] = ranked_hospitals['Hospital_ID'].apply(
                lambda x: hospital_type_map.get(x, 'Type 1')
            )
        """
        content = content.replace(method_text, replacement)
        print("Updated select_optimal_hospital to add Hospital_Type column")
    
    # Write the fixed content back to the file
    with open(filepath, "w") as f:
        f.write(content)
    
    print(f"All fixes applied to {filepath}")
    return True

if __name__ == "__main__":
    if fix_hospital_selection_file():
        print("Fix completed successfully!")
    else:
        print("Failed to apply fixes.")