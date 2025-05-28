import pandas as pd
import json
import os
import argparse

def convert_excel_to_json(excel_file):
    """
    Convert MIHCSME Excel template to JSON format
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file
        
    Returns:
    --------
    dict
        JSON-compatible Python dictionary
    """
    # Read the Excel file
    xls = pd.ExcelFile(excel_file)
    
    # Initialize the result dictionary
    result = {}
    
    # Process regular sheets (key-value pairs)
    regular_sheets = ["InvestigationInformation", "StudyInformation", "AssayInformation"]
    
    for sheet_name in regular_sheets:
        if sheet_name in xls.sheet_names:
            # Read the sheet
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Skip rows that start with '#'
            df = df[~df.iloc[:, 0].astype(str).str.startswith('#')]
            
            # Convert to nested structure
            sheet_data = {}
            current_group = None
            
            for _, row in df.iterrows():
                # Get the first column which contains the group
                group = row.iloc[0]
                
                # Skip header rows or empty rows
                if pd.isna(group) or group == "Annotation_groups" or str(group).startswith('#'):
                    continue
                
                # Get key and value (columns 1 and 2)
                if len(row) > 2:  # Make sure there are enough columns
                    key = row.iloc[1]
                    value = row.iloc[2]
                else:
                    continue
                
                # Skip rows with no key
                if pd.isna(key):
                    continue
                
                # Initialize the group if it doesn't exist
                if group not in sheet_data:
                    sheet_data[group] = {}
                
                # Add the key-value pair to the group
                sheet_data[group][key] = value
            
            # Add the sheet data to the result
            result[sheet_name] = sheet_data
    
    # Process AssayConditions sheet (tabular data)
    if "AssayConditions" in xls.sheet_names:
        try:
            # Read the sheet
            df = pd.read_excel(excel_file, sheet_name="AssayConditions")
            
            # Skip rows that start with '#'
            df = df[~df.iloc[:, 0].astype(str).str.startswith('#')]
            
            # Find the header row
            header_row_idx = None
            for idx, row in df.iterrows():
                if isinstance(row.iloc[0], str) and row.iloc[0] == "Plate":
                    header_row_idx = idx
                    break
            
            if header_row_idx is not None:
                # Get the headers
                headers = df.iloc[header_row_idx].tolist()
                
                # Get the data rows
                data_rows = df.iloc[header_row_idx + 1:].copy()
                
                # Set the column names
                data_rows.columns = headers
                
                # Drop columns with NaN headers
                data_rows = data_rows.loc[:, ~pd.isna(headers)]
                
                # Convert to list of dictionaries
                conditions = data_rows.to_dict(orient="records")
                
                # Add to result
                result["AssayConditions"] = conditions
            else:
                print(f"Warning: Could not find header row with 'Plate' in AssayConditions sheet")
                result["AssayConditions"] = []
        except Exception as e:
            print(f"Error processing AssayConditions sheet: {str(e)}")
            result["AssayConditions"] = []
    
    # Process reference sheets
    reference_sheets = [s for s in xls.sheet_names if s.startswith('_')]
    
    for sheet_name in reference_sheets:
        try:
            # Read the sheet
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Skip rows that start with '#'
            df = df[~df.iloc[:, 0].astype(str).str.startswith('#')]
            
            # Skip empty rows
            df = df.dropna(how='all')
            
            # Check if dataframe is empty
            if df.empty:
                result[sheet_name] = {}
                continue
                
            # Find the first non-comment row with data
            valid_rows = []
            for idx, row in df.iterrows():
                if not all(pd.isna(val) for val in row):
                    valid_rows.append(idx)
            
            if not valid_rows:
                result[sheet_name] = {}
                continue
                
            header_row_idx = valid_rows[0]
            
            # Get data
            headers = df.iloc[header_row_idx].tolist()
            
            # Check if we have at least two columns
            if len(headers) < 2:
                result[sheet_name] = {}
                continue
                
            data_rows = df.iloc[header_row_idx + 1:].copy()
            
            # Set column names
            data_rows.columns = headers
            
            # Convert to dictionary
            ref_data = {}
            for _, row in data_rows.iterrows():
                # Use first column as key, second as value
                if len(row) >= 2:
                    key = row.iloc[0]
                    value = row.iloc[1]
                    if not pd.isna(key):
                        ref_data[key] = value
            
            # Add to result
            result[sheet_name] = ref_data
        except Exception as e:
            print(f"Error processing sheet {sheet_name}: {str(e)}")
            result[sheet_name] = {}
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Convert MIHCSME Excel template to JSON')
    parser.add_argument('excel_file', help='Path to the Excel file')
    parser.add_argument('--output', '-o', help='Output JSON file path (default: same as input with .json extension)')
    
    args = parser.parse_args()
    
    # Default output filename
    if args.output is None:
        base_name = os.path.splitext(args.excel_file)[0]
        args.output = f"{base_name}.json"
    
    # Convert Excel to JSON
    result = convert_excel_to_json(args.excel_file)
    
    # Save as JSON file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Conversion completed. JSON saved to {args.output}")

if __name__ == "__main__":
    main()