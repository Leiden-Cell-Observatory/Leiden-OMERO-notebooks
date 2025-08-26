#!/usr/bin/env python3

"""
OMERO MIHCSME Metadata Uploader Library

Provides functions to convert MIHCSME Excel to JSON-like dict,
validate MIHCSME files, and annotate OMERO objects (Screen and contained Wells).
"""

import json
import logging
import sys
import getpass
from pathlib import Path
import pandas as pd
import ezomero
import argparse
import re

# --- Configuration ---
SHEET_INVESTIGATION = "InvestigationInformation"
SHEET_STUDY = "StudyInformation"
SHEET_ASSAY = "AssayInformation"
SHEET_CONDITIONS = "AssayConditions"
# Default base namespace, can be overridden
DEFAULT_NS_BASE = "MIHCSME"

log = logging.getLogger(__name__)

# --- Helper Functions ---
def well_coord_to_name(row, col, rows=8, cols=12):
    """Converts 0-based row/col index to alphanumeric well name (e.g., A01, B12)."""
    if not (0 <= row < rows and 0 <= col < cols):
        log.warning(f"Invalid well coordinate ({row},{col}) for grid size {rows}x{cols}")
        return None
    return f"{chr(ord('A') + row)}{col + 1:02d}"

def normalize_well_name(well_name):
    """
    Normalize well names to handle both padded (A01) and non-padded (A1) formats.
    Returns the zero-padded format (A01) as the canonical form.
    """
    if not well_name:
        return None
    
    well_name = str(well_name).strip().upper()
    if len(well_name) < 2:
        return None
    
    # Extract row letter and column number
    row_letter = well_name[0]
    col_part = well_name[1:]
    
    try:
        col_num = int(col_part)
        # Return zero-padded format
        return f"{row_letter}{col_num:02d}"
    except ValueError:
        return None

def create_well_name_mapping(well_names):
    """
    Create a mapping from both padded and non-padded well names to the canonical padded format.
    This allows lookup of wells using either format.
    """
    mapping = {}
    for well_name in well_names:
        normalized = normalize_well_name(well_name)
        if normalized:
            # Map both the original and normalized forms to the normalized form
            mapping[well_name] = normalized
            mapping[normalized] = normalized
            
            # Also map the non-padded version if the original was padded
            if len(well_name) >= 3 and well_name[1] == '0':
                non_padded = well_name[0] + str(int(well_name[1:]))
                mapping[non_padded] = normalized
            # And map the padded version if the original was non-padded
            elif len(well_name) >= 2 and well_name[1] != '0':
                try:
                    col_num = int(well_name[1:])
                    padded = f"{well_name[0]}{col_num:02d}"
                    mapping[padded] = normalized
                except ValueError:
                    pass
    
    return mapping

# --- Validation Functions (Phase 1) ---
def validate_mihcsme_file_structure(excel_path):
    """
    Validates basic MIHCSME file structure
    
    Args:
        excel_path (str): Path to the Excel file
        
    Returns:
        dict: Validation report with is_valid, errors, warnings, and summary
    """
    validation_report = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {
            'required_sheets_found': 0,
            'optional_sheets_found': 0,
            'total_sheets': 0
        }
    }
    
    filepath = Path(excel_path)
    
    # Check if file exists and is readable
    if not filepath.exists():
        validation_report['is_valid'] = False
        validation_report['errors'].append(f"File does not exist: {excel_path}")
        return validation_report
    
    if filepath.suffix.lower() not in ['.xlsx', '.xls']:
        validation_report['is_valid'] = False
        validation_report['errors'].append(f"File is not an Excel format (.xlsx/.xls): {excel_path}")
        return validation_report
    
    try:
        # Try to read Excel file
        xls = pd.ExcelFile(filepath)
        available_sheets = xls.sheet_names
        validation_report['summary']['total_sheets'] = len(available_sheets)
        
        # Check required sheets
        required_sheets = [SHEET_INVESTIGATION, SHEET_STUDY, SHEET_ASSAY, SHEET_CONDITIONS]
        missing_sheets = [s for s in required_sheets if s not in available_sheets]
        
        if missing_sheets:
            validation_report['is_valid'] = False
            validation_report['errors'].append(f"Missing required sheets: {', '.join(missing_sheets)}")
        
        validation_report['summary']['required_sheets_found'] = len(required_sheets) - len(missing_sheets)
        
        # Check optional reference sheets (these are expected, not warnings)
        reference_sheets = [s for s in available_sheets if s.startswith('_')]
        validation_report['summary']['optional_sheets_found'] = len(reference_sheets)
        
        # Reference sheets are expected and helpful, so we don't warn about them
        # They provide controlled vocabulary references for the metadata
        
        # Validate sheet contents if basic structure is valid
        if validation_report['is_valid']:
            # Validate AssayConditions sheet
            if SHEET_CONDITIONS in available_sheets:
                conditions_validation = _validate_assay_conditions_sheet(xls, SHEET_CONDITIONS)
                validation_report['errors'].extend(conditions_validation['errors'])
                validation_report['warnings'].extend(conditions_validation['warnings'])
                if conditions_validation['errors']:
                    validation_report['is_valid'] = False
            
            # Validate key-value sheets
            for sheet_name in [SHEET_INVESTIGATION, SHEET_STUDY, SHEET_ASSAY]:
                if sheet_name in available_sheets:
                    kv_validation = _validate_key_value_sheet(xls, sheet_name)
                    validation_report['errors'].extend(kv_validation['errors'])
                    validation_report['warnings'].extend(kv_validation['warnings'])
                    if kv_validation['errors']:
                        validation_report['is_valid'] = False
        
        xls.close()
        
    except Exception as e:
        validation_report['is_valid'] = False
        validation_report['errors'].append(f"Error reading Excel file: {str(e)}")
    
    return validation_report

def _validate_assay_conditions_sheet(xls, sheet_name):
    """Validate AssayConditions sheet structure and content"""
    validation = {'errors': [], 'warnings': []}
    
    try:
        # Read the sheet
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Skip rows that start with '#'
        df = df[~df.iloc[:, 0].astype(str).str.startswith('#')]
        
        # The first remaining row is the header
        if df.empty:
            validation['errors'].append(f"No data found in {sheet_name} sheet after removing comments")
            return validation
        
        header_row_idx = df.index[0]
        headers = df.iloc[0].tolist()
        
        # Check required columns
        if 'Plate' not in headers:
            validation['errors'].append(f"Missing required 'Plate' column in {sheet_name}")
        if 'Well' not in headers:
            validation['errors'].append(f"Missing required 'Well' column in {sheet_name}")
        
        if validation['errors']:
            return validation
        
        # Get data rows
        data_rows = df.iloc[1:].copy()
        data_rows.columns = headers
        
        # Validate well format and check for duplicates
        if 'Well' in data_rows.columns:
            # Updated pattern to handle both 96-well (A01-H12) and 384-well (A01-P24) plates
            # Accepts both zero-padded (A01) and non-zero-padded (A1) formats
            well_pattern = re.compile(r'^[A-P]([1-9]|[12][0-9]|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48)$|^[A-P](0[1-9]|[12][0-9]|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48)$')
            # Simpler alternative pattern:
            well_pattern = re.compile(r'^[A-P](0?[1-9]|[12][0-9]|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48)$')
            # Even simpler - just check A-P rows and 1-48 columns with optional zero padding
            well_pattern = re.compile(r'^[A-P]([1-9]|0[1-9]|[1-4][0-9]|[1-4][0-8])$')
            # Most permissive but accurate pattern:
            well_pattern = re.compile(r'^[A-P](0?[1-9]|[1-3][0-9]|4[0-8])$')  # A1-P48 with optional zero padding
            
            invalid_wells = []
            for idx, well in data_rows['Well'].dropna().items():
                if not well_pattern.match(str(well)):
                    invalid_wells.append(str(well))
            
            if invalid_wells:
                validation['warnings'].append(f"Invalid well format found (expected A1-P48 format): {', '.join(invalid_wells[:5])}{'...' if len(invalid_wells) > 5 else ''}")
        
        # Check for duplicate (Plate, Well) combinations
        if 'Plate' in data_rows.columns and 'Well' in data_rows.columns:
            plate_well_combos = data_rows[['Plate', 'Well']].dropna()
            duplicates = plate_well_combos[plate_well_combos.duplicated()]
            if not duplicates.empty:
                validation['errors'].append(f"Found duplicate (Plate, Well) combinations in {sheet_name}")
        
        # Check for empty plate identifiers
        if 'Plate' in data_rows.columns:
            empty_plates = data_rows['Plate'].isna().sum()
            if empty_plates > 0:
                validation['warnings'].append(f"Found {empty_plates} rows with empty plate identifiers")
    
    except Exception as e:
        validation['errors'].append(f"Error validating {sheet_name} sheet: {str(e)}")
    
    return validation

def _validate_key_value_sheet(xls, sheet_name):
    """Validate key-value sheets (Investigation/Study/Assay Information)"""
    validation = {'errors': [], 'warnings': []}
    
    try:
        # Read the sheet
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Check minimum columns
        if df.shape[1] < 3:
            validation['errors'].append(f"Sheet {sheet_name} should have at least 3 columns (Group, Key, Value)")
            return validation
        
        # Skip rows that start with '#'
        df = df[~df.iloc[:, 0].astype(str).str.startswith('#')]
        
        # Check for actual data
        non_empty_rows = 0
        for _, row in df.iterrows():
            if not all(pd.isna(val) for val in row[:3]):
                non_empty_rows += 1
        
        if non_empty_rows == 0:
            validation['warnings'].append(f"Sheet {sheet_name} appears to be empty (no non-comment data found)")
    
    except Exception as e:
        validation['errors'].append(f"Error validating {sheet_name} sheet: {str(e)}")
    
    return validation

# --- Validation Functions (Phase 2) ---
def validate_omero_target(conn, object_type, object_id):
    """
    Validates the target OMERO object exists and is accessible
    
    Args:
        conn: OMERO connection
        object_type (str): "Screen" or "Plate"
        object_id (int): OMERO object ID
        
    Returns:
        dict: Validation report
    """
    validation_report = {
        'target_valid': False,
        'errors': [],
        'warnings': [],
        'object_info': None
    }
    
    if object_type.lower() not in ["screen", "plate"]:
        validation_report['errors'].append(f"Object type '{object_type}' not supported. Use 'Screen' or 'Plate'.")
        return validation_report
    
    try:
        # Try to get the object
        obj = conn.getObject(object_type, object_id)
        if obj is None:
            validation_report['errors'].append(f"{object_type} with ID {object_id} not found or not accessible")
            return validation_report
        
        validation_report['target_valid'] = True
        validation_report['object_info'] = {
            'id': obj.getId(),
            'name': obj.getName(),
            'type': object_type
        }
        
        # Additional checks for Screen
        if object_type.lower() == "screen":
            plates = list(obj.listChildren())
            validation_report['object_info']['plate_count'] = len(plates)
            validation_report['object_info']['plate_names'] = [p.getName() for p in plates]
        
        # Additional checks for Plate
        elif object_type.lower() == "plate":
            wells = list(obj.listChildren())
            validation_report['object_info']['well_count'] = len(wells)
    
    except Exception as e:
        validation_report['errors'].append(f"Error accessing {object_type} {object_id}: {str(e)}")
    
    return validation_report

def validate_screen_metadata_match(conn, screen_id, excel_path):
    """
    Validates Screen structure matches MIHCSME metadata
    
    Args:
        conn: OMERO connection
        screen_id (int): Screen ID
        excel_path (str): Path to MIHCSME Excel file
        
    Returns:
        dict: Validation report
    """
    validation_report = {
        'structure_match': True,
        'errors': [],
        'warnings': [],
        'plate_summary': {
            'omero_plates': 0,
            'metadata_plates': 0,
            'matching_plates': 0
        },
        'plate_details': []
    }
    
    try:
        # Get OMERO screen info
        screen = conn.getObject("Screen", screen_id)
        if not screen:
            validation_report['errors'].append(f"Screen {screen_id} not found")
            validation_report['structure_match'] = False
            return validation_report
        
        omero_plates = list(screen.listChildren())
        omero_plate_names = {p.getName(): p.getId() for p in omero_plates}
        validation_report['plate_summary']['omero_plates'] = len(omero_plates)
        
        # Get metadata plates
        try:
            xls = pd.ExcelFile(excel_path)
            df = pd.read_excel(xls, sheet_name=SHEET_CONDITIONS)
            df = df[~df.iloc[:, 0].astype(str).str.startswith('#')]
            
            # The first remaining row is the header
            if df.empty:
                validation_report['errors'].append("No data found in AssayConditions after removing comments")
                validation_report['structure_match'] = False
                return validation_report
            
            headers = df.iloc[0].tolist()
            data_rows = df.iloc[1:].copy()
            data_rows.columns = headers
            
            if 'Plate' not in data_rows.columns:
                validation_report['errors'].append("Missing 'Plate' column in AssayConditions")
                validation_report['structure_match'] = False
                return validation_report
            
            metadata_plate_names = set(data_rows['Plate'].dropna().astype(str).unique())
            validation_report['plate_summary']['metadata_plates'] = len(metadata_plate_names)
            
            xls.close()
            
        except Exception as e:
            validation_report['errors'].append(f"Error reading metadata plates: {str(e)}")
            validation_report['structure_match'] = False
            return validation_report
        
        # Compare plates
        omero_plate_names_set = set(omero_plate_names.keys())
        matching_plates = omero_plate_names_set.intersection(metadata_plate_names)
        validation_report['plate_summary']['matching_plates'] = len(matching_plates)
        
        # Check for mismatches
        plates_in_omero_not_metadata = omero_plate_names_set - metadata_plate_names
        plates_in_metadata_not_omero = metadata_plate_names - omero_plate_names_set
        
        if plates_in_omero_not_metadata:
            validation_report['warnings'].append(f"Plates in OMERO but not in metadata: {', '.join(sorted(plates_in_omero_not_metadata))}")
        
        if plates_in_metadata_not_omero:
            validation_report['errors'].append(f"Plates in metadata but not in OMERO: {', '.join(sorted(plates_in_metadata_not_omero))}")
            validation_report['structure_match'] = False
        
        # Detailed plate validation
        for plate_name in matching_plates:
            plate_id = omero_plate_names[plate_name]
            plate_validation = validate_plate_metadata_match(conn, plate_id, plate_name, excel_path)
            validation_report['plate_details'].append({
                'plate_name': plate_name,
                'plate_id': plate_id,
                'validation': plate_validation
            })
            
            if plate_validation['errors']:
                validation_report['structure_match'] = False
    
    except Exception as e:
        validation_report['errors'].append(f"Error validating screen structure: {str(e)}")
        validation_report['structure_match'] = False
    
    return validation_report

def validate_plate_metadata_match(conn, plate_id, plate_identifier, excel_path):
    """
    Validates individual plate structure matches metadata
    
    Args:
        conn: OMERO connection
        plate_id (int): Plate ID
        plate_identifier (str): Plate name/identifier
        excel_path (str): Path to MIHCSME Excel file
        
    Returns:
        dict: Validation report
    """
    validation_report = {
        'well_match': True,
        'errors': [],
        'warnings': [],
        'well_summary': {
            'omero_wells': 0,
            'metadata_wells': 0,
            'matching_wells': 0,
            'omero_grid': None,
            'metadata_grid': None
        }
    }
    
    try:
        # Get OMERO plate and wells using direct OMERO API
        plate = conn.getObject("Plate", plate_id)
        if not plate:
            validation_report['errors'].append(f"Could not retrieve Plate object for ID {plate_id}")
            validation_report['well_match'] = False
            return validation_report
        
        # Get all wells from the plate
        omero_wells = list(plate.listChildren())
        if not omero_wells:
            validation_report['warnings'].append(f"No wells found in OMERO plate {plate_id}")
            return validation_report
        
        # Convert wells to normalized names and analyze grid structure
        omero_well_names = set()
        omero_rows = set()
        omero_cols = set()
        
        for well in omero_wells:
            row = well.row
            col = well.column
            omero_rows.add(row)
            omero_cols.add(col)
            # Convert to normalized well name (A01, B12, etc.)
            well_name = f"{chr(ord('A') + row)}{col + 1:02d}"
            omero_well_names.add(well_name)
        
        # Determine OMERO grid structure
        if omero_rows and omero_cols:
            omero_min_row, omero_max_row = min(omero_rows), max(omero_rows)
            omero_min_col, omero_max_col = min(omero_cols), max(omero_cols)
            omero_row_count = omero_max_row - omero_min_row + 1
            omero_col_count = omero_max_col - omero_min_col + 1
            validation_report['well_summary']['omero_grid'] = {
                'rows': omero_row_count,
                'cols': omero_col_count,
                'row_range': f"{chr(ord('A') + omero_min_row)}-{chr(ord('A') + omero_max_row)}",
                'col_range': f"{omero_min_col + 1}-{omero_max_col + 1}",
                'total_wells': len(omero_well_names),
                'sparse': len(omero_wells) < (omero_row_count * omero_col_count)
            }
        
        validation_report['well_summary']['omero_wells'] = len(omero_well_names)
        
        # Get metadata wells for this plate
        try:
            xls = pd.ExcelFile(excel_path)
            df = pd.read_excel(xls, sheet_name=SHEET_CONDITIONS)
            df = df[~df.iloc[:, 0].astype(str).str.startswith('#')]
            
            # The first remaining row is the header
            if df.empty:
                validation_report['errors'].append("No data found in AssayConditions after removing comments")
                validation_report['well_match'] = False
                return validation_report
            
            headers = df.iloc[0].tolist()
            data_rows = df.iloc[1:].copy()
            data_rows.columns = headers
            
            # Filter for this plate
            plate_data = data_rows[data_rows['Plate'].astype(str) == str(plate_identifier)]
            raw_metadata_wells = set(plate_data['Well'].dropna().astype(str))
            
            # Normalize metadata well names
            metadata_well_names = set()
            for well_name in raw_metadata_wells:
                normalized = normalize_well_name(well_name)
                if normalized:
                    metadata_well_names.add(normalized)
            
            # Analyze metadata grid structure
            metadata_rows = set()
            metadata_cols = set()
            for well_name in metadata_well_names:
                if len(well_name) >= 3:  # Should be like A01, B12, etc.
                    row_letter = well_name[0].upper()
                    try:
                        col_num = int(well_name[1:])
                        if 'A' <= row_letter <= 'P':  # Valid row letters
                            row_index = ord(row_letter) - ord('A')
                            metadata_rows.add(row_index)
                            metadata_cols.add(col_num - 1)  # Convert to 0-based
                    except ValueError:
                        continue  # Skip invalid well names
            
            # Determine metadata grid structure
            if metadata_rows and metadata_cols:
                meta_min_row, meta_max_row = min(metadata_rows), max(metadata_rows)
                meta_min_col, meta_max_col = min(metadata_cols), max(metadata_cols)
                meta_row_count = meta_max_row - meta_min_row + 1
                meta_col_count = meta_max_col - meta_min_col + 1
                validation_report['well_summary']['metadata_grid'] = {
                    'rows': meta_row_count,
                    'cols': meta_col_count,
                    'row_range': f"{chr(ord('A') + meta_min_row)}-{chr(ord('A') + meta_max_row)}",
                    'col_range': f"{meta_min_col + 1}-{meta_max_col + 1}",
                    'total_wells': len(metadata_well_names),
                    'sparse': len(metadata_well_names) < (meta_row_count * meta_col_count)
                }
            
            validation_report['well_summary']['metadata_wells'] = len(metadata_well_names)
            
            xls.close()
            
        except Exception as e:
            validation_report['errors'].append(f"Error reading metadata wells: {str(e)}")
            validation_report['well_match'] = False
            return validation_report
        
        # Compare wells by normalized names
        matching_wells = omero_well_names.intersection(metadata_well_names)
        validation_report['well_summary']['matching_wells'] = len(matching_wells)
        
        # Check for mismatches
        wells_in_omero_not_metadata = omero_well_names - metadata_well_names
        wells_in_metadata_not_omero = metadata_well_names - omero_well_names
        
        if wells_in_omero_not_metadata:
            validation_report['warnings'].append(f"Wells in OMERO but not in metadata: {', '.join(sorted(wells_in_omero_not_metadata))}")
        
        if wells_in_metadata_not_omero:
            validation_report['warnings'].append(f"Wells in metadata but not in OMERO: {', '.join(sorted(wells_in_metadata_not_omero))}")
        
        # Check if critical mismatch (different well counts)
        if len(omero_well_names) != len(metadata_well_names):
            validation_report['well_match'] = False
            if abs(len(omero_well_names) - len(metadata_well_names)) > len(omero_well_names) * 0.1:  # More than 10% difference
                validation_report['errors'].append(f"Significant well count mismatch: OMERO has {len(omero_well_names)} wells, metadata has {len(metadata_well_names)} wells")
    
    except Exception as e:
        validation_report['errors'].append(f"Error validating plate wells: {str(e)}")
        validation_report['well_match'] = False
    
    return validation_report

# --- Master Validation Function (Phase 3) ---
def validate_mihcsme_for_omero(excel_path, conn, target_type, target_id):
    """
    Complete validation pipeline
    
    Args:
        excel_path (str): Path to MIHCSME Excel file
        conn: OMERO connection
        target_type (str): "Screen" or "Plate"
        target_id (int): OMERO object ID
        
    Returns:
        dict: Comprehensive validation report
    """
    comprehensive_report = {
        'overall_status': 'valid',  # 'valid', 'warning', 'error'
        'can_proceed': True,
        'file_validation': {},
        'omero_validation': {},
        'recommendations': []
    }
    
    # Phase 1: File Structure Validation
    log.info("=== Phase 1: MIHCSME File Structure Validation ===")
    file_validation = validate_mihcsme_file_structure(excel_path)
    comprehensive_report['file_validation'] = file_validation
    
    if not file_validation['is_valid']:
        comprehensive_report['overall_status'] = 'error'
        comprehensive_report['can_proceed'] = False
        comprehensive_report['recommendations'].append("Fix MIHCSME file structure errors before proceeding")
        return comprehensive_report
    
    if file_validation['warnings']:
        comprehensive_report['overall_status'] = 'warning'
    
    # Phase 2: OMERO Target Validation
    log.info("=== Phase 2: OMERO Target Validation ===")
    target_validation = validate_omero_target(conn, target_type, target_id)
    comprehensive_report['omero_validation']['target_validation'] = target_validation
    
    if not target_validation['target_valid']:
        comprehensive_report['overall_status'] = 'error'
        comprehensive_report['can_proceed'] = False
        comprehensive_report['recommendations'].append("Fix OMERO target accessibility issues")
        return comprehensive_report
    
    # Phase 3: Data Structure Matching
    log.info("=== Phase 3: Data Structure Matching Validation ===")
    if target_type.lower() == "screen":
        structure_validation = validate_screen_metadata_match(conn, target_id, excel_path)
        comprehensive_report['omero_validation']['structure_validation'] = structure_validation
        
        if not structure_validation['structure_match']:
            comprehensive_report['overall_status'] = 'error'
            comprehensive_report['can_proceed'] = False
            comprehensive_report['recommendations'].append("Fix plate/well structure mismatches between OMERO and metadata")
        elif structure_validation['warnings']:
            if comprehensive_report['overall_status'] != 'error':
                comprehensive_report['overall_status'] = 'warning'
    
    elif target_type.lower() == "plate":
        # For single plate validation
        plate_name = target_validation['object_info']['name']
        structure_validation = validate_plate_metadata_match(conn, target_id, plate_name, excel_path)
        comprehensive_report['omero_validation']['structure_validation'] = structure_validation
        
        if not structure_validation['well_match']:
            comprehensive_report['overall_status'] = 'error'
            comprehensive_report['can_proceed'] = False
            comprehensive_report['recommendations'].append("Fix well structure mismatches between OMERO plate and metadata")
        elif structure_validation['warnings']:
            if comprehensive_report['overall_status'] != 'error':
                comprehensive_report['overall_status'] = 'warning'
    
    # Final recommendations
    if comprehensive_report['overall_status'] == 'valid':
        comprehensive_report['recommendations'].append("✅ All validations passed - ready to proceed with metadata upload")
    elif comprehensive_report['overall_status'] == 'warning':
        comprehensive_report['recommendations'].append("⚠️ Warnings found - review issues and confirm before proceeding")
    
    return comprehensive_report

# --- Excel to JSON Conversion Function ---
def convert_excel_to_json(excel_file):
    """
    Convert MIHCSME Excel template to JSON format - Working version from convert_excel_to_json.py
    
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
            
            # The first remaining row is the header
            if not df.empty:
                headers = df.iloc[0].tolist()
                
                # Get the data rows
                data_rows = df.iloc[1:].copy()
                
                # Set the column names
                data_rows.columns = headers
                
                # Drop columns with NaN headers
                data_rows = data_rows.loc[:, ~pd.isna(headers)]
                
                # Convert to list of dictionaries
                conditions = data_rows.to_dict(orient="records")
                
                # Add to result
                result["AssayConditions"] = conditions
            else:
                print(f"Warning: No data found in AssayConditions sheet after removing comments")
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

# --- Core Annotation Logic (Internal) ---
def _apply_metadata_to_object(conn, obj_type, obj_id, metadata_dict, namespace):
    """Internal function to apply key-value pairs as MapAnnotation."""
    if not metadata_dict:
        log.debug(f"No metadata for {obj_type} {obj_id}, ns={namespace}. Skipping.")
        return True
    
    success = True
    
    # Check if metadata_dict contains nested groups (like AssayInformation structure)
    if isinstance(metadata_dict, dict) and any(isinstance(v, dict) for v in metadata_dict.values()):
        # Handle nested structure: create separate annotations for each group
        log.info(f"Processing nested metadata groups for {obj_type} {obj_id} (Base namespace: {namespace})")
        
        for group_name, group_data in metadata_dict.items():
            if isinstance(group_data, dict):
                # Create a specific namespace for this group
                group_namespace = f"{namespace}/{group_name}"
                # Filter out NaN values and convert to strings
                kv_pairs = {str(k): str(v) for k, v in group_data.items() if pd.notna(v)}
                
                if kv_pairs:
                    try:
                        map_ann_id = ezomero.post_map_annotation(conn, obj_type, obj_id, kv_pairs, ns=group_namespace)
                        if map_ann_id:
                            log.debug(f"Successfully applied group '{group_name}' metadata (Annotation ID: {map_ann_id})")
                        else:
                            log.error(f"Failed to apply group '{group_name}' metadata to {obj_type} {obj_id}")
                            success = False
                    except Exception as e:
                        log.error(f"Error applying group '{group_name}' metadata to {obj_type} {obj_id}: {e}")
                        success = False
                else:
                    log.debug(f"Group '{group_name}' metadata empty after filtering NaN values. Skipping.")
            else:
                log.warning(f"Group '{group_name}' data is not a dictionary, skipping")
    else:
        # Handle flat structure: apply as single annotation
        log.info(f"Applying flat metadata to {obj_type} {obj_id} (Namespace: {namespace})")
        kv_pairs = {str(k): str(v) for k, v in metadata_dict.items() if pd.notna(v)}
        
        if kv_pairs:
            try:
                map_ann_id = ezomero.post_map_annotation(conn, obj_type, obj_id, kv_pairs, ns=namespace)
                if map_ann_id:
                    log.debug(f"Successfully applied metadata (Annotation ID: {map_ann_id})")
                else:
                    log.error(f"Failed to apply metadata to {obj_type} {obj_id}")
                    success = False
            except Exception as e:
                log.error(f"Error applying metadata to {obj_type} {obj_id}: {e}")
                success = False
        else:
            log.warning(f"Metadata for {obj_type} {obj_id}, ns={namespace} empty after filtering. Skipping.")
    
    return success

def _apply_assay_conditions_to_wells(conn, plate_id, plate_identifier, assay_conditions_df, namespace):
    """Internal function to apply AssayConditions metadata to wells."""
    log.info(f"Processing Wells for Plate ID: {plate_id} (Identifier: '{plate_identifier}')")
    success_count = 0
    fail_count = 0

    # 1. Filter metadata and create well name lookup
    try:
        # Ensure comparison works correctly (string vs string)
        assay_conditions_df['Plate'] = assay_conditions_df['Plate'].astype(str)
        plate_metadata = assay_conditions_df[assay_conditions_df['Plate'] == str(plate_identifier)].copy()

        if plate_metadata.empty:
            log.warning(f"No metadata found for Plate identifier '{plate_identifier}' in AssayConditions.")
            try: 
                plate = conn.getObject("Plate", plate_id)
                omero_wells = list(plate.listChildren()) if plate else []
                return 0, len(omero_wells)
            except Exception: 
                return 0, 0

        if 'Well' not in plate_metadata.columns:
             log.error(f"Missing 'Well' column for Plate '{plate_identifier}'. Cannot map wells.")
             try: 
                 plate = conn.getObject("Plate", plate_id)
                 omero_wells = list(plate.listChildren()) if plate else []
                 return 0, len(omero_wells)
             except Exception: 
                 return 0, 0

        # Create a lookup dictionary for metadata by normalized well names
        metadata_lookup = {}
        for _, row in plate_metadata.iterrows():
            well_name = str(row['Well'])
            normalized_well = normalize_well_name(well_name)
            if normalized_well:
                # Exclude 'Plate' and 'Well' columns from the metadata
                well_metadata = {str(k): str(v) for k, v in row.items() 
                               if k not in ['Plate', 'Well'] and pd.notna(v)}
                metadata_lookup[normalized_well] = well_metadata
        
        log.debug(f"Metadata contains {len(metadata_lookup)} wells: {sorted(metadata_lookup.keys())}")

    except KeyError as e: 
        log.error(f"Missing column '{e}' in AssayConditions for Plate '{plate_identifier}'.")
        return 0, 0
    except Exception as e: 
        log.error(f"Error filtering metadata for Plate '{plate_identifier}': {e}")
        return 0, 0

    # 2. Get OMERO wells using direct OMERO API
    try:
        plate = conn.getObject("Plate", plate_id)
        if not plate:
            log.error(f"Could not retrieve Plate object for ID {plate_id}")
            return 0, len(metadata_lookup)
            
        omero_wells = list(plate.listChildren())
        if not omero_wells: 
            log.warning(f"No wells found in OMERO Plate ID {plate_id}.")
            return 0, len(metadata_lookup)
            
        log.debug(f"OMERO contains {len(omero_wells)} wells")
            
    except Exception as e: 
        log.error(f"Error retrieving wells for Plate ID {plate_id}: {e}")
        return 0, len(metadata_lookup)

    # 3. Create a coordinate-based lookup for OMERO wells (don't assume sorting)
    omero_well_coords = {}
    omero_well_names = set()
    
    for well in omero_wells:
        row = well.row
        col = well.column
        well_id = well.getId()
        # Convert to normalized well name (always zero-padded format)
        well_name = f"{chr(ord('A') + row)}{col + 1:02d}"
        omero_well_coords[(row, col)] = {'well_id': well_id, 'well_name': well_name}
        omero_well_names.add(well_name)
    
    log.debug(f"OMERO wells by coordinates: {sorted(omero_well_names)}")

    # 4. Match metadata to wells using coordinates, not order
    processed_well_names = set()
    
    for (row, col), well_info in omero_well_coords.items():
        well_id = well_info['well_id']
        well_name = well_info['well_name']
        processed_well_names.add(well_name)

        if well_name in metadata_lookup:
            well_metadata = metadata_lookup[well_name]

            if not well_metadata: 
                log.debug(f"Metadata for Well '{well_name}' empty after filtering. Skipping.")
                success_count += 1
                continue

            # Apply metadata to the well
            if _apply_metadata_to_object(conn, "Well", well_id, well_metadata, namespace):
                log.debug(f"  Applied metadata to Well ID {well_id} (Name: {well_name}, Row: {row}, Col: {col})")
                success_count += 1
            else: 
                log.error(f"  Failed to apply metadata to Well ID {well_id} (Name: {well_name})")
                fail_count += 1
        else: 
            log.warning(f"  No metadata found for Well '{well_name}' (ID: {well_id}, Row: {row}, Col: {col}) in Plate '{plate_identifier}'.")
            fail_count += 1

    # 5. Check for extra metadata wells that weren't found in OMERO
    metadata_wells = set(metadata_lookup.keys())
    extra_metadata = metadata_wells - processed_well_names
    if extra_metadata: 
        log.warning(f"  Metadata found for wells not in OMERO Plate {plate_id}: {', '.join(sorted(list(extra_metadata)))}")
        fail_count += len(extra_metadata)

    # Debug summary
    log.debug(f"Well matching summary for Plate {plate_id}:")
    log.debug(f"  OMERO wells: {sorted(omero_well_names)}")
    log.debug(f"  Metadata wells: {sorted(metadata_wells)}")
    log.debug(f"  Matched wells: {sorted(omero_well_names.intersection(metadata_wells))}")

    log.info(f"Plate {plate_id} ('{plate_identifier}') processing complete. Success: {success_count}, Failures: {fail_count}")
    return success_count, fail_count

# --- Enhanced Upload Function (Phase 4) ---
def annotate_omero_object(conn, target_object_type, target_object_id, metadata_json, base_namespace=DEFAULT_NS_BASE, replace=False):
    """
    Annotates an OMERO object (Screen or Plate) and its children (Wells) based on MIHCSME structure.

    Args:
        conn: Active OMERO connection object.
        target_object_type (str): Type of the main target object ("Screen" or "Plate").
        target_object_id (int): ID of the main target OMERO object.
        metadata_json (dict): The JSON-like dictionary returned by convert_excel_to_json.
        base_namespace (str, optional): Base namespace for annotations. Defaults to DEFAULT_NS_BASE.
        replace (bool, optional): If True, remove existing annotations with target namespaces before applying new ones.

    Returns:
        dict: Summary of annotation process.
    """
    summary = {'status': 'error', 'message': 'Initialization failed',
               'target_type': target_object_type, 'target_id': target_object_id,
               'wells_processed': 0, 'wells_succeeded': 0, 'wells_failed': 0,
               'removed_annotations': 0}
    processed_ok = True

    if target_object_type.lower() not in ["screen", "plate"]:
        summary['message'] = f"This function only supports 'Screen' or 'Plate' as target object types, not '{target_object_type}'."
        log.error(summary['message'])
        return summary

    try:
        # If replace=True, remove existing annotations first
        if replace:
            log.info(f"Replacing existing metadata for {target_object_type} {target_object_id}...")
            if target_object_type.lower() == "screen":
                removal_summary = remove_screen_metadata(conn, target_object_id, base_namespace, delete_annotations=True)
            else:  # plate
                removal_summary = remove_plate_metadata(conn, target_object_id, base_namespace, delete_annotations=True)
            
            summary['removed_annotations'] = removal_summary['total_removed']
            
            if removal_summary['errors']:
                summary['message'] = f"Errors during metadata removal: {removal_summary['errors']}"
                return summary
            
            log.info(f"Removed {removal_summary['total_removed']} existing annotations")

        # 1. Apply Object-Level Metadata (Screen or Plate level)
        log.info(f"--- Applying Metadata to {target_object_type} {target_object_id} ---")
        ns_investigation = f"{base_namespace}/{SHEET_INVESTIGATION}"
        ns_study = f"{base_namespace}/{SHEET_STUDY}"
        ns_assay = f"{base_namespace}/{SHEET_ASSAY}"

        processed_ok &= _apply_metadata_to_object(conn, target_object_type, target_object_id, metadata_json.get(SHEET_INVESTIGATION), ns_investigation)
        processed_ok &= _apply_metadata_to_object(conn, target_object_type, target_object_id, metadata_json.get(SHEET_STUDY), ns_study)
        processed_ok &= _apply_metadata_to_object(conn, target_object_type, target_object_id, metadata_json.get(SHEET_ASSAY), ns_assay)

        # 2. Apply Well-Level Metadata
        log.info("--- Applying Well-Level Metadata (AssayConditions) ---")
        assay_conditions_list = metadata_json.get(SHEET_CONDITIONS)
        ns_conditions = f"{base_namespace}/{SHEET_CONDITIONS}"

        if not assay_conditions_list or not isinstance(assay_conditions_list, list):
            log.warning(f"'{SHEET_CONDITIONS}' data missing or not a list in input JSON. Skipping well metadata.")
        else:
            # Convert list of dicts back to DataFrame for easier processing
            assay_conditions_df = pd.DataFrame(assay_conditions_list)
            # Ensure Plate and Well are strings if they exist
            if 'Plate' in assay_conditions_df.columns: 
                assay_conditions_df['Plate'] = assay_conditions_df['Plate'].astype(str)
            if 'Well' in assay_conditions_df.columns: 
                assay_conditions_df['Well'] = assay_conditions_df['Well'].astype(str)

            # Handle different target types
            if target_object_type.lower() == "screen":
                # Get plates from screen
                screen = conn.getObject("Screen", target_object_id)
                if not screen:
                    log.warning(f"Could not retrieve Screen object for ID {target_object_id}.")
                    plates = []
                else:
                    plates = list(screen.listChildren())
            else:  # plate
                # Single plate target
                plate = conn.getObject("Plate", target_object_id)
                if not plate:
                    log.warning(f"Could not retrieve Plate object for ID {target_object_id}.")
                    plates = []
                else:
                    plates = [plate]

            if not plates:
                log.warning(f"No plates found for {target_object_type} ID {target_object_id}.")
            else:
                log.info(f"Found {len(plates)} plate(s) to process.")
                total_well_success = 0
                total_well_fail = 0
                
                for plate in plates:
                    plate_id = plate.getId()
                    plate_identifier = plate.getName()
                    log.debug(f"Processing Plate ID: {plate_id}, Identifier: '{plate_identifier}'")
                    s, f = _apply_assay_conditions_to_wells(conn, plate_id, plate_identifier, assay_conditions_df, ns_conditions)
                    total_well_success += s
                    total_well_fail += f
            
                summary['wells_succeeded'] = total_well_success
                summary['wells_failed'] = total_well_fail
                summary['wells_processed'] = total_well_success + total_well_fail
                log.info(f"Well metadata summary: Processed={summary['wells_processed']}, Success={total_well_success}, Failures={total_well_fail}")

        if processed_ok:
            summary['status'] = 'success'
            if replace:
                summary['message'] = f'Metadata replaced successfully: removed {summary["removed_annotations"]} old annotations, applied new metadata (check logs for well details).'
            else:
                summary['message'] = 'Annotations applied successfully (check logs for well details).'
        else:
             summary['status'] = 'partial_success'
             summary['message'] = f'Some {target_object_type.lower()}-level annotations may have failed (check logs).'

        log.info(f"Annotation process finished for {target_object_type} {target_object_id}.")

    except Exception as e:
        summary['message'] = f"An unexpected error occurred during annotation: {e}"
        log.error(summary['message'], exc_info=True)
        summary['status'] = 'error'

    return summary

# --- Annotation Management Functions ---
def remove_screen_metadata(conn, screen_id, namespace=None, delete_annotations=True):
    """
    Remove all annotations from a Screen and its child objects (Plates, Wells)
    
    Args:
        conn: OMERO connection
        screen_id (int): Screen ID
        namespace (str, optional): If specified, only remove annotations with this namespace
        delete_annotations (bool): If True, delete annotations. If False, just unlink them.
        
    Returns:
        dict: Summary of removal operation
    """
    summary = {
        'screen_annotations': 0,
        'plate_annotations': 0,
        'well_annotations': 0,
        'total_removed': 0,
        'errors': []
    }
    
    try:
        # Get screen object
        screen = conn.getObject("Screen", screen_id)
        if not screen:
            summary['errors'].append(f"Screen {screen_id} not found")
            return summary
        
        # Remove annotations from Screen
        screen_ann_count = _remove_object_annotations(conn, "Screen", screen_id, namespace, delete_annotations)
        summary['screen_annotations'] = screen_ann_count
        
        # Get all plates in the screen
        plates = list(screen.listChildren())
        for plate in plates:
            plate_id = plate.getId()
            
            # Remove annotations from Plate
            plate_ann_count = _remove_object_annotations(conn, "Plate", plate_id, namespace, delete_annotations)
            summary['plate_annotations'] += plate_ann_count
            
            # Remove annotations from Wells in this Plate
            wells = list(plate.listChildren())
            for well in wells:
                well_id = well.getId()
                well_ann_count = _remove_object_annotations(conn, "Well", well_id, namespace, delete_annotations)
                summary['well_annotations'] += well_ann_count
        
        summary['total_removed'] = summary['screen_annotations'] + summary['plate_annotations'] + summary['well_annotations']
        log.info(f"Removed {summary['total_removed']} annotations from Screen {screen_id} and children")
        
    except Exception as e:
        summary['errors'].append(f"Error removing annotations: {str(e)}")
        log.error(f"Error in remove_screen_metadata: {e}")
    
    return summary

def remove_plate_metadata(conn, plate_id, namespace=None, delete_annotations=True):
    """
    Remove all annotations from a Plate and its child Wells
    
    Args:
        conn: OMERO connection
        plate_id (int): Plate ID
        namespace (str, optional): If specified, only remove annotations with this namespace
        delete_annotations (bool): If True, delete annotations. If False, just unlink them.
        
    Returns:
        dict: Summary of removal operation
    """
    summary = {
        'plate_annotations': 0,
        'well_annotations': 0,
        'total_removed': 0,
        'errors': []
    }
    
    try:
        # Get plate object
        plate = conn.getObject("Plate", plate_id)
        if not plate:
            summary['errors'].append(f"Plate {plate_id} not found")
            return summary
        
        # Remove annotations from Plate
        plate_ann_count = _remove_object_annotations(conn, "Plate", plate_id, namespace, delete_annotations)
        summary['plate_annotations'] = plate_ann_count
        
        # Remove annotations from Wells in this Plate
        wells = list(plate.listChildren())
        for well in wells:
            well_id = well.getId()
            well_ann_count = _remove_object_annotations(conn, "Well", well_id, namespace, delete_annotations)
            summary['well_annotations'] += well_ann_count
        
        summary['total_removed'] = summary['plate_annotations'] + summary['well_annotations']
        log.info(f"Removed {summary['total_removed']} annotations from Plate {plate_id} and children")
        
    except Exception as e:
        summary['errors'].append(f"Error removing annotations: {str(e)}")
        log.error(f"Error in remove_plate_metadata: {e}")
    
    return summary

def _remove_object_annotations(conn, object_type, object_id, namespace=None, delete_annotations=True):
    """
    Remove annotations from a single object
    
    Args:
        conn: OMERO connection
        object_type (str): OMERO object type
        object_id (int): Object ID
        namespace (str, optional): Filter by namespace
        delete_annotations (bool): If True, delete. If False, unlink.
        
    Returns:
        int: Number of annotations removed
    """
    count = 0
    try:
        obj = conn.getObject(object_type, object_id)
        if not obj:
            return 0
        
        annotations_to_remove = []
        for ann in obj.listAnnotations():
            # Filter by namespace if specified
            if namespace and hasattr(ann, 'getNs'):
                ann_ns = ann.getNs()
                if ann_ns and not ann_ns.startswith(namespace):
                    continue
            
            if delete_annotations:
                # Delete the annotation itself
                annotations_to_remove.append(ann.getId())
            else:
                # Just unlink (remove the link)
                link_id = ann.link.getId() if hasattr(ann, 'link') else None
                if link_id:
                    annotations_to_remove.append(link_id)
        
        if annotations_to_remove:
            if delete_annotations:
                conn.deleteObjects('Annotation', annotations_to_remove, wait=True)
            else:
                link_type = f"{object_type}AnnotationLink"
                conn.deleteObjects(link_type, annotations_to_remove, wait=True)
            count = len(annotations_to_remove)
        
    except Exception as e:
        log.error(f"Error removing annotations from {object_type} {object_id}: {e}")
    
    return count

# --- Main Execution Block ---
def main():
    """Basic command-line execution"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log.info("Running OMERO MIHCSME Uploader from command line...")

    parser = argparse.ArgumentParser(description="OMERO MIHCSME Metadata Uploader with Validation")
    parser.add_argument("excel_file", help="Path to the MIHCSME Excel file.")
    parser.add_argument("-s", "--screen_id", type=int, help="ID of the target OMERO Screen.")
    parser.add_argument("-p", "--plate_id", type=int, help="ID of the target OMERO Plate.")
    parser.add_argument("-H", "--host", help="OMERO host.")
    parser.add_argument("--port", type=int, help="OMERO port.")
    parser.add_argument("-u", "--user", help="OMERO username.")
    parser.add_argument("-g", "--group", help="OMERO group.")
    parser.add_argument("--namespace", default=DEFAULT_NS_BASE, help=f"Base namespace for annotations (default: {DEFAULT_NS_BASE})")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation, don't upload")
    parser.add_argument("--replace", action="store_true", help="Replace existing metadata (remove old annotations first)")

    args = parser.parse_args()

    # Determine target
    if args.screen_id and args.plate_id:
        log.error("Specify either --screen_id OR --plate_id, not both")
        sys.exit(1)
    elif args.screen_id:
        target_type, target_id = "Screen", args.screen_id
    elif args.plate_id:
        target_type, target_id = "Plate", args.plate_id
    else:
        log.error("Must specify either --screen_id or --plate_id")
        sys.exit(1)

    # Connect to OMERO
    try:
        password = getpass.getpass(prompt=f"OMERO Password for user '{args.user or 'default'}': ")
        conn = ezomero.connect(user=args.user, password=password, group=args.group,
                               host=args.host, port=args.port, secure=True)
        if conn is None: 
            raise ConnectionError("OMERO connection failed.")

        if args.validate_only:
            # Run validation only
            log.info("Running validation only...")
            validation_report = validate_mihcsme_for_omero(args.excel_file, conn, target_type, target_id)
            
            print("\n" + "="*60)
            print("VALIDATION REPORT")
            print("="*60)
            print(f"Overall Status: {validation_report['overall_status'].upper()}")
            print(f"Can Proceed: {'✅ Yes' if validation_report['can_proceed'] else '❌ No'}")
            
            if validation_report['file_validation']['errors']:
                print(f"\n❌ File Errors:")
                for error in validation_report['file_validation']['errors']:
                    print(f"  - {error}")
            
            if validation_report['omero_validation'].get('structure_validation', {}).get('errors'):
                print(f"\n❌ Structure Errors:")
                for error in validation_report['omero_validation']['structure_validation']['errors']:
                    print(f"  - {error}")
            
            if validation_report['file_validation']['warnings'] or validation_report['omero_validation'].get('structure_validation', {}).get('warnings'):
                print(f"\n⚠️ Warnings:")
                for warning in validation_report['file_validation']['warnings']:
                    print(f"  - {warning}")
                for warning in validation_report['omero_validation'].get('structure_validation', {}).get('warnings', []):
                    print(f"  - {warning}")
            
            print(f"\n📋 Recommendations:")
            for rec in validation_report['recommendations']:
                print(f"  - {rec}")
            
        else:
            # Convert Excel to JSON and run upload
            log.info("Converting Excel to JSON...")
            metadata_json = convert_excel_to_json(args.excel_file)
            
            # Run upload using the core function
            result = annotate_omero_object(conn, target_type, target_id, metadata_json, args.namespace, replace=args.replace)
            
            print(f"\n📊 Upload Results:")
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            print(f"Wells processed: {result['wells_processed']}")
            print(f"Wells succeeded: {result['wells_succeeded']}")
            print(f"Wells failed: {result['wells_failed']}")
            if result.get('removed_annotations', 0) > 0:
                print(f"Annotations removed: {result['removed_annotations']}")
            
            if result['status'] == 'error':
                sys.exit(1)

    except Exception as e:
        log.error(f"Command-line execution failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'conn' in locals() and conn: 
            conn.close()
            log.info("OMERO connection closed.")

if __name__ == "__main__":
    main()