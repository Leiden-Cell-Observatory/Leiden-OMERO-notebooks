# ```python
#!/usr/bin/env python3

"""
OMERO MIHCSME Metadata Uploader Library

Provides functions to convert MIHCSME Excel to JSON-like dict
and annotate OMERO objects (Screen and contained Wells).
"""

import json
import logging
import sys
import getpass
from pathlib import Path
import pandas as pd
import ezomero
import argparse # Keep for potential command-line usage

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

# --- Core Annotation Logic (Internal) ---
def _apply_metadata_to_object(conn, obj_type, obj_id, metadata_dict, namespace):
    """Internal function to apply key-value pairs as MapAnnotation."""
    if not metadata_dict:
        log.debug(f"No metadata for {obj_type} {obj_id}, ns={namespace}. Skipping.")
        return True
    log.info(f"Applying metadata to {obj_type} {obj_id} (Namespace: {namespace})")
    kv_pairs = {str(k): str(v) for k, v in metadata_dict.items() if pd.notna(v)}
    if not kv_pairs:
         log.warning(f"Metadata for {obj_type} {obj_id}, ns={namespace} empty after filtering. Skipping.")
         return True
    try:
        map_ann_id = ezomero.post_map_annotation(conn, obj_type, obj_id, kv_pairs, ns=namespace)
        if map_ann_id:
            log.debug(f"Successfully applied metadata (Annotation ID: {map_ann_id})")
            return True
        else:
            log.error(f"Failed to apply metadata to {obj_type} {obj_id} (ezomero returned None)")
            return False
    except Exception as e:
        log.error(f"Error applying metadata to {obj_type} {obj_id} for ns {namespace}: {e}")
        return False

def _apply_assay_conditions_to_wells(conn, plate_id, plate_identifier, assay_conditions_df, namespace):
    """Internal function to apply AssayConditions metadata to wells."""
    log.info(f"Processing Wells for Plate ID: {plate_id} (Identifier: '{plate_identifier}')")
    success_count = 0
    fail_count = 0
    omero_wells = None

    # 1. Filter metadata
    try:
        # Ensure comparison works correctly (string vs string)
        assay_conditions_df['Plate'] = assay_conditions_df['Plate'].astype(str)
        plate_metadata = assay_conditions_df[assay_conditions_df['Plate'] == str(plate_identifier)].copy()

        if plate_metadata.empty:
            log.warning(f"No metadata found for Plate identifier '{plate_identifier}' in AssayConditions.")
            try: omero_wells_check = ezomero.get_wells(conn, plate_id, images=False); return 0, len(omero_wells_check) if omero_wells_check else 0
            except Exception: return 0, 0 # Can't determine well count

        if 'Well' not in plate_metadata.columns:
             log.error(f"Missing 'Well' column for Plate '{plate_identifier}'. Cannot map wells.")
             try: omero_wells_check = ezomero.get_wells(conn, plate_id, images=False); return 0, len(omero_wells_check) if omero_wells_check else 0
             except Exception: return 0, 0

        plate_metadata['Well'] = plate_metadata['Well'].astype(str)
        plate_metadata.set_index('Well', inplace=True) # Use Well name as index

    except KeyError as e: log.error(f"Missing column '{e}' in AssayConditions for Plate '{plate_identifier}'."); return 0, 0
    except Exception as e: log.error(f"Error filtering metadata for Plate '{plate_identifier}': {e}"); return 0, 0

    # 2. Get OMERO wells and dimensions
    try:
        omero_wells = ezomero.get_wells(conn, plate_id, images=False)
        if not omero_wells: log.warning(f"No wells found in OMERO Plate ID {plate_id}."); return 0, len(plate_metadata) if not plate_metadata.empty else 0
        plate_dims = ezomero.get_plate_dimensions(conn, plate_id)
        if not plate_dims: log.warning(f"Could not get dimensions for Plate ID {plate_id}."); return 0, len(omero_wells)
    except Exception as e: log.error(f"Error retrieving wells/dims for Plate ID {plate_id}: {e}"); return 0, len(omero_wells) if omero_wells else (len(plate_metadata) if not plate_metadata.empty else 0)

    # 3. Iterate and apply
    processed_well_names = set()
    for (r, c), well_id in omero_wells.items():
        well_name = well_coord_to_name(r, c, rows=plate_dims[0], cols=plate_dims[1])
        if not well_name: log.warning(f"Skipping invalid coord ({r},{c}) for Well ID {well_id}"); fail_count += 1; continue
        processed_well_names.add(well_name)

        if well_name in plate_metadata.index:
            well_data_series = plate_metadata.loc[well_name]
            # Exclude 'Plate' column if it exists in the series (already used for filtering)
            well_kv_dict = {str(k): str(v) for k, v in well_data_series.items() if k != 'Plate' and pd.notna(v)}

            if not well_kv_dict: log.debug(f"Metadata for Well '{well_name}' empty. Skipping."); success_count += 1; continue

            # Apply metadata to the well using the internal function
            if _apply_metadata_to_object(conn, "Well", well_id, well_kv_dict, namespace):
                log.debug(f"  Applied metadata to Well ID {well_id} (Name: {well_name})")
                success_count += 1
            else: log.error(f"  Failed to apply metadata to Well ID {well_id} (Name: {well_name})"); fail_count += 1
        else: log.warning(f"  No metadata found for Well '{well_name}' (ID: {well_id}) in Plate '{plate_identifier}'."); fail_count += 1

    # 4. Check for extra metadata
    extra_metadata = set(plate_metadata.index) - processed_well_names
    if extra_metadata: log.warning(f"  Metadata found for wells not matched in OMERO Plate {plate_id}: {', '.join(sorted(list(extra_metadata)))}"); fail_count += len(extra_metadata)

    log.info(f"Plate {plate_id} ('{plate_identifier}') processing complete. Success: {success_count}, Failures: {fail_count}")
    return success_count, fail_count

# --- Public Functions ---
def convert_excel_to_json(excel_path):
    """
    Parses a MIHCSME Excel file and returns a JSON-serializable dictionary.

    Args:
        excel_path (str or Path): Path to the MIHCSME Excel file.

    Returns:
        dict or None: A dictionary representing the metadata, ready for JSON
                      serialization (AssayConditions as list of dicts).
                      Returns None on failure.
    """
    log.info(f"Converting Excel to JSON structure: {excel_path}")
    filepath = Path(excel_path)
    required_sheets = [SHEET_INVESTIGATION, SHEET_STUDY, SHEET_ASSAY, SHEET_CONDITIONS]
    data = {}
    try:
        xls = pd.ExcelFile(filepath)
        available_sheets = xls.sheet_names
        missing_sheets = [s for s in required_sheets if s not in available_sheets]
        if missing_sheets:
            log.error(f"Missing required sheets in Excel file: {', '.join(missing_sheets)}")
            return None

        for sheet_name in available_sheets:
            if sheet_name.startswith("_"): continue
            log.debug(f"Processing sheet: {sheet_name}")
            if sheet_name == SHEET_CONDITIONS:
                df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
                if 'Plate' not in df.columns or 'Well' not in df.columns:
                     log.error(f"Sheet '{sheet_name}' missing required 'Plate' or 'Well' column.")
                     return None
                # Convert DataFrame to list of dicts, replacing NaN with None
                data[sheet_name] = df.where(pd.notnull(df), None).to_dict(orient='records')
                log.info(f"Parsed '{sheet_name}' as list of dicts ({len(data[sheet_name])} records).")
            elif sheet_name in [SHEET_INVESTIGATION, SHEET_STUDY, SHEET_ASSAY]:
                try:
                    df_kv = pd.read_excel(xls, sheet_name=sheet_name, header=None, index_col=0, usecols=[0, 1], dtype=str)
                    # Convert NaN values from pandas to None for JSON compatibility
                    data[sheet_name] = {k: (v if pd.notna(v) else None) for k, v in df_kv[1].dropna().to_dict().items()}
                    log.info(f"Parsed '{sheet_name}' key-value sheet.")
                except Exception as e:
                    log.error(f"Error parsing key-value sheet '{sheet_name}': {e}")
                    return None
            else:
                log.warning(f"Skipping unrecognized sheet: {sheet_name}")
        return data
    except FileNotFoundError:
        log.error(f"Excel file not found: {filepath}")
        return None
    except Exception as e:
        log.error(f"Failed to parse Excel file '{filepath}': {e}")
        return None

def annotate_omero_object(conn, target_object_type, target_object_id, metadata_json, base_namespace=DEFAULT_NS_BASE):
    """
    Annotates an OMERO object (Screen) and its children (Wells) based on MIHCSME structure.

    Args:
        conn: Active ezomero connection object.
        target_object_type (str): Type of the main target object (e.g., "Screen").
        target_object_id (int): ID of the main target OMERO object.
        metadata_json (dict): The JSON-like dictionary returned by convert_excel_to_json.
        base_namespace (str, optional): Base namespace for annotations. Defaults to DEFAULT_NS_BASE.

    Returns:
        dict: Summary of annotation process.
    """
    summary = {'status': 'error', 'message': 'Initialization failed',
               'target_type': target_object_type, 'target_id': target_object_id,
               'wells_processed': 0, 'wells_succeeded': 0, 'wells_failed': 0}
    processed_ok = True

    if target_object_type.lower() != "screen":
        summary['message'] = f"This function currently only supports 'Screen' as the target object type, not '{target_object_type}'."
        log.error(summary['message'])
        return summary

    screen_id = target_object_id

    try:
        # 1. Apply Screen-Level Metadata
        log.info(f"--- Applying Metadata to {target_object_type} {target_object_id} ---")
        ns_investigation = f"{base_namespace}/{SHEET_INVESTIGATION}"
        ns_study = f"{base_namespace}/{SHEET_STUDY}"
        ns_assay = f"{base_namespace}/{SHEET_ASSAY}"

        processed_ok &= _apply_metadata_to_object(conn, target_object_type, screen_id, metadata_json.get(SHEET_INVESTIGATION), ns_investigation)
        processed_ok &= _apply_metadata_to_object(conn, target_object_type, screen_id, metadata_json.get(SHEET_STUDY), ns_study)
        processed_ok &= _apply_metadata_to_object(conn, target_object_type, screen_id, metadata_json.get(SHEET_ASSAY), ns_assay)

        # 2. Apply Well-Level Metadata
        log.info("--- Applying Well-Level Metadata (AssayConditions) ---")
        assay_conditions_list = metadata_json.get(SHEET_CONDITIONS)
        ns_conditions = f"{base_namespace}/{SHEET_CONDITIONS}"

        if not assay_conditions_list or not isinstance(assay_conditions_list, list):
            log.warning(f"'{SHEET_CONDITIONS}' data missing or not a list in input JSON. Skipping well metadata.")
        else:
            # Convert list of dicts back to DataFrame for easier processing by internal function
            assay_conditions_df = pd.DataFrame(assay_conditions_list)
             # Ensure Plate and Well are strings if they exist
            if 'Plate' in assay_conditions_df.columns: assay_conditions_df['Plate'] = assay_conditions_df['Plate'].astype(str)
            if 'Well' in assay_conditions_df.columns: assay_conditions_df['Well'] = assay_conditions_df['Well'].astype(str)

            plate_ids = ezomero.get_plate_ids(conn, screen=screen_id)
            if not plate_ids:
                log.warning(f"No plates found linked to Screen ID {screen_id}.")
            else:
                log.info(f"Found {len(plate_ids)} plates linked to the screen.")
                total_well_success = 0
                total_well_fail = 0
                for plate_id in plate_ids:
                    plate = conn.getObject("Plate", plate_id)
                    if plate:
                        plate_identifier = plate.getName() # Assumes Plate Name matches 'Plate' column
                        log.debug(f"Processing Plate ID: {plate_id}, Identifier: '{plate_identifier}'")
                        s, f = _apply_assay_conditions_to_wells(conn, plate_id, plate_identifier, assay_conditions_df, ns_conditions)
                        total_well_success += s
                        total_well_fail += f
                    else:
                        log.warning(f"Could not retrieve Plate object for ID {plate_id}. Skipping wells.")
                        # How to count failures here? Difficult without knowing expected wells.
                summary['wells_succeeded'] = total_well_success
                summary['wells_failed'] = total_well_fail
                summary['wells_processed'] = total_well_success + total_well_fail
                log.info(f"Well metadata summary: Processed={summary['wells_processed']}, Success={total_well_success}, Failures={total_well_fail}")

        if processed_ok:
            summary['status'] = 'success'
            summary['message'] = 'Annotations applied successfully (check logs for well details).'
        else:
             summary['status'] = 'partial_success'
             summary['message'] = 'Some screen-level annotations may have failed (check logs).'

        log.info(f"Annotation process finished for {target_object_type} {target_object_id}.")

    except Exception as e:
        summary['message'] = f"An unexpected error occurred during annotation: {e}"
        log.error(summary['message'], exc_info=True)
        summary['status'] = 'error'

    return summary


# --- Main Execution Block (Optional: for basic command-line use) ---
def main():
    """ Basic command-line execution """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log.info("Running OMERO Uploader from command line...")

    parser = argparse.ArgumentParser(description="OMERO MIHCSME Metadata Uploader (Library Script)")
    parser.add_argument("excel_file", help="Path to the MIHCSME Excel file.")
    parser.add_argument("-s", "--screen_id", type=int, required=True, help="ID of the target OMERO Screen.")
    parser.add_argument("-H", "--host", help="OMERO host.")
    parser.add_argument("-p", "--port", type=int, help="OMERO port.")
    parser.add_argument("-u", "--user", help="OMERO username.")
    parser.add_argument("-g", "--group", help="OMERO group.")
    parser.add_argument("--namespace", default=DEFAULT_NS_BASE, help=f"Base namespace for annotations (default: {DEFAULT_NS_BASE})")

    args = parser.parse_args()

    # 1. Convert Excel
    metadata = convert_excel_to_json(args.excel_file)
    if not metadata:
        sys.exit(1)

    # 2. Connect
    conn = None
    try:
        password = getpass.getpass(prompt=f"OMERO Password for user '{args.user or 'default'}': ")
        conn = ezomero.connect(user=args.user, password=password, group=args.group,
                               host=args.host, port=args.port, secure=True)
        if conn is None: raise ConnectionError("OMERO connection failed.")

        # 3. Annotate
        result = annotate_omero_object(conn, "Screen", args.screen_id, metadata, args.namespace)
        log.info(f"Annotation Result: {result}")

    except Exception as e:
        log.error(f"Command-line execution failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if conn: conn.close(); log.info("OMERO connection closed.")

    if result.get('status') == 'error':
         sys.exit(1)

if __name__ == "__main__":
    main()
# ```
#