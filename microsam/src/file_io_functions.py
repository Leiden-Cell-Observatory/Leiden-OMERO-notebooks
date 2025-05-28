"""
Functions for file operations, including zarr storage and local annotation organization.
"""
import os
import shutil
import zipfile
import zarr
import json
import pandas as pd
from tifffile import imwrite
from .utils import NumpyEncoder


def zip_directory(source_path, zarr_path, zip_file):
    """
    Zip a directory while handling null characters in paths.
    
    Args:
        source_path: Base source path
        zarr_path: Path to zarr directory
        zip_file: ZipFile object to write to
    """
    for root, dirs, files in os.walk(zarr_path):
        for file in files:
            try:
                # Create paths
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, source_path)
                
                # Remove null characters while preserving the path structure
                safe_full_path = full_path.replace('\x00', '')
                safe_rel_path = rel_path.replace('\x00', '')
                
                # Add file to zip if it exists
                if os.path.exists(safe_full_path):
                    zip_file.write(safe_full_path, safe_rel_path)
            except Exception as e:
                print(f"Warning: Error processing {file}: {str(e)}")
                continue


def store_annotations_in_zarr(mask_data, output_folder, image_num):
    """
    Store annotation masks in zarr format for efficient access
    
    Args:
        mask_data: Numpy array with mask data
        output_folder: Base folder to store zarr data
        image_num: Image number/identifier
        
    Returns:
        path: Path to the zarr store
    """
    # Create zarr directory if it doesn't exist
    zarr_dir = os.path.join(output_folder, "annotations")
    os.makedirs(zarr_dir, exist_ok=True)
    
    # Create zarr filename
    zarr_path = os.path.join(zarr_dir, f"annotation_{image_num:05d}.zarr")
    
    # Remove existing zarr store if it exists
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
        
    # Create zarr array from mask data
    z = zarr.open(zarr_path, mode='w')
    z.create_dataset('masks', data=mask_data, chunks=(256, 256))
    
    # Return path to zarr store
    return zarr_path


def zarr_to_tiff(zarr_path, output_tiff_path):
    """
    Convert zarr store to TIFF file for OMERO upload
    
    Args:
        zarr_path: Path to zarr store
        output_tiff_path: Path to save TIFF file
        
    Returns:
        output_tiff_path: Path to saved TIFF file
    """
    # Load data from zarr
    z = zarr.open(zarr_path, mode='r')
    mask_data = z['masks'][:]
    
    # Save as TIFF
    imwrite(output_tiff_path, mask_data)
    
    return output_tiff_path


def cleanup_local_embeddings(output_folder):
    """
    Check for and clean up any existing embeddings from previous interrupted runs
    
    Args:
        output_folder: Path to the output folder containing embeddings
    """
    embed_path = os.path.join(output_folder, "embed")
    
    if os.path.exists(embed_path):
        # Look for embedding zarr directories and zip files
        for item in os.listdir(embed_path):
            item_path = os.path.join(embed_path, item)
            if os.path.isdir(item_path) and "embedding_" in item and item.endswith(".zarr"):
                print(f"Cleaning up leftover embedding directory: {item}")
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path) and "embedding_" in item and item.endswith(".zip"):
                print(f"Cleaning up leftover embedding zip: {item}")
                os.remove(item_path)
    
    # Check output directory for segmentation files
    output_path = os.path.join(output_folder, "output")
    if os.path.exists(output_path):
        for item in os.listdir(output_path):
            item_path = os.path.join(output_path, item)
            if os.path.isfile(item_path) and "seg_" in item and (item.endswith(".tif") or item.endswith(".tiff")):
                print(f"Cleaning up leftover segmentation file: {item}")
                os.remove(item_path)


def organize_local_outputs(local_dir, container_type, container_id, image_id, timepoint=0, z_slice=0, 
                          is_patch=False, patch_coords=None):
    """
    Organize local storage for annotations when working with read-only OMERO servers
    
    Args:
        local_dir: Base directory for local storage
        container_type: Type of OMERO container ('dataset', 'plate', etc.)
        container_id: ID of the container
        image_id: ID of the image
        timepoint: Time point index
        z_slice: Z-slice index or 'all' for volumetric data
        is_patch: Whether this is a patch of a larger image
        patch_coords: Optional tuple of (x, y, width, height) for patch info
    
    Returns:
        dict: Dictionary with paths for various outputs
    """
    # Create the base container directory
    container_path = os.path.join(local_dir, f"{container_type}_{container_id}")
    os.makedirs(container_path, exist_ok=True)
    
    # Create image-specific directory
    image_path = os.path.join(container_path, f"image_{image_id}")
    os.makedirs(image_path, exist_ok=True)
    
    # Create subdirectories for different outputs
    embed_path = os.path.join(image_path, "embeddings")
    label_path = os.path.join(image_path, "labels")
    roi_path = os.path.join(image_path, "rois")
    
    for path in [embed_path, label_path, roi_path]:
        os.makedirs(path, exist_ok=True)
    
    # Determine file naming based on dimensionality and patch info
    name_parts = []
    
    # Add z-slice info
    if z_slice == 'all':
        name_parts.append("vol")  # Volumetric data
    else:
        name_parts.append(f"z{z_slice}")
    
    # Add timepoint info
    name_parts.append(f"t{timepoint}")
    
    # Add patch info if applicable
    if is_patch and patch_coords:
        x, y, width, height = patch_coords
        name_parts.append(f"patch_x{x}_y{y}_w{width}_h{height}")
    
    # Create base filename
    base_name = "_".join(name_parts)
    
    # Return paths for different output types
    return {
        "base_dir": image_path,
        "embedding_dir": embed_path,
        "embedding_path": os.path.join(embed_path, f"{base_name}_embedding.zip"),
        "label_path": os.path.join(label_path, f"{base_name}_label.tiff"),
        "roi_path": os.path.join(roi_path, f"{base_name}_rois.json"),
        "metadata_path": os.path.join(image_path, f"{base_name}_metadata.json"),
        "base_name": base_name
    }


def save_annotations_schema(metadata_path, image_id, label_path, roi_data, annotation_metadata):
    """
    Save annotation metadata and ROIs in a structured JSON schema for local storage
    
    Args:
        metadata_path: Path to save the metadata JSON file
        image_id: OMERO image ID
        label_path: Path to the label image file
        roi_data: List of ROI data extracted from label image
        annotation_metadata: Dictionary with additional metadata
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create the JSON schema
        schema = {
            "schema_version": "1.0",
            "created_at": pd.Timestamp.now().isoformat(),
            "image": {
                "id": int(image_id),
                "name": annotation_metadata.get("image_name", ""),
                "server": annotation_metadata.get("server", "")
            },
            "annotation": {
                "model": annotation_metadata.get("model_type", ""),
                "is_volumetric": annotation_metadata.get("is_volumetric", False),
                "channel": annotation_metadata.get("channel", 0),
                "z_slice": annotation_metadata.get("z_slice", 0),
                "timepoint": annotation_metadata.get("timepoint", 0),
                "is_patch": annotation_metadata.get("is_patch", False),
                "patch_coords": annotation_metadata.get("patch_coords", None)
            },
            "files": {
                "label_path": os.path.relpath(label_path, os.path.dirname(metadata_path)),
                "embedding_path": annotation_metadata.get("embedding_path", "")
            },
            "rois": roi_data
        }
        
        # Write the schema to file
        with open(metadata_path, 'w') as f:
            json.dump(schema, f, indent=2, cls=NumpyEncoder)
            
        print(f"Saved annotation schema to {metadata_path}")
        return True
        
    except Exception as e:
        print(f"Error saving annotation schema: {e}")
        return False
