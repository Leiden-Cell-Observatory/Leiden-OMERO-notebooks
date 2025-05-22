"""
Main processing pipeline for OMERO batch annotation with micro-SAM.
"""
import os
import shutil
import zipfile
import numpy as np
import pandas as pd
import imageio.v3 as imageio
import napari
from napari.settings import get_settings

from micro_sam.sam_annotator import image_series_annotator

from .utils import interleave_arrays
from .image_functions import generate_patch_coordinates
from .file_io_functions import (
    zip_directory,
    store_annotations_in_zarr,
    zarr_to_tiff,
    cleanup_local_embeddings
)
from .omero_functions import upload_rois_and_labels


def process_omero_batch_with_dask(
    conn,
    images_list,
    output_folder: str,
    container_type: str,
    container_id: int,
    source_desc: str,
    model_type: str = 'vit_l',
    batch_size: int = 3,
    channel: int = 0,
    timepoints: list = [0],
    timepoint_mode: str = "specific",
    z_slices: list = [0],
    z_slice_mode: str = "specific",
    segment_all: bool = True,
    train_n: int = 3,
    validate_n: int = 3,
    three_d: bool = False,
    use_patches: bool = False,
    patch_size: tuple = (512, 512),
    patches_per_image: int = 1,
    random_patches: bool = True,
    resume_from_table: bool = False,
    read_only_mode: bool = False,
    local_output_dir: str = "./omero_annotations"
):
    """
    Process OMERO images in batches for SAM segmentation using dask for lazy loading
    and zarr for temporary annotation storage
    
    Args:
        conn: OMERO connection
        images_list: List of OMERO image objects
        output_folder: Path to store temporary files
        container_type: Type of OMERO container ('dataset', 'plate', 'project', 'screen', 'image')
        container_id: ID of the container
        source_desc: Description of the container (for tracking)
        model_type: SAM model type
        batch_size: Number of images/patches to process at once
        channel: Channel to segment
        timepoints: List of timepoints to process
        timepoint_mode: How to handle timepoints ("all", "random", "specific")
        z_slices: List of Z-slices to process (used only when three_d=False)
        z_slice_mode: How to handle z-slices ("all", "random", "specific")
        segment_all: Segment all images in the dataset or only train/validate subset
        train_n: Number of training images if not segment_all
        validate_n: Number of validation images if not segment_all
        three_d: Whether to use 3D volumetric mode
        use_patches: Whether to extract and process patches instead of full images
        patch_size: Size of patches to extract (width, height)
        patches_per_image: Number of patches to extract from each image (if random_patches=True)
        random_patches: Whether to extract random patches or centered patches
        resume_from_table: Whether to resume annotation from an existing tracking table
        read_only_mode: Whether to save annotations locally instead of uploading to OMERO
        local_output_dir: Directory to save annotations in read-only mode
    
    Returns:
        tuple: (table_id, combined_images)
    """
    import ezomero
    
    # Setup output directories
    output_path = os.path.join(output_folder, "output")
    embed_path = os.path.join(output_folder, "embed")
    zarr_path = os.path.join(output_folder, "zarr")
    
    # Check for and clean up any existing embeddings from interrupted runs
    cleanup_local_embeddings(output_folder)
    
    # Remove directories if they exist
    for path in [output_path, embed_path, zarr_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        
    # Create or retrieve tracking DataFrame with additional columns for the new features
    df = pd.DataFrame(columns=[
        "image_id", "image_name", "train", "validate", 
        "channel", "z_slice", "timepoint", "sam_model", "embed_id", "label_id", "roi_id", 
        "is_volumetric", "processed", "is_patch", "patch_x", "patch_y", "patch_width", "patch_height",
        "schema_attachment_id"  # New column for schema attachment
    ])
    
    table_id = None
    
    # Check if we should resume from an existing table
    if resume_from_table:
        try:
            # Get existing tracking table
            existing_tables = ezomero.get_table_names(conn, container_type.capitalize(), container_id)
            if "micro_sam_training_data" in existing_tables:
                # Get the table ID and data
                table_ids = ezomero.get_table_ids(conn, container_type.capitalize(), container_id)
                for tid in table_ids:
                    table_name = ezomero.get_table_names(conn, container_type.capitalize(), container_id, tid)
                    if table_name == "micro_sam_training_data":
                        table_id = tid
                        existing_df = ezomero.get_table(conn, table_id)
                        
                        # Add any missing columns (for backward compatibility)
                        for col in df.columns:
                            if col not in existing_df.columns:
                                existing_df[col] = None
                        
                        # Ensure schema_attachment_id column exists if resuming
                        if 'schema_attachment_id' not in existing_df.columns:
                            existing_df['schema_attachment_id'] = None
                                
                        df = existing_df
                        
                        print(f"Resuming from existing table ID: {table_id}")
                        print(f"Found {len(df)} previously processed images/patches")
                        break
        except Exception as e:
            print(f"Error retrieving existing table: {e}. Starting fresh.")
            resume_from_table = False
    
    # Get images list (already provided as argument)
    combined_images_sequence = np.zeros(len(images_list))  # Initialize sequence array
    
    # Select images based on segment_all flag
    if segment_all:
        combined_images = images_list
        combined_images_sequence = np.zeros(len(combined_images))  # All treated as training
    else:
        # Check if we have enough images
        if len(images_list) < train_n + validate_n:
            print("Not enough images in container for training and validation")
            raise ValueError(f"Need at least {train_n + validate_n} images but found {len(images_list)}")
            
        # Select random images for training and validation
        train_indices = np.random.choice(len(images_list), train_n, replace=False)
        train_images = [images_list[i] for i in train_indices]
        
        # Get validation images from the remaining ones
        validate_candidates = [img for i, img in enumerate(images_list) if i not in train_indices]
        validate_images = np.random.choice(validate_candidates, validate_n, replace=False)
        
        # Interleave the arrays and create sequence markers
        combined_images, combined_images_sequence = interleave_arrays(train_images, validate_images)
    
    # If resuming, filter out already processed images/patches
    processing_units = []  # Will contain tuples of (image, sequence_val, [metadata])
    
    if resume_from_table and len(df) > 0:
        # For patch mode, we need to check image_id + patch coordinates
        if use_patches:
            # Get list of already processed image+patch combinations
            processed_patches = set()
            for _, row in df[df['processed'] == True].iterrows():
                patch_key = (row['image_id'], row.get('patch_x', 0), row.get('patch_y', 0), 
                             row.get('patch_width', 0), row.get('patch_height', 0))
                processed_patches.add(patch_key)
            
            # Generate all possible patches
            for i, img in enumerate(combined_images):
                img_id = img.getId()
                seq_val = combined_images_sequence[i]
                
                # Get image dimensions
                size_x = img.getSizeX()
                size_y = img.getSizeY()
                
                # Generate patches for this image
                img_patches = generate_patch_coordinates(
                    size_x, size_y, patch_size, patches_per_image, random_patches)
                
                # Filter out already processed patches
                for patch in img_patches:
                    patch_key = (img_id, patch[0], patch[1], patch[2], patch[3])
                    if patch_key not in processed_patches:
                        processing_units.append((img, seq_val, patch))
                        
            print(f"Found {len(processing_units)} remaining patches to process")
            
        else:
            # Get list of already processed image IDs
            processed_ids = set(df[df['processed'] == True]['image_id'].values)
            
            # Filter combined_images
            for i, img in enumerate(combined_images):
                if img.getId() not in processed_ids:
                    processing_units.append((img, combined_images_sequence[i], None))
            
            print(f"Found {len(processing_units)} remaining images to process")
    else:
        # Not resuming, generate all processing units
        if use_patches:
            # Generate patches for all images
            for i, img in enumerate(combined_images):
                seq_val = combined_images_sequence[i]
                
                # Get image dimensions
                size_x = img.getSizeX()
                size_y = img.getSizeY()
                
                # Generate patches for this image
                img_patches = generate_patch_coordinates(
                    size_x, size_y, patch_size, patches_per_image, random_patches)
                
                for patch in img_patches:
                    processing_units.append((img, seq_val, patch))
                    
            print(f"Generated {len(processing_units)} patches to process")
        else:
            # Use full images
            for i, img in enumerate(combined_images):
                processing_units.append((img, combined_images_sequence[i], None))
    
    # Calculate total number of batches
    total_batches = (len(processing_units) + batch_size - 1) // batch_size
    
    if use_patches:
        print(f"Processing {len(processing_units)} patches in {total_batches} batches")
    else:
        print(f"Processing {len(processing_units)} images in {total_batches} batches")
    
    print(f"3D mode: {three_d}")
    
    # Process images/patches in batches
    for batch_idx in range(total_batches):
        print(f"\nProcessing batch {batch_idx+1}/{total_batches}")
        
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(processing_units))
        batch_units = processing_units[start_idx:end_idx]
        
        # Load batch images as dask arrays for lazy loading
        images = []
        dask_images = []
        image_data = []  # Store metadata about each image/patch
        
        for unit_idx, (image, seq_val, patch) in enumerate(batch_units):
            image_id = image.getId()
            
            # Determine which timepoint to use
            if timepoint_mode == "all":
                # Use all timepoints (not yet supported in this function)
                actual_timepoint = timepoints[0]  # Default to first timepoint for now
                print("Warning: 'all' timepoint mode not fully supported yet, using first timepoint")
            elif timepoint_mode == "random":
                # Select a random timepoint from the list
                actual_timepoint = np.random.choice(timepoints)
            else:  # "specific"
                # Use the first timepoint in the list
                actual_timepoint = timepoints[0]
            
            # For 3D mode or 2D with patches
            if three_d:
                # 3D mode - process entire Z-stack or specified z-range
                pixels = image.getPrimaryPixels()
                
                if patch is not None:
                    # Extract 3D patch (x, y, z-stack)
                    x, y, width, height = patch
                    img_3d = np.zeros((image.getSizeZ(), height, width), dtype=np.uint16)
                    
                    # Load each z-slice for the patch
                    for z in range(image.getSizeZ()):
                        full_plane = pixels.getPlane(z, channel, actual_timepoint)
                        img_3d[z] = full_plane[y:y+height, x:x+width]
                        
                    # Record metadata
                    image_data.append({
                        'image_id': image_id, 
                        'sequence': seq_val,
                        'timepoint': actual_timepoint,
                        'z_slice': 'all',
                        'is_patch': True,
                        'patch_x': x,
                        'patch_y': y,
                        'patch_width': width,
                        'patch_height': height
                    })
                else:
                    # Process full 3D volume
                    img_3d = np.stack([pixels.getPlane(z, channel, actual_timepoint) 
                                     for z in range(image.getSizeZ())])
                    
                    # Record metadata
                    image_data.append({
                        'image_id': image_id, 
                        'sequence': seq_val,
                        'timepoint': actual_timepoint,
                        'z_slice': 'all',
                        'is_patch': False,
                        'patch_x': 0,
                        'patch_y': 0,
                        'patch_width': image.getSizeX(),
                        'patch_height': image.getSizeY()
                    })
                
                images.append(img_3d)
                print(f"Loaded 3D image/patch for image {image_id} with shape {img_3d.shape}")
                
            else:
                # 2D mode - determine which z-slice to use
                if z_slice_mode == "all":
                    # Use all z-slices (not yet supported in this function)
                    actual_z_slice = z_slices[0]  # Default to first z-slice for now
                    print("Warning: 'all' z-slice mode not fully supported yet, using first z-slice")
                elif z_slice_mode == "random":
                    # Select a random z-slice from the list
                    actual_z_slice = np.random.choice(z_slices)
                else:  # "specific"
                    # Use the first z-slice in the list
                    actual_z_slice = z_slices[0]
                
                pixels = image.getPrimaryPixels()
                
                if patch is not None:
                    # Extract 2D patch from the specified plane
                    x, y, width, height = patch
                    full_plane = pixels.getPlane(actual_z_slice, channel, actual_timepoint)
                    img = full_plane[y:y+height, x:x+width]
                    
                    # Record metadata
                    image_data.append({
                        'image_id': image_id, 
                        'sequence': seq_val,
                        'timepoint': actual_timepoint,
                        'z_slice': actual_z_slice,
                        'is_patch': True,
                        'patch_x': x,
                        'patch_y': y,
                        'patch_width': width,
                        'patch_height': height
                    })
                else:
                    # Get full 2D plane
                    img = pixels.getPlane(actual_z_slice, channel, actual_timepoint)
                    
                    # Record metadata
                    image_data.append({
                        'image_id': image_id, 
                        'sequence': seq_val,
                        'timepoint': actual_timepoint,
                        'z_slice': actual_z_slice,
                        'is_patch': False,
                        'patch_x': 0,
                        'patch_y': 0,
                        'patch_width': image.getSizeX(),
                        'patch_height': image.getSizeY()
                    })
                
                images.append(img)
                print(f"Loaded 2D image/patch for image {image_id} with shape {img.shape}")
        
        # Process batch with SAM using standard numpy arrays
        print("Starting napari viewer with SAM annotator. Close the viewer window when done.")
        
        # Create viewer without context management
        viewer = napari.Viewer()
        
        # Add image series annotator
        image_series_annotator(
            images, 
            model_type=model_type,
            viewer=viewer,
            embedding_path=os.path.join(output_folder, "embed"),
            output_folder=os.path.join(output_folder, "output"),
            is_volumetric=three_d
        )
        
        # Start the napari application - this blocks until the viewer is closed
        try:
            napari.run()
            print("Napari viewer closed.")
        except KeyboardInterrupt:
            print("Napari viewer was interrupted. Processing results anyway...")
        except Exception as e:
            print(f"Error in napari: {e}")
            
        print("Processing results from batch...")
        print("Done annotating batch, storing results in zarr and uploading to OMERO")
        
        # Initialize batch progress tracking
        batch_completed = 0
        batch_skipped = 0
        
        # Process results for batch
        batch_df = pd.DataFrame(columns=df.columns)
        
        for n, unit_data in enumerate(image_data):
            local_n = n  # Index within current batch
            global_n = start_idx + n  # Global index across all batches
            
            # Get the image object
            image = conn.getObject("Image", unit_data['image_id'])
            is_patch = unit_data['is_patch']
            patch_info = None
            
            if is_patch:
                patch_info = (unit_data['patch_x'], unit_data['patch_y'], 
                             unit_data['patch_width'], unit_data['patch_height'])
            
            # Store segmentation mask in zarr before uploading to OMERO
            seg_file_path = os.path.join(output_folder, "output", f"seg_{local_n:05d}.tif")
            if not os.path.exists(seg_file_path):
                print(f"Warning: Segmentation file not found for image {image.getId()}, skipping")
                batch_skipped += 1
                
                # Add a row for skipped image but mark as not processed
                is_train = unit_data['sequence'] == 0 if not segment_all else True
                is_validate = unit_data['sequence'] == 1 if not segment_all else False
                
                # Z-slice information
                z_info = 'all' if three_d else unit_data['z_slice']
                
                new_row = pd.DataFrame([{
                    "image_id": image.getId(),
                    "image_name": image.getName(),
                    "train": is_train,
                    "validate": is_validate,
                    "channel": channel,
                    "z_slice": z_info,
                    "timepoint": unit_data['timepoint'],
                    "sam_model": model_type,
                    "embed_id": None,
                    "label_id": None,
                    "roi_id": None,
                    "is_volumetric": three_d,
                    "processed": False,
                    "is_patch": is_patch,
                    "patch_x": unit_data.get('patch_x', 0),
                    "patch_y": unit_data.get('patch_y', 0),
                    "patch_width": unit_data.get('patch_width', 0),
                    "patch_height": unit_data.get('patch_height', 0)
                }])
                batch_df = pd.concat([batch_df, new_row], ignore_index=True)
                continue
                
            batch_completed += 1
            
            # Read the segmentation mask
            mask_data = imageio.imread(seg_file_path)
            
            # Store in zarr format for efficient processing
            zarr_file_path = store_annotations_in_zarr(mask_data, zarr_path, global_n)
            
            # Store embedding in zarr format and zip for OMERO upload
            embed_zarr = f"embedding_{local_n:05d}.zarr"
            embed_dir = os.path.join(output_folder, "embed")
            zip_path = os.path.join(embed_dir, f"embedding_{global_n:05d}.zip")
            
            # Check if the embedding directory exists before trying to zip it
            embed_zarr_path = os.path.join(embed_dir, embed_zarr)
            if not os.path.exists(embed_zarr_path):
                print(f"Warning: Embedding directory {embed_zarr} not found, skipping embedding upload")
                embed_id = None
            else:
                with zipfile.ZipFile(zip_path, 'w') as zip_file:
                    zip_directory(embed_dir, embed_zarr, zip_file)
                
                # Upload embedding to OMERO
                embed_id = ezomero.post_file_annotation(
                    conn,
                    str(zip_path),
                    ns='microsam.embeddings',
                    object_type="Image",
                    object_id=image.getId(),
                    description=f'SAM embedding ({model_type}), 3D={three_d}, Patch={is_patch}'
                )
            
            # Convert zarr annotation to TIFF for OMERO compatibility
            tiff_path = os.path.join(output_folder, "output", f"seg_{global_n:05d}.tiff")
            zarr_to_tiff(zarr_file_path, tiff_path)
            
            # For ROI creation, we need to handle patches differently
            if is_patch:
                # We need to create ROIs with the proper offset in the original image
                patch_x, patch_y = patch_info
            else:
                patch_x, patch_y = 0, 0
                
            # Upload labels and create ROIs - handle 3D and patches
            if three_d:
                # For 3D data, handle z-dimension correctly
                z_for_roi = range(image.getSizeZ())
                label_id, roi_id = upload_rois_and_labels(
                    conn, 
                    image, 
                    tiff_path, 
                    z_for_roi,
                    channel, 
                    unit_data['timepoint'], 
                    model_type,
                    is_volumetric=True,
                    patch_offset=(patch_x, patch_y) if is_patch else None,
                    read_only_mode=read_only_mode,
                    local_output_dir=local_output_dir
                )
            else:
                # For 2D data - with potential patch offset
                label_id, roi_id = upload_rois_and_labels(
                    conn, 
                    image, 
                    tiff_path, 
                    unit_data['z_slice'], 
                    channel, 
                    unit_data['timepoint'], 
                    model_type,
                    is_volumetric=False,
                    patch_offset=(patch_x, patch_y) if is_patch else None,
                    read_only_mode=read_only_mode,
                    local_output_dir=local_output_dir
                )
            
            # Update tracking dataframe
            is_train = unit_data['sequence'] == 0 if not segment_all else True
            is_validate = unit_data['sequence'] == 1 if not segment_all else False
            
            # Z-slice information
            z_info = 'all' if three_d else unit_data['z_slice']
            
            new_row = pd.DataFrame([{
                "image_id": image.getId(),
                "image_name": image.getName(),
                "train": is_train,
                "validate": is_validate,
                "channel": channel,
                "z_slice": z_info,
                "timepoint": unit_data['timepoint'],
                "sam_model": model_type,
                "embed_id": embed_id,
                "label_id": label_id,
                "roi_id": roi_id,
                "is_volumetric": three_d,
                "processed": True,
                "is_patch": is_patch,
                "patch_x": unit_data.get('patch_x', 0),
                "patch_y": unit_data.get('patch_y', 0),
                "patch_width": unit_data.get('patch_width', 0),
                "patch_height": unit_data.get('patch_height', 0)
            }])
            batch_df = pd.concat([batch_df, new_row], ignore_index=True)
        
        # Update the main DataFrame with the batch results
        df = pd.concat([df, batch_df], ignore_index=True)
        
        # Upload batch tracking table to OMERO
        if table_id is not None:
            # Delete the existing table before creating a new one
            try:
                print(f"Deleting existing table with ID: {table_id}")
                # Get the file annotation object for the table
                ann = conn.getObject("FileAnnotation", table_id)
                if ann:
                    # Delete the file annotation (which contains the table)
                    conn.deleteObjects("FileAnnotation", [table_id], wait=True)
                    print(f"Existing table deleted successfully")
                else:
                    print(f"Warning: Could not find table with ID: {table_id}")
            except Exception as e:
                print(f"Warning: Could not delete existing table: {e}")
                # Continue anyway, as we'll create a new table
        
        # Prepare DataFrame for OMERO table: Convert potentially None/NaN ID columns to string
        df_for_omero = df.copy()
        id_columns_to_convert = ['embed_id', 'label_id', 'roi_id', 'schema_attachment_id']
        for col in id_columns_to_convert:
            if col in df_for_omero.columns: # Ensure column exists
                # Convert to string, handling potential float NaNs first if necessary
                df_for_omero[col] = df_for_omero[col].astype(str)


        # Create a new table with the updated data
        table_id = ezomero.post_table(
            conn, 
            object_type=container_type.capitalize(), 
            object_id=container_id, 
            table=df_for_omero, # Use the converted DataFrame
            title="micro_sam_training_data"
        )
        if table_id is None:
            print("Warning: Failed to create tracking table")
        else:
            print(f"Created new tracking table with ID: {table_id}")
        
        print(f"Batch {batch_idx+1}/{total_batches} results:")
        print(f"  - Completed: {batch_completed}/{len(batch_units)} units")
        print(f"  - Skipped: {batch_skipped}/{len(batch_units)} units")
        
        if batch_skipped > 0 and batch_idx < total_batches - 1:
            # Ask user if they want to continue with next batch or stop here
            try:
                response = input("Some units were skipped. Continue with next batch? (y/n): ")
                if response.lower() not in ['y', 'yes']:
                    print("Stopping processing at user request.")
                    break
            except:
                # In case of non-interactive environment, continue by default
                print("Non-interactive environment detected. Continuing with next batch.")
        
        # Clean up temporary files for this batch
        for n in range(batch_size):  # Use local indexing for cleanup
            if start_idx + n >= len(processing_units):  # Skip if we've processed all units
                continue
                
            embed_zip = os.path.join(output_folder, "embed", f"embedding_{n:05d}.zip")
            embed_zarr = os.path.join(output_folder, "embed", f"embedding_{n:05d}.zarr")
            seg_file = os.path.join(output_folder, "output", f"seg_{n:05d}.tif")
            
            for path in [embed_zip, seg_file]:
                if os.path.exists(path):
                    os.remove(path)
                    
            if os.path.exists(embed_zarr) and os.path.isdir(embed_zarr):
                shutil.rmtree(embed_zarr)
    
    # Final statistics
    total_processed = df[df['processed'] == True].shape[0]
    total_skipped = df[df['processed'] == False].shape[0]
    
    print(f"\nAll batches completed.")
    print(f"Total processed: {total_processed} units")
    print(f"Total skipped: {total_skipped} units")
    print(f"Final tracking table ID: {table_id} in {source_desc}")
    
    return table_id, combined_images
