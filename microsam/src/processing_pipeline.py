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
from .omero_functions import (
    upload_rois_and_labels,
    initialize_tracking_table,
    update_tracking_table_rows,
    get_table_by_name,
    get_dask_dimensions,
    get_dask_image_multiple,
    get_annotation_configurations
)


def process_omero_batch(
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
    local_output_dir: str = "./omero_annotations",
    trainingset_name: str = None,
    group_by_image: bool = True
):
    """    Process OMERO images in batches for SAM segmentation using dask for lazy loading
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
        trainingset_name: Optional name for the training set (used in naming tables and annotations)
        group_by_image: Whether to keep all z-slices and timepoints from the same image together
                      in either training or validation set. When True, all slices and timepoints
                      from an image will either all be in training or all in validation set.
    
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
          # Table parameters
    table_title = f"micro_sam_{trainingset_name}" if trainingset_name else "micro_sam_training_data"
      # Check if we should resume from existing table or create a new one
    if resume_from_table:
        try:
            # Get existing tracking table using our helper function
            table_id, df = get_table_by_name(
                conn, 
                container_type.capitalize(), 
                container_id, 
                table_title
            )
            
            if table_id is not None and df is not None:
                # Store table metadata
                df.attrs['container_type'] = container_type
                df.attrs['container_id'] = container_id
                df.attrs['table_title'] = table_title
                
                print(f"Resuming from existing table ID: {table_id}")
                print(f"Found {len(df)} previously tracked images/patches")
            else:
                print(f"Table '{table_title}' not found, creating a new one")
                resume_from_table = False
                
        except Exception as e:
            print(f"Error retrieving existing table: {e}. Creating a new one.")
            resume_from_table = False
    
    # If not resuming or no table found, create a new complete tracking table
    if not resume_from_table:
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
            

            # Interleave training and validation sets
            combined_images, combined_images_sequence = interleave_arrays(train_images, validate_images)
            print(f"Interleaving training and validation images: {train_n} training images, {validate_n} validation images")
        
    # Initialize the complete tracking table with all planned processing units
        print("Creating complete tracking table with all planned images/patches...")
        table_id, df = initialize_tracking_table(
            conn, 
            combined_images, 
            container_type, 
            container_id, 
            segment_all=segment_all, 
            train_n=train_n, 
            validate_n=validate_n,
            use_patches=use_patches, 
            patch_size=patch_size, 
            patches_per_image=patches_per_image, 
            random_patches=random_patches,
            model_type=model_type,
            channel=channel,
            three_d=three_d,
            trainingset_name=trainingset_name
        )
        
        # Store annotation configuration in OMERO using the recommended namespace
        config = {
            "model_type": model_type,
            "channel": int(channel),
            "timepoints": timepoints,
            "timepoint_mode": timepoint_mode,
            "z_slices": z_slices,
            "z_slice_mode": z_slice_mode,
            "three_d": three_d,
            "use_patches": use_patches,
            "patch_size": patch_size if use_patches else None,
            "patches_per_image": patches_per_image if use_patches else None,
            "random_patches": random_patches if use_patches else None,
            "segment_all": segment_all,
            "train_n": train_n if not segment_all else None,
            "validate_n": validate_n if not segment_all else None,
            "group_by_image": group_by_image
        }
        
        
        # Add annotation to container with standard namespace
        schema_id = ezomero.post_map_annotation(
            conn,
            container_type.capitalize(),  # Format required by ezomero
            container_id,
            config,
            ns="openmicroscopy.org/omero/annotation",
        )
        print(f"Stored annotation configuration in OMERO with ID: {schema_id}")
    else:
        # When resuming, we need to rebuild combined_images and sequence arrays
        combined_images = []
        combined_images_sequence = []
        
        # Get unique images from the table
        unique_image_ids = df['image_id'].unique()
        for img_id in unique_image_ids:
            img = conn.getObject("Image", img_id)
            if img:
                rows = df[df['image_id'] == img_id]
                # Use train flag to determine sequence value
                seq_val = 0 if rows.iloc[0]['train'] else 1
                combined_images.append(img)
                combined_images_sequence.append(seq_val)
        
        combined_images_sequence = np.array(combined_images_sequence)
          # Build processing units from unprocessed rows in tracking table
    processing_units = []  # Will contain tuples of (image, sequence_val, [metadata], row_index)
    for idx, row in df.iterrows():
        if not row['processed']:  # Process any rows that aren't marked as processed
            img_id = int(row['image_id'])
            image = conn.getObject("Image", img_id)
            
            if image:
                seq_val = 0 if row['train'] else 1
                
                if row['is_patch']:
                    # Extract patch coordinates
                    patch = (
                        int(row['patch_x']), 
                        int(row['patch_y']), 
                        int(row['patch_width']), 
                        int(row['patch_height'])
                    )
                    processing_units.append((image, seq_val, patch, idx))
                else:
                    processing_units.append((image, seq_val, None, idx))
            else:
                print(f"Warning: Could not find image with ID {img_id}, skipping")
    
    if len(processing_units) == 0:
        print("No unprocessed images/patches found in tracking table.")
        return table_id, combined_images
    
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
        batch_units = processing_units[start_idx:end_idx]          # Load batch images for processing
        images = []
        image_data = []  # Store metadata about each image/patch
        
        for unit_idx, (image, seq_val, patch, row_idx) in enumerate(batch_units):
            image_id = image.getId()
              # Determine which timepoints to use
            if timepoint_mode == "all":
                # Use all timepoints
                actual_timepoints = list(range(image.getSizeT())) if image.getSizeT() > 0 else [0]
            elif timepoint_mode == "random":
                # Select a random timepoint from the list
                actual_timepoints = [np.random.choice(timepoints)]
            else:  # "specific"
                # Use all specified timepoints
                actual_timepoints = timepoints
              # Get primary pixels
            pixels = image.getPrimaryPixels()
            
            # Determine which z-slices to use
            if z_slice_mode == "all":
                # Use all z-slices
                actual_z_slices = list(range(image.getSizeZ())) if image.getSizeZ() > 0 else [0]
            elif z_slice_mode == "random":
                # Select a random z-slice from the list
                actual_z_slices = [np.random.choice(z_slices)]
            else:  # "specific"
                # Use all specified z-slices
                actual_z_slices = z_slices
            
            # For 3D mode or 2D with patches
            if three_d:
                # For each timepoint
                for t_idx, actual_timepoint in enumerate(actual_timepoints):
                    # 3D mode - process entire Z-stack or specified z-range
                    if patch is not None:
                        # Extract 3D patch (x, y, z-stack)
                        x, y, width, height = patch
                        img_3d = np.zeros((len(actual_z_slices), height, width), dtype=np.uint16)
                        
                        # Load each z-slice for the patch
                        for z_idx, z in enumerate(actual_z_slices):
                            full_plane = pixels.getPlane(z, channel, actual_timepoint)
                            img_3d[z_idx] = full_plane[y:y+height, x:x+width]
                            
                        # Record metadata
                        image_data.append({
                            'image_id': image_id, 
                            'sequence': seq_val,
                            'timepoint': actual_timepoint,
                            'z_slice': actual_z_slices,  # Store all z-slices
                            'is_patch': True,
                            'patch_x': x,
                            'patch_y': y,
                            'patch_width': width,
                            'patch_height': height
                        })
                    else:
                        # Process full 3D volume for the selected z-slices
                        img_3d = np.stack([pixels.getPlane(z, channel, actual_timepoint) 
                                        for z in actual_z_slices])
                        
                        # Record metadata
                        image_data.append({
                            'image_id': image_id, 
                            'sequence': seq_val,
                            'timepoint': actual_timepoint,
                            'z_slice': actual_z_slices,  # Store all z-slices
                            'is_patch': False,
                            'patch_x': 0,
                            'patch_y': 0,
                            'patch_width': image.getSizeX(),
                            'patch_height': image.getSizeY()
                        })
                    
                    images.append(img_3d)
                    print(f"Loaded 3D image/patch for image {image_id} at timepoint {actual_timepoint} with shape {img_3d.shape}")
            
            else:
                # 2D mode
                # Process each timepoint
                for t_idx, actual_timepoint in enumerate(actual_timepoints):
                    # Process each z-slice
                    for z_idx, actual_z_slice in enumerate(actual_z_slices):
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
                        print(f"Loaded 2D image/patch for image {image_id} at timepoint {actual_timepoint}, z-slice {actual_z_slice} with shape {img.shape}")
        
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
            #napari.run()
            get_settings().application.ipy_interactive = True
            viewer.show(block=True)
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
                
                # Z-slice information - convert list to string for storage in table if needed
                z_info = 'all' if three_d else unit_data['z_slice']
                if isinstance(z_info, list):
                    z_info = str(z_info)
                
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
                patch_x, patch_y, _, _ = patch_info
            else:
                patch_x, patch_y = 0, 0            # Upload labels and create ROIs - handle 3D and patches
            if three_d:
                # For 3D data, handle z-dimension correctly
                # Use the actual z-slices that were processed if available
                z_for_roi = unit_data['z_slice'] if isinstance(unit_data['z_slice'], list) else range(image.getSizeZ())
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
                    local_output_dir=local_output_dir,
                    trainingset_name=trainingset_name
                )
            else:
                # For 2D data - with potential patch offset
                # Pass the exact z-slice that was processed
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
                    local_output_dir=local_output_dir,
                    trainingset_name=trainingset_name
                )
            
            # Update tracking dataframe
            is_train = unit_data['sequence'] == 0 if not segment_all else True
            is_validate = unit_data['sequence'] == 1 if not segment_all else False
              # Z-slice information - convert lists to string for storage if needed
            z_info = 'all' if three_d else unit_data['z_slice']
            if isinstance(z_info, list):
                z_info = str(z_info)
            
            # Timepoint - also convert to string if needed
            t_info = unit_data['timepoint']
            if isinstance(t_info, list):
                t_info = str(t_info)
                
            new_row = pd.DataFrame([{
                "image_id": image.getId(),
                "image_name": image.getName(),
                "train": is_train,
                "validate": is_validate,
                "channel": channel,
                "z_slice": z_info,
                "timepoint": t_info,
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
          # Update the tracking table with batch results
        updated_indices = []
        updated_values = []
        
        for i, row in batch_df.iterrows():
            # Get the original index from processing_units
            orig_idx = processing_units[start_idx + i][3] if start_idx + i < len(processing_units) else None
            
            if orig_idx is not None:
                updated_indices.append(orig_idx)
                updated_values.append(row.to_dict())
            else:
                print(f"Warning: Could not find original index for batch row {i}")
        
        # Use our new function to update the table rows
        if updated_indices and table_id is not None:
            print(f"Updating tracking table (ID: {table_id}) with {len(updated_indices)} processed rows")
            table_id, df = update_tracking_table_rows(
                conn,
                table_id,
                df,
                updated_indices,
                updated_values
            )
        else:
            print("No updates to apply to tracking table")
        
        print(f"Batch {batch_idx+1}/{total_batches} results:")
        print(f"  - Completed: {batch_completed}/{len(batch_units)} units")
        print(f"  - Skipped: {batch_skipped}/{len(batch_units)} units")
        
        if batch_skipped > 0 and batch_idx < total_batches - 1:        # Ask user if they want to continue with next batch or stop here
            try:
                response = input("Some units were skipped. Continue with next batch? (y/n): ")
                if response.lower() not in ['y', 'yes']:
                    print("Stopping processing at user request.")
                    break
            except Exception as e:
                # In case of non-interactive environment, continue by default
                print(f"Non-interactive environment detected ({str(e)}). Continuing with next batch.")
        
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
                shutil.rmtree(embed_zarr)    # Final statistics
    if 'processed' in df.columns:
        processed_mask = df['processed'].astype(bool).values
        total_processed = df.loc[processed_mask].shape[0]
        total_skipped = df.loc[~processed_mask].shape[0]
    else:
        total_processed = 0
        total_skipped = 0
    
    print("\nAll batches completed.")
    print(f"Total processed: {total_processed} units")
    print(f"Total skipped: {total_skipped} units")
    print(f"Final tracking table ID: {table_id} in {source_desc}")
    
    return table_id, combined_images
