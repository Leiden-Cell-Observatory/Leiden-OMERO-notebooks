"""
Functions for working with OMERO server and data.
"""
import os
import numpy as np
import imageio.v3 as imageio
import pandas as pd
import json
import shutil
from .image_functions import label_to_rois
from .utils import NumpyEncoder


def print_object_details(conn, obj, datatype):
    """
    Print detailed information about OMERO objects
    
    Args:
        conn: OMERO connection
        obj: OMERO object (Project, Dataset, Image, etc.)
        datatype: Type of object ('project', 'dataset', 'image', etc.)
    """
    print(f"\n{datatype.capitalize()} Details:")
    print(f"- Name: {obj.getName()}")
    print(f"- ID: {obj.getId()}")
    print(f"- Owner: {obj.getOwner().getFullName()}")
    print(f"- Group: {obj.getDetails().getGroup().getName()}")
    
    if datatype == "project":
        datasets = list(obj.listChildren())
        dataset_count = len(datasets)
        total_images = sum(len(list(ds.listChildren())) for ds in datasets)
        print(f"- Number of datasets: {dataset_count}")
        print(f"- Total images: {total_images}")
        
    elif datatype == "plate":
        wells = list(obj.listChildren())
        well_count = len(wells)
        print(f"- Number of wells: {well_count}")
        
    elif datatype == "dataset":
        images = list(obj.listChildren())
        image_count = len(images)
        # Get project info if dataset is in a project
        projects = obj.getParent()
        if projects:
            print(f"- Project: {projects.getName()} (ID: {projects.getId()})")
        else:
            print("- Project: None (orphaned dataset)")
        print(f"- Number of images: {image_count}")
        
    elif datatype == "image":
        size_x = obj.getSizeX()
        size_y = obj.getSizeY()
        size_z = obj.getSizeZ()
        size_c = obj.getSizeC()
        size_t = obj.getSizeT()
        # Get dataset info if image is in a dataset
        datasets = obj.getParent()
        if datasets:
            print(f"- Dataset: {datasets.getName()} (ID: {datasets.getId()})")
            # Get project info if dataset is in a project
            projects = datasets.getParent()
            if projects:
                print(f"- Project: {projects.getName()} (ID: {projects.getId()})")
        else:
            print("- Dataset: None (orphaned image)")
        print(f"- Dimensions: {size_x}x{size_y}")
        print(f"- Z-stack: {size_z}")
        print(f"- Channels: {size_c}")
        print(f"- Timepoints: {size_t}")


def get_images_from_container(conn, datatype, container_id):
    """
    Extract all images from a given OMERO container (Project, Dataset, Plate, Screen)
    
    Args:
        conn: OMERO connection
        datatype: Type of container ('project', 'dataset', 'plate', 'screen', 'image')
        container_id: ID of the container
        
    Returns:
        list: List of OMERO image objects
        str: Description of the source (for tracking)
    """
    images = []
    source_desc = ""
    
    if datatype == "image":
        image = conn.getObject("Image", container_id)
        if image is None:
            raise ValueError(f"Image with ID {container_id} not found")
        images = [image]
        source_desc = f"Image: {image.getName()} (ID: {container_id})"
    
    elif datatype == "dataset":
        dataset = conn.getObject("Dataset", container_id)
        if dataset is None:
            raise ValueError(f"Dataset with ID {container_id} not found")
        images = list(dataset.listChildren())
        source_desc = f"Dataset: {dataset.getName()} (ID: {container_id})"
    
    elif datatype == "project":
        project = conn.getObject("Project", container_id)
        if project is None:
            raise ValueError(f"Project with ID {container_id} not found")
        # Get all datasets in the project
        for dataset in project.listChildren():
            # Get all images in each dataset
            for image in dataset.listChildren():
                images.append(image)
        source_desc = f"Project: {project.getName()} (ID: {container_id})"
    
    elif datatype == "plate":
        plate = conn.getObject("Plate", container_id)
        if plate is None:
            raise ValueError(f"Plate with ID {container_id} not found")
        # Get all wells in the plate
        for well in plate.listChildren():
            # Get all images (fields) in each well
            for wellSample in well.listChildren():
                images.append(wellSample.getImage())
        source_desc = f"Plate: {plate.getName()} (ID: {container_id})"
    
    elif datatype == "screen":
        screen = conn.getObject("Screen", container_id)
        if screen is None:
            raise ValueError(f"Screen with ID {container_id} not found")
        # Get all plates in the screen
        for plate in screen.listChildren():
            # Get all wells in each plate
            for well in plate.listChildren():
                # Get all images (fields) in each well
                for wellSample in well.listChildren():
                    images.append(wellSample.getImage())
        source_desc = f"Screen: {screen.getName()} (ID: {container_id})"
    
    else:
        raise ValueError(f"Unsupported datatype: {datatype}")
    
    print(f"Found {len(images)} images from {source_desc}")
    return images, source_desc


def get_dask_image(conn, image_id, z_slice=None, timepoint=None, channel=None, three_d=False, patch_coords=None):
    """
    Get a dask array representation of an OMERO image for lazy loading
    
    Args:
        conn: OMERO connection
        image_id: ID of image to load
        z_slice: Optional specific Z slice to load (int or list)
        timepoint: Optional specific timepoint to load (int or list)
        channel: Optional specific channel to load (int or list)
        three_d: Whether to load a 3D volume (all z-slices) instead of a single slice
        patch_coords: Optional tuple of (x, y, width, height) to extract a patch
    
    Returns:
        dask array representation of image
    """
    import dask
    import dask.array as da
    
    image = conn.getObject("Image", image_id)
    pixels = image.getPrimaryPixels()
    
    # Get image dimensions
    size_z = image.getSizeZ()
    size_c = image.getSizeC()
    size_t = image.getSizeT()
    size_y = image.getSizeY()
    size_x = image.getSizeX()
    
    # Define specific dimensions to load if provided
    # If three_d is True, we want all z-slices, otherwise use the provided z_slice
    if three_d:
        z_range = range(size_z)  # Load all z-slices for 3D
    else:
        z_range = [z_slice] if isinstance(z_slice, int) else (range(size_z) if z_slice is None else z_slice)
    
    t_range = [timepoint] if isinstance(timepoint, int) else (range(size_t) if timepoint is None else timepoint)
    c_range = [channel] if isinstance(channel, int) else (range(size_c) if channel is None else channel)
    
    # Extract patch information if provided
    x_offset = 0
    y_offset = 0
    if patch_coords:
        x_offset, y_offset, patch_width, patch_height = patch_coords
        size_x = patch_width
        size_y = patch_height
    
    # Create empty dict to store delayed objects
    delayed_planes = {}
    
    desc = "patch" if patch_coords else "image"
    print(f"Creating dask array for {desc} {image_id} with lazy loading")
    print(f"Dimensions: Z={len(z_range)}, C={len(c_range)}, T={len(t_range)}, Y={size_y}, X={size_x}")
    print(f"3D mode: {three_d}")
    
    # Create lazy loading function
    @dask.delayed
    def get_plane(z, c, t):
        print(f"Loading plane: Z={z}, C={c}, T={t}")
        if patch_coords:
            full_plane = pixels.getPlane(z, c, t)
            return full_plane[y_offset:y_offset+size_y, x_offset:x_offset+size_x]
        else:
            return pixels.getPlane(z, c, t)
    
    # Build dask arrays
    arrays = []
    for t in t_range:
        t_arrays = []
        for z in z_range:
            z_arrays = []
            for c in c_range:
                # Create a key for this plane
                key = (z, c, t)
                
                # Check if we've already created this delayed object
                if key not in delayed_planes:
                    # Create a delayed object for this plane
                    delayed_plane = get_plane(z, c, t)
                    delayed_planes[key] = delayed_plane
                else:
                    delayed_plane = delayed_planes[key]
                
                # Convert to dask array with known shape and dtype
                shape = (size_y, size_x)
                dtype = np.uint16  # Most OMERO images use 16-bit
                dask_plane = da.from_delayed(delayed_plane, shape=shape, dtype=dtype)
                z_arrays.append(dask_plane)
            if z_arrays:
                # Stack channels for this z position
                t_arrays.append(da.stack(z_arrays))
        if t_arrays:
            # Stack z-planes for this timepoint
            arrays.append(da.stack(t_arrays))
    
    if arrays:
        # Stack all timepoints
        return da.stack(arrays)
    else:
        return None


def upload_rois_and_labels(conn, image, label_file, z_slice, channel, timepoint, model_type, 
                          is_volumetric=False, patch_offset=None, read_only_mode=False, local_output_dir="./omero_annotations",
                          trainingset_name=None):
    """
    Upload both label map and ROIs for a segmented image or save them locally in read-only mode
    
    Args:
        conn: OMERO connection
        image: OMERO image object
        label_file: Path to the label image file
        z_slice: Z-slice index or range of indices
        channel: Channel index
        timepoint: Time point index
        model_type: SAM model type used
        is_volumetric: Whether the data is 3D volumetric
        patch_offset: Optional (x,y) offset for placing ROIs in a larger image
        read_only_mode: If True, save annotations locally instead of uploading to OMERO
        local_output_dir: Directory to save local annotations when in read-only mode
        trainingset_name: Optional name for the training set (used in naming ROIs and annotations)
    
    Returns:
        tuple: (label_id, roi_id) or (local_label_path, local_roi_path) in read-only mode
    """
    import ezomero
    
    # Add patch info to description if applicable
    patch_desc = ""
    if patch_offset:
        patch_desc = f", Patch offset: ({patch_offset[0]}, {patch_offset[1]})"
    
    # Create ROIs from label image
    label_img = imageio.imread(label_file)
    shapes = label_to_rois(label_img, z_slice, channel, timepoint, model_type, 
                          is_volumetric, patch_offset)
    
    if read_only_mode:
        # Save annotations locally instead of uploading to OMERO
        
        # Create local directories
        image_id = image.getId()
        
        # Include trainingset_name in directory structure if provided
        if trainingset_name:
            image_dir = os.path.join(local_output_dir, trainingset_name, f"image_{image_id}")
        else:
            image_dir = os.path.join(local_output_dir, f"image_{image_id}")
            
        os.makedirs(image_dir, exist_ok=True)
        
        # Save label image file
        local_label_path = os.path.join(image_dir, os.path.basename(label_file))
        shutil.copy2(label_file, local_label_path)
        
        # Save ROI data as JSON
        local_roi_path = os.path.join(image_dir, f"roi_{os.path.basename(label_file).split('.')[0]}.json")
        
        # Prepare ROI metadata
        roi_metadata = {
            "image_id": image_id,
            "image_name": image.getName(),
            "timestamp": str(pd.Timestamp.now()),
            "model_type": model_type,
            "is_volumetric": is_volumetric,
            "z_slice": z_slice if not isinstance(z_slice, range) else list(z_slice),
            "channel": channel,
            "timepoint": timepoint,
            "patch_offset": patch_offset,
            "shapes_count": len(shapes) if shapes else 0,
            # We can't store the actual shapes because they're OMERO objects
            # but we can save the label image which can be used to recreate them
            "label_image_path": os.path.relpath(local_label_path, local_output_dir)
        }
        
        # Save metadata
        with open(local_roi_path, 'w') as f:
            json.dump(roi_metadata, f, indent=2, cls=NumpyEncoder)
            
        print(f"Saved annotation locally in read-only mode to {image_dir}")
        return local_label_path, local_roi_path
    else:
        # Normal OMERO upload mode
        # Create label name with trainingset_name if provided
        label_desc = f'SAM {"volumetric" if is_volumetric else "manual"} segmentation ({model_type}){patch_desc}'
        if trainingset_name:
            label_desc = f'{trainingset_name} - {label_desc}'
            
        # Upload label map as attachment
        label_id = ezomero.post_file_annotation(
            conn,
            str(label_file),
            ns='microsam.labelimage',
            object_type="Image",
            object_id=image.getId(),
            description=label_desc
        )
        
        if shapes:  # Only create ROI if shapes were found
            # Create ROI name with trainingset_name if provided
            roi_name = f'SAM_{model_type}{"_3D" if is_volumetric else ""}{patch_desc}'
            roi_desc = f'micro_sam.{"volumetric" if is_volumetric else "manual"}_instance_segmentation.{model_type}{patch_desc}'
            
            if trainingset_name:
                roi_name = f'{trainingset_name}_{roi_name}'
                roi_desc = f'{trainingset_name} - {roi_desc}'
                
            roi_id = ezomero.post_roi(
                conn,
                image.getId(),
                shapes,
                name=roi_name,
                description=roi_desc
            )
        else:
            roi_id = None
            
        return label_id, roi_id


def initialize_tracking_table(
    conn, 
    images_list, 
    container_type, 
    container_id, 
    segment_all=True, 
    train_n=3, 
    validate_n=3,
    use_patches=False, 
    patch_size=(512, 512), 
    patches_per_image=1, 
    random_patches=True,
    model_type=None,
    channel=None,
    three_d=False,
    trainingset_name=None
):
    """
    Initialize a complete tracking table with rows for all images/patches that will be processed.
    All rows will be marked as 'processed=False' initially.
    
    Args:
        conn: OMERO connection
        images_list: List of OMERO image objects
        container_type: Type of OMERO container ('dataset', 'plate', etc.)
        container_id: ID of the container
        segment_all: Whether to include all images in training set
        train_n: Number of training images if not segment_all
        validate_n: Number of validation images if not segment_all
        use_patches: Whether to extract patches instead of using full images
        patch_size: Size of patches to extract (width, height)
        patches_per_image: Number of patches to extract per image
        random_patches: Whether to extract patches randomly or from center
        model_type: SAM model type to use (for documentation in table)
        channel: Channel to segment (for documentation in table)
        three_d: Whether to use 3D volumetric mode
        trainingset_name: Optional name for the training set
        
    Returns:
        tuple: (table_id, df) - ID of created table and corresponding DataFrame
    """
    import ezomero
    from .image_functions import generate_patch_coordinates
    from .utils import interleave_arrays
    import numpy as np
    import pandas as pd
    
    # Create DataFrame to store tracking info
    df = pd.DataFrame(columns=[
        "image_id", "image_name", "train", "validate", 
        "channel", "z_slice", "timepoint", "sam_model", "embed_id", "label_id", "roi_id", 
        "is_volumetric", "processed", "is_patch", "patch_x", "patch_y", "patch_width", "patch_height",
        "schema_attachment_id"
    ])
    
    # Determine which images to include based on segment_all flag
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
    
    # Create rows for each image/patch
    for i, img in enumerate(combined_images):
        img_id = img.getId()
        seq_val = combined_images_sequence[i]
        is_train = seq_val == 0 if not segment_all else True
        is_validate = seq_val == 1 if not segment_all else False
        
        if use_patches:
            # Generate patches for this image
            size_x = img.getSizeX()
            size_y = img.getSizeY()
            patches = generate_patch_coordinates(
                size_x, size_y, patch_size, patches_per_image, random_patches)
            
            for patch in patches:
                x, y, width, height = patch
                new_row = pd.DataFrame([{
                    "image_id": img_id,
                    "image_name": img.getName(),
                    "train": is_train,
                    "validate": is_validate,
                    "channel": channel,  # Preset if provided
                    "z_slice": None,  # Will be set during processing
                    "timepoint": None,  # Will be set during processing
                    "sam_model": model_type,  # Preset if provided
                    "embed_id": None,
                    "label_id": None,
                    "roi_id": None,
                    "is_volumetric": three_d,  # Preset based on parameter
                    "processed": False,
                    "is_patch": True,
                    "patch_x": x,
                    "patch_y": y,
                    "patch_width": width,
                    "patch_height": height,
                    "schema_attachment_id": None
                }])
                df = pd.concat([df, new_row], ignore_index=True)
        else:
            # Create row for full image
            new_row = pd.DataFrame([{
                "image_id": img_id,
                "image_name": img.getName(),
                "train": is_train,
                "validate": is_validate,
                "channel": channel,  # Preset if provided
                "z_slice": None,  # Will be set during processing
                "timepoint": None,  # Will be set during processing
                "sam_model": model_type,  # Preset if provided
                "embed_id": None,
                "label_id": None,
                "roi_id": None,
                "is_volumetric": three_d,  # Preset based on parameter
                "processed": False,
                "is_patch": False,
                "patch_x": 0,
                "patch_y": 0,
                "patch_width": img.getSizeX(),
                "patch_height": img.getSizeY(),
                "schema_attachment_id": None
            }])
            df = pd.concat([df, new_row], ignore_index=True)
      # Store container info in the DataFrame for reference
    df.attrs['container_type'] = container_type
    df.attrs['container_id'] = container_id
      # Generate and store the table title - make sure it's saved in the DataFrame attributes
    # This is critical for the update_tracking_table_rows function to work properly
    table_title = f"micro_sam_{trainingset_name}" if trainingset_name else "micro_sam_training_data"
    df.attrs['table_title'] = table_title
    print(f"Using table title: {table_title}")
    
    # Prepare DataFrame for OMERO table: Convert potentially None/NaN ID columns to string
    df_for_omero = df.copy()
    id_columns = ['embed_id', 'label_id', 'roi_id', 'schema_attachment_id']
    for col in id_columns:
        df_for_omero[col] = df_for_omero[col].astype(str)
    
    # Create the table
    table_id = ezomero.post_table(
        conn,
        object_type=container_type.capitalize(),
        object_id=container_id,
        table=df_for_omero,
        title=table_title
    )
    
    print(f"Created tracking table with {len(df)} rows, ID: {table_id}")
    
    return table_id, df


def update_tracking_table_rows(conn, table_id, df, updated_indices, updated_values):
    """
    Update specific rows in an OMERO table with new values.
    Simplified implementation: updates DataFrame locally then recreates the table.
    
    Args:
        conn: OMERO connection
        table_id: ID of the table to update
        df: Current DataFrame of the complete table
        updated_indices: List of row indices to update
        updated_values: List of dictionaries with column values to update
        
    Returns:
        tuple: (new_table_id, updated_df) - ID of the updated table and updated DataFrame
    """
    import ezomero
    
    # Update the rows in our DataFrame
    for idx, row_data in zip(updated_indices, updated_values):
        if idx >= len(df):
            print(f"Warning: Index {idx} out of bounds for DataFrame (length {len(df)})")
            continue
            
        for col, val in row_data.items():
            if col in df.columns:
                df.at[idx, col] = val
            else:
                print(f"Warning: Column '{col}' not found in DataFrame")
    
    # Get container info from DataFrame attrs
    container_type = df.attrs.get('container_type', 'Dataset')
    container_id = df.attrs.get('container_id', None)
    
    if container_id is None:
        # Try to determine from the existing table data
        image_id = df.iloc[0].get('image_id') if len(df) > 0 else None
        if image_id:
            print(f"Using first image ID ({image_id}) to determine container")
            obj = conn.getObject("Image", image_id)
            if obj:
                parents = list(obj.listParents())
                if parents:
                    parent = parents[0]
                    container_type = parent.__class__.__name__
                    container_id = parent.getId()
        
        if container_id is None:
            print("Could not determine container ID, cannot recreate table")
            return table_id, df
    
    # Get table title (use the stored title or default)
    table_title = df.attrs.get('table_title', "micro_sam_training_data")
      # Prepare DataFrame for OMERO: Handle all columns properly
    df_for_omero = df.copy()
    
    # First ensure numeric columns have proper types
    numeric_columns = ['image_id', 'patch_x', 'patch_y', 'patch_width', 'patch_height']
    for col in numeric_columns:
        if col in df_for_omero.columns:
            df_for_omero[col] = pd.to_numeric(df_for_omero[col], errors='ignore')
    
    # Then convert ID columns to string, handling None values properly
    id_columns = ['embed_id', 'label_id', 'roi_id', 'schema_attachment_id']
    for col in id_columns:
        if col in df_for_omero.columns:
            # First convert valid numbers to integers where possible
            for idx in range(len(df_for_omero)):
                val = df_for_omero.at[idx, col]
                if pd.notna(val) and val not in ('', 'None', 'nan'):
                    try:
                        df_for_omero.at[idx, col] = int(float(val))
                    except (ValueError, TypeError):
                        pass  # Keep the original value
            
            # Then convert everything to strings for OMERO
            df_for_omero[col] = df_for_omero[col].fillna('None')
            df_for_omero[col] = df_for_omero[col].astype(str)
    
    # Try to delete the existing table
    try:
        conn.deleteObjects("FileAnnotation", [table_id], wait=True)
        print(f"Deleted existing table with ID: {table_id}")
    except Exception as e:
        print(f"Warning: Could not delete existing table: {e}")
    
    # Create a new table with the updated data
    try:
        new_table_id = ezomero.post_table(
            conn,
            object_type=container_type.capitalize(),
            object_id=container_id,
            table=df_for_omero,
            title=table_title
        )
        
        print(f"Created updated table with ID: {new_table_id}")
        return new_table_id, df
        
    except Exception as e:
        print(f"Error creating updated table: {e}")
        return table_id, df  # Return original values on error
