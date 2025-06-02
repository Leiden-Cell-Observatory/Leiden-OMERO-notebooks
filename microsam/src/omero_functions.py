"""
Functions for working with OMERO server and data.
"""
import os
import numpy as np
import imageio.v3 as imageio
import pandas as pd
import json
import shutil
import ezomero
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


def get_dask_image_multiple(conn, image_id, z_slices=None, timepoints=None, channel=None, three_d=False, patch_coords=None):
    """
    Enhanced version of get_dask_image that properly handles multiple z-slices and timepoints
    
    Args:
        conn: OMERO connection
        image_id: ID of image to load
        z_slices: List of Z-slice indices to load
        timepoints: List of timepoint indices to load
        channel: Optional specific channel to load (int or list)
        three_d: Whether to load a 3D volume (all z-slices) instead of a single slice
        patch_coords: Optional tuple of (x, y, width, height) to extract a patch
    
    Returns:
        dask array representation of image with shape (T, Z, C, Y, X) or reduced dimensions as appropriate
    """
    import dask
    import dask.array as da
    
    image = conn.getObject("Image", image_id)
    if not image:
        return None
        
    pixels = image.getPrimaryPixels()
    
    # Get image dimensions
    size_z = image.getSizeZ()
    size_c = image.getSizeC()
    size_t = image.getSizeT()
    size_y = image.getSizeY()
    size_x = image.getSizeX()
    
    # Define specific dimensions to load if provided
    # If three_d is True, we want all z-slices, otherwise use the provided z_slices
    if three_d:
        z_range = range(size_z)  # Load all z-slices for 3D
    else:
        if z_slices is None:
            z_range = [0]  # Default to first slice if none specified
        elif isinstance(z_slices, (list, tuple)):
            z_range = z_slices
        else:
            z_range = [z_slices]  # Convert single value to list
    
    # Handle timepoints similarly
    if timepoints is None:
        t_range = [0]  # Default to first timepoint if none specified
    elif isinstance(timepoints, (list, tuple)):
        t_range = timepoints
    else:
        t_range = [timepoints]  # Convert single value to list
    
    # Handle channels
    if channel is None:
        c_range = range(size_c)
    elif isinstance(channel, (list, tuple)):
        c_range = channel
    else:
        c_range = [channel]
    
    # Extract patch information if provided
    x_offset = 0
    y_offset = 0
    if patch_coords:
        x_offset, y_offset, patch_width, patch_height = patch_coords
        size_x = patch_width
        size_y = patch_height
    
    # Create empty dict to store delayed objects
    delayed_planes = {}
    
    # Describe what we're loading
    desc_type = "3D volume" if three_d else "image"
    patch_desc = " patch" if patch_coords else ""
    print(f"Creating dask array for {desc_type}{patch_desc} {image_id} with lazy loading")
    print(f"Dimensions: T={len(t_range)}, Z={len(z_range)}, C={len(c_range)}, Y={size_y}, X={size_x}")
    
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
    t_arrays = []
    for t in t_range:
        z_arrays = []
        for z in z_range:
            c_arrays = []
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
                c_arrays.append(dask_plane)
                
            if c_arrays:
                # Stack channels for this z position
                z_arrays.append(da.stack(c_arrays))
        
        if z_arrays:
            # Stack z-slices for this timepoint
            t_arrays.append(da.stack(z_arrays))
    
    if t_arrays:
        # Stack all timepoints
        return da.stack(t_arrays)
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
        
        # Save metadata using the standard OMERO namespace
        roi_metadata["namespace"] = "openmicroscopy.org/omero/annotation"
        
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
                
            roi = ezomero.post_roi(
                conn,
                image.getId(),
                shapes,
                name=roi_name,
                description=roi_desc
            )
            roi_id = roi.getId().getValue()
            
            # Add configuration details as map annotation with standard namespace
            config = {
                "model_type": model_type,
                "channel": str(channel),
                "timepoint": str(timepoint),
                "z_slice": str(z_slice) if not isinstance(z_slice, (list, range)) else str(list(z_slice)),
                "is_volumetric": str(is_volumetric),
                "has_patch_offset": str(patch_offset is not None)
            }
              # Add annotation with standard namespace
            ezomero.post_map_annotation(
                conn,
                object_type="Roi", 
                object_id=roi_id,
                kv_dict=config,
                ns="openmicroscopy.org/omero/annotation"
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
                    "channel": int(channel) if channel is not None else -1,  # Ensure integer type
                    "z_slice": -1,  # Using -1 as placeholder instead of None for consistent int type
                    "timepoint": -1,  # Using -1 as placeholder instead of None for consistent int type
                    "sam_model": model_type if model_type is not None else "",  # String type
                    "embed_id": -1,  # Using -1 as placeholder for numeric IDs
                    "label_id": -1,  # Using -1 as placeholder for numeric IDs
                    "roi_id": -1,  # Using -1 as placeholder for numeric IDs
                    "is_volumetric": three_d,  # Boolean type
                    "processed": False,  # Boolean type
                    "is_patch": True,  # Boolean type
                    "patch_x": x,  # Integer type
                    "patch_y": y,  # Integer type
                    "patch_width": width,  # Integer type
                    "patch_height": height,  # Integer type
                    "schema_attachment_id": -1  # Using -1 as placeholder for numeric IDs
                }])
                df = pd.concat([df, new_row], ignore_index=True)
        else:
            # Create row for full image with consistent types
            new_row = pd.DataFrame([{
                "image_id": img_id,
                "image_name": img.getName(),
                "train": is_train,
                "validate": is_validate,
                "channel": int(channel) if channel is not None else -1,  # Ensure integer type
                "z_slice": -1,  # Using -1 as placeholder instead of None for consistent int type
                "timepoint": -1,  # Using -1 as placeholder instead of None for consistent int type
                "sam_model": model_type if model_type is not None else "",  # String type
                "embed_id": -1,  # Using -1 as placeholder for numeric IDs
                "label_id": -1,  # Using -1 as placeholder for numeric IDs
                "roi_id": -1,  # Using -1 as placeholder for numeric IDs
                "is_volumetric": three_d,  # Boolean type
                "processed": False,  # Boolean type
                "is_patch": False,  # Boolean type 
                "patch_x": 0,  # Integer type
                "patch_y": 0,  # Integer type
                "patch_width": img.getSizeX(),  # Integer type
                "patch_height": img.getSizeY(),  # Integer type
                "schema_attachment_id": -1  # Using -1 as placeholder for numeric IDs
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
    
    # Prepare DataFrame for OMERO table: ensure consistent typing
    df_for_omero = df.copy()
    
    # First ensure numeric columns have proper and consistent types
    numeric_columns = ['image_id', 'patch_x', 'patch_y', 'patch_width', 'patch_height', 'z_slice', 'timepoint']
    for col in numeric_columns:
        if col in df_for_omero.columns:
            # Convert to integer type explicitly to ensure consistent typing
            try:
                # Convert to numeric with coercion
                numeric_series = pd.to_numeric(df_for_omero[col], errors='coerce')
                # Then fill any NAs and convert to integers
                df_for_omero[col] = numeric_series.fillna(-1).astype(int)
            except Exception:
                print(f"Warning: Could not convert column '{col}' to numeric. Setting to -1.")
                df_for_omero[col] = -1
    
    # Boolean columns need to be boolean type
    boolean_columns = ['train', 'validate', 'processed', 'is_patch', 'is_volumetric']
    for col in boolean_columns:
        if col in df_for_omero.columns:
            df_for_omero[col] = df_for_omero[col].fillna(False).astype(bool)
            
    # Then convert ID columns to string, handling None values properly
    id_columns = ['embed_id', 'label_id', 'roi_id', 'schema_attachment_id']
    for col in id_columns:
        if col in df_for_omero.columns:
            # Replace NaN/None with 'None' string then convert all to string
            df_for_omero[col] = df_for_omero[col].fillna('None').astype(str)
    
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
    import pandas as pd
    
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
      
    # First ensure numeric columns have proper and consistent types
    numeric_columns = ['image_id', 'patch_x', 'patch_y', 'patch_width', 'patch_height', 'z_slice', 'timepoint']
    for col in numeric_columns:
        if col in df_for_omero.columns:
            # Convert to integer type explicitly to ensure consistent typing
            try:
                # Convert to numeric with coercion
                numeric_series = pd.to_numeric(df_for_omero[col], errors='coerce')
                # Then fill any NAs and convert to integers
                df_for_omero[col] = numeric_series.fillna(-1).astype(int)
            except Exception:
                print(f"Warning: Could not convert column '{col}' to numeric. Setting to -1.")
                df_for_omero[col] = -1
    
    # Boolean columns need to be boolean type
    boolean_columns = ['train', 'validate', 'processed', 'is_patch', 'is_volumetric']
    for col in boolean_columns:
        if col in df_for_omero.columns:
            df_for_omero[col] = df_for_omero[col].fillna(False).astype(bool)
    
    # Then convert ID columns to string, handling None values properly
    id_columns = ['embed_id', 'label_id', 'roi_id', 'schema_attachment_id']
    for col in id_columns:
        if col in df_for_omero.columns:
            # Replace NaN/None with 'None' string then convert all to string
            df_for_omero[col] = df_for_omero[col].fillna('None').astype(str)
    
    # Try to delete the existing table
    try:
        print(f"Attempting to delete existing table with ID: {table_id}")
        conn.deleteObjects("FileAnnotation", [table_id], wait=True)
        print(f"Deleted existing table with ID: {table_id}")
    except Exception as e:
        print(f"Warning: Could not delete existing table: {e}")
        # If we can't delete the table, try with a new title to avoid conflicts
        if "cannot read all the specified objects" in str(e):
            table_title = f"{table_title}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Using alternative table title: {table_title}")
    
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


def get_table_by_name(conn, obj_type, obj_id, table_title):
    """
    Get table annotation attached to an object by name.
    
    Parameters:
        conn: OMERO connection
        obj_type: Type of object ('Dataset', 'Project', etc.)
        obj_id: ID of object
        table_title: Name of the table to find
        
    Returns:
        tuple: (table_id, table_df) or (None, None) if not found
    """
    import ezomero
    import pandas as pd
    
    obj = conn.getObject(obj_type, obj_id)
    if not obj:
        print(f"Object {obj_type} with ID {obj_id} not found")
        return None, None
    
    # Get all file annotations
    file_ann_ids = ezomero.get_file_annotation_ids(conn, obj_type, obj_id)
    print(f"Found {len(file_ann_ids)} file annotations on {obj_type} {obj_id}")
    
    # Check each annotation to see if it's a table with the right name
    for ann_id in file_ann_ids:
        try:
            # Get the actual annotation object
            ann = conn.getObject("FileAnnotation", ann_id)
            if ann is None:
                continue
                
            filename = ann.getFileName()
                            
            # Check if the filename contains our table title
            if table_title in filename:
                # Try to open it as a table
                try:
                    table_df = ezomero.get_table(conn, ann_id)
                    if isinstance(table_df, pd.DataFrame):
                        print(f"Found matching table: {filename} (ID: {ann_id})")
                        return ann_id, table_df
                except Exception as e:
                    print(f"Could not read table from annotation {ann_id}: {e}")
        except Exception as e:
            print(f"Error processing annotation {ann_id}: {e}")
    
    print(f"No table found with title '{table_title}'")
    return None, None


def get_image_dimensions(images):    
    """
    Determine the maximum Z, T, and C dimensions across all selected images
    
    Args:
        images: List of OMERO image objects
        
    Returns:
        tuple: (max_z, max_t, max_c) - maximum dimensions across all images
    """
    max_z = 0
    max_t = 0
    max_c = 0
    
    for image in images:
        size_z = image.getSizeZ()
        size_t = image.getSizeT()
        size_c = image.getSizeC()
        
        max_z = max(max_z, size_z)
        max_t = max(max_t, size_t)
        max_c = max(max_c, size_c)
    
    print(f"Analyzed {len(images)} images:")
    print(f"  Max Z-slices: {max_z}")
    print(f"  Max timepoints: {max_t}")
    print(f"  Max channels: {max_c}")
    
    return max_z, max_t, max_c


def load_configs_from_omero(conn, datatype, object_id):
    """
    Load previously saved configurations from OMERO map annotations
    
    Args:
        conn: OMERO connection
        datatype: Type of OMERO object ('Dataset', 'Project', etc.)
        object_id: ID of the OMERO object
        
    Returns:
        list: List of dictionaries containing configuration data
    """
    try:
        # Get all map annotations for the object
        map_ann_ids = ezomero.get_map_annotation_ids(
            conn, 
            object_type=datatype, 
            object_id=object_id,
            ns="openmicroscopy.org/omero/annotation"
        )
        
        configs = []
        for ann_id in map_ann_ids:
            try:
                # Get the map annotation data
                kv_dict = ezomero.get_map_annotation(conn, ann_id)
                
                # Check if this is a microsam configuration
                if kv_dict.get('config_type') == 'microsam_annotation_settings':
                    config_name = kv_dict.get('config_name', f'Config_{ann_id}')
                    created_at = kv_dict.get('created_at', 'Unknown')
                    
                    # Extract the actual configuration (exclude metadata keys)
                    metadata_keys = {'config_name', 'created_at', 'config_type'}
                    config_data = {k: v for k, v in kv_dict.items() if k not in metadata_keys}
                    
                    # Convert string values back to appropriate types
                    processed_config = {}
                    for key, value in config_data.items():
                        if key.endswith('_list') or ',' in value:
                            # Convert comma-separated strings back to lists
                            try:
                                processed_config[key] = [item.strip() for item in value.split(',')]
                            except:
                                processed_config[key] = value
                        elif value.lower() in ('true', 'false'):
                            # Convert string booleans back to boolean
                            processed_config[key] = value.lower() == 'true'
                        else:
                            # Try to convert to number if possible, otherwise keep as string
                            try:
                                processed_config[key] = int(value)
                            except ValueError:
                                try:
                                    processed_config[key] = float(value)
                                except ValueError:
                                    processed_config[key] = value
                    
                    configs.append({
                        'id': ann_id,
                        'name': config_name,
                        'created_at': created_at,
                        'config': processed_config
                    })
                    
            except Exception as e:
                print(f"Error processing map annotation {ann_id}: {e}")
                continue
        
        print(f"Found {len(configs)} saved configurations")
        return configs
        
    except Exception as e:
        print(f"Error loading configurations from OMERO: {e}")
        return []


def save_config_to_omero(conn, datatype, object_id, config, trainingset_name=None):
    """
    Save configuration settings as OMERO map annotation using the namespace
    
    Args:
        conn: OMERO connection
        datatype: Type of OMERO object ('Dataset', 'Project', etc.)
        object_id: ID of the OMERO object
        config: Dictionary containing configuration settings
        trainingset_name: Optional name for the training set
        
    Returns:
        int: ID of created map annotation, or None if failed
    """
    try:
        # Create a name for the configuration
        config_name = trainingset_name if trainingset_name else f"microsam_config_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare the key-value pairs for the map annotation
        # Convert all values to strings as required by OMERO map annotations
        key_value_data = []
        for key, value in config.items():
            if isinstance(value, (list, tuple)):
                # Convert lists/tuples to comma-separated strings
                str_value = ','.join(map(str, value))
            elif isinstance(value, bool):
                # Convert boolean to string
                str_value = str(value).lower()
            else:
                str_value = str(value)
            key_value_data.append([key, str_value])
        
        # Add metadata about the configuration
        key_value_data.extend([
            ['config_name', config_name],
            ['created_at', pd.Timestamp.now().isoformat()],
            ['config_type', 'microsam_annotation_settings']
        ])
        
        # Create map annotation using ezomero
        map_ann_id = ezomero.post_map_annotation(
            conn,
            object_type=datatype,
            object_id=object_id,
            kv_dict=dict(key_value_data),
            ns="openmicroscopy.org/omero/annotation"
        )
        
        print(f"Saved configuration '{config_name}' as map annotation ID: {map_ann_id}")
        return map_ann_id
        
    except Exception as e:
        print(f"Error saving configuration to OMERO: {e}")
        return None


def get_dask_dimensions(conn, image_id):
    """
    Get dimensions of an image using dask for efficient handling of large images
    
    Args:
        conn: OMERO connection
        image_id: ID of the image to analyze
        
    Returns:
        dict: Dictionary containing image dimensions (sizeZ, sizeT, sizeC, sizeY, sizeX)
    """
    image = conn.getObject("Image", image_id)
    if not image:
        return None
    
    # Get dimensions directly without loading pixel data
    dims = {
        "sizeZ": image.getSizeZ(),
        "sizeT": image.getSizeT(),
        "sizeC": image.getSizeC(),
        "sizeY": image.getSizeY(),
        "sizeX": image.getSizeX()
    }
    
    return dims


def get_annotation_configurations(conn, container_type, container_id, namespace="openmicroscopy.org/omero/annotation"):
    """
    Retrieve configuration settings stored as annotations in OMERO
    
    Args:
        conn: OMERO connection
        container_type: Type of container ('project', 'dataset', 'image', etc.)
        container_id: ID of the container
        namespace: Namespace for the annotations to retrieve
        
    Returns:
        list: List of configuration dictionaries or empty list if none found
    """
    try:
        import json
        
        # Get annotations for this object with the specified namespace
        # Format required by ezomero
        formatted_type = f"{container_type.capitalize()}I"
        
        # Get all annotations with the specified namespace
        annotations = conn.getObjects("MapAnnotation", 
                                     opts={"object": formatted_type, 
                                          "id": container_id,
                                          "namespace": namespace})
        
        # Extract configuration settings
        configs = []
        for ann in annotations:
            try:
                # Get the key-value pairs
                kv_pairs = ann.getValue()
                
                # Convert to dictionary
                config = {}
                for k, v in kv_pairs:
                    # Try to convert values that look like JSON
                    if v.startswith('[') and v.endswith(']') or v.startswith('{') and v.endswith('}'):
                        try:
                            config[k] = json.loads(v)
                        except json.JSONDecodeError:
                            config[k] = v
                    # Handle boolean values
                    elif v.lower() == 'true':
                        config[k] = True
                    elif v.lower() == 'false':
                        config[k] = False
                    # Handle numeric values
                    elif v.isdigit():
                        config[k] = int(v)
                    else:
                        try:
                            config[k] = float(v)
                        except ValueError:
                            config[k] = v
                
                # Add the annotation ID
                config['annotation_id'] = ann.getId()
                configs.append(config)
                
            except Exception as e:
                print(f"Error processing annotation: {e}")
                continue
        
        return configs
        
    except Exception as e:
        print(f"Error getting annotation configurations: {e}")
        return []
