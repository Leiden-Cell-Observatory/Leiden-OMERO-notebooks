"""
Functions for image processing and ROI manipulation.
"""
import cv2
import numpy as np
import ezomero.rois


def mask_to_contour(mask):
    """Converts a binary mask to a list of ROI coordinates.

    Args:
        mask (np.ndarray): binary mask

    Returns:
        list: list of ROI coordinates
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def process_label_plane(label_plane, z_slice, channel, timepoint, model_type, x_offset=0, y_offset=0):
    """
    Process a single 2D label plane to generate OMERO shapes with optional offset
    
    Args:
        label_plane: 2D label plane (numpy array)
        z_slice: Z-slice index
        channel: Channel index
        timepoint: Time point index
        model_type: SAM model type identifier
        x_offset: X offset for contour coordinates (default: 0)
        y_offset: Y offset for contour coordinates (default: 0)
        
    Returns:
        list: List of OMERO shapes
    """
    shapes = []
    unique_labels = np.unique(label_plane)
    
    # Skip background (label 0)
    for label in unique_labels[1:]:
        # Create binary mask for this label
        mask = (label_plane == label).astype(np.uint8)
        
        # Get contours
        contours = mask_to_contour(mask)
        
        # Convert each contour to polygon ROI
        for contour in contours:
            contour = contour[:, 0, :]  # Reshape to (N, 2)
            
            # Apply offset to contour points if needed
            if x_offset != 0 or y_offset != 0:
                contour = contour + np.array([x_offset, y_offset])
                
            # Create polygon without text parameter
            poly = ezomero.rois.Polygon(
                points=contour,  # explicitly name the points parameter
                z=z_slice,
                c=channel,
                t=timepoint,
                label=f'micro_sam.{"volumetric" if isinstance(z_slice, (list, range)) or z_slice > 0 else "manual"}_instance_segmentation.{model_type}'
            )
            shapes.append(poly)
    
    return shapes


def label_to_rois(label_img, z_slice, channel, timepoint, model_type, is_volumetric=False, patch_offset=None):
    """
    Convert a 2D or 3D label image to OMERO ROI shapes
    
    Args:
        label_img (np.ndarray): 2D labeled image or 3D labeled stack
        z_slice (int or list): Z-slice index or list/range of Z indices
        channel (int): Channel index
        timepoint (int): Time point index
        model_type (str): SAM model type used
        is_volumetric (bool): Whether the label image is 3D volumetric data
        patch_offset: Optional (x,y) offset for placing ROIs in a larger image
    
    Returns:
        list: List of OMERO shape objects
    """
    shapes = []
    
    # Unpack patch offset if provided
    x_offset, y_offset = (0, 0) if patch_offset is None else patch_offset
    
    if is_volumetric and label_img.ndim > 2:
        # 3D volumetric data - process each z slice
        for z_index, z_plane in enumerate(label_img):
            # If z_slice is a range or list, use the actual z-index from that range
            if isinstance(z_slice, (range, list)):
                actual_z = z_slice[z_index] if z_index < len(z_slice) else z_slice[0] + z_index
            else:
                actual_z = z_slice + z_index  # Assume z_slice is the starting index
                
            print(f"Processing volumetric ROIs for z-slice {actual_z}")
            shapes.extend(process_label_plane(z_plane, actual_z, channel, timepoint, model_type, 
                                            x_offset, y_offset))
    else:
        # 2D data - process single plane
        shapes.extend(process_label_plane(label_img, z_slice, channel, timepoint, model_type, 
                                        x_offset, y_offset))
    
    return shapes


def generate_patch_coordinates(image_width, image_height, patch_size, num_patches, random_patches=True):
    """
    Generate coordinates for image patches
    
    Args:
        image_width: Width of the full image
        image_height: Height of the full image
        patch_size: Tuple of (width, height) for the patch
        num_patches: Number of patches to generate
        random_patches: If True, generate random patches; if False, generate centered patches
        
    Returns:
        list: List of patch coordinates as tuples (x, y, width, height)
    """
    patch_width, patch_height = patch_size
    
    # Ensure patch size is not larger than image
    patch_width = min(patch_width, image_width)
    patch_height = min(patch_height, image_height)
    
    patches = []
    
    if random_patches:
        # Generate random patches
        for _ in range(num_patches):
            # Calculate valid coordinate ranges
            max_x = image_width - patch_width
            max_y = image_height - patch_height
            
            if max_x <= 0 or max_y <= 0:
                # Image is too small for the patch, use full image
                patches.append((0, 0, image_width, image_height))
            else:
                # Generate random coordinates
                x = np.random.randint(0, max_x + 1)
                y = np.random.randint(0, max_y + 1)
                patches.append((x, y, patch_width, patch_height))
    else:
        # Generate centered patch
        x = (image_width - patch_width) // 2
        y = (image_height - patch_height) // 2
        
        # Add the centered patch (potentially multiple times if num_patches > 1)
        for _ in range(num_patches):
            patches.append((x, y, patch_width, patch_height))
    
    return patches


def extract_patch(image_array, patch_coords):
    """
    Extract a patch from an image array
    
    Args:
        image_array: Numpy array containing the image data
        patch_coords: Tuple of (x, y, width, height)
        
    Returns:
        numpy.ndarray: Extracted patch
    """
    x, y, width, height = patch_coords
    
    # Handle different dimensionality
    if image_array.ndim == 2:
        # 2D image
        return image_array[y:y+height, x:x+width]
    elif image_array.ndim == 3:
        # 3D image (z-stack or multi-channel)
        return image_array[:, y:y+height, x:x+width]
    else:
        # Higher dimensions (e.g., z-stack + multi-channel)
        return image_array[..., y:y+height, x:x+width]
