"""Image processing functions for micro-SAM workflows."""

from typing import List, Tuple
import random


def generate_patch_coordinates(image_shape: Tuple[int, int], patch_size: Tuple[int, int], 
                              n_patches: int, random: bool = True) -> List[Tuple[int, int]]:
    """Generate patch coordinates for an image.
    
    Args:
        image_shape: (height, width) of the image
        patch_size: (height, width) of patches
        n_patches: Number of patches to generate
        random: Whether to generate random patches or centered patches
        
    Returns:
        List of (x, y) coordinates for patch top-left corners
    """
    height, width = image_shape
    patch_h, patch_w = patch_size
    
    # Ensure patches fit within image
    max_x = max(0, width - patch_w)
    max_y = max(0, height - patch_h)
    
    if max_x <= 0 or max_y <= 0:
        # Image smaller than patch, return image center
        return [(0, 0)]
    
    coordinates = []
    
    if random:
        # Generate random coordinates
        for _ in range(n_patches):
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            coordinates.append((x, y))
    else:
        # Generate centered patches
        if n_patches == 1:
            # Single centered patch
            x = max_x // 2
            y = max_y // 2
            coordinates.append((x, y))
        else:
            # Grid of patches
            grid_size = int(n_patches ** 0.5)
            if grid_size * grid_size < n_patches:
                grid_size += 1
            
            step_x = max_x // max(1, grid_size - 1) if grid_size > 1 else 0
            step_y = max_y // max(1, grid_size - 1) if grid_size > 1 else 0
            
            for i in range(n_patches):
                grid_x = i % grid_size
                grid_y = i // grid_size
                
                x = min(grid_x * step_x, max_x)
                y = min(grid_y * step_y, max_y)
                coordinates.append((x, y))
    
    return coordinates[:n_patches]