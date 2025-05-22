"""
Utility functions for omero-microsam batch annotation.
"""
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def interleave_arrays(train_images, validate_images):
    """
    Interleave two arrays of images in the pattern: train[0], validate[0], train[1], validate[1], ...
    If arrays are of unequal length, remaining elements are appended at the end.
    
    Args:
        train_images: List of training images
        validate_images: List of validation images
        
    Returns:
        tuple: (interleaved_images, sequence_markers)
            where sequence_markers has 0 for train images and 1 for validate images
    """
    # Create empty list to store interleaved images
    interleaved = []
    sequence = []
    # Get the length of the longer array
    max_len = max(len(train_images), len(validate_images))
    
    # Interleave the arrays
    for i in range(max_len):
        # Add train image if available
        if i < len(train_images):
            interleaved.append(train_images[i])
            sequence.append(0)
        # Add validate image if available
        if i < len(validate_images):
            interleaved.append(validate_images[i])
            sequence.append(1)
    
    return np.array(interleaved), np.array(sequence)
