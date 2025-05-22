"""
OMERO MicroSAM module for batch annotation of OMERO images.
"""

from .utils import NumpyEncoder, interleave_arrays
from .image_functions import (
    mask_to_contour, 
    process_label_plane, 
    label_to_rois, 
    generate_patch_coordinates, 
    extract_patch
)
from .omero_functions import (
    print_object_details, 
    get_images_from_container, 
    get_dask_image, 
    upload_rois_and_labels
)
from .file_io_functions import (
    zip_directory,
    store_annotations_in_zarr,
    zarr_to_tiff,
    cleanup_local_embeddings,
    organize_local_outputs,
    save_annotations_schema
)
from .processing_pipeline import process_omero_batch_with_dask

__all__ = [
    'NumpyEncoder', 
    'interleave_arrays',
    'mask_to_contour', 
    'process_label_plane', 
    'label_to_rois', 
    'generate_patch_coordinates', 
    'extract_patch',
    'print_object_details', 
    'get_images_from_container', 
    'get_dask_image', 
    'upload_rois_and_labels',
    'zip_directory',
    'store_annotations_in_zarr',
    'zarr_to_tiff',
    'cleanup_local_embeddings',
    'organize_local_outputs',
    'save_annotations_schema',
    'process_omero_batch_with_dask'
]
