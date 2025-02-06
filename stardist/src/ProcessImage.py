import logging
from typing import List, Tuple, Any, Optional
import numpy as np
from pathlib import Path
from tifffile import imsave
import ezomero

# Import Python System Packages
import os

#stardist related
from stardist.models import StarDist2D
from csbdeep.utils import normalize

import pyclesperanto_prototype as cle
import pandas as pd

def measure_intensity(pixels, labels, size_z, size_t, size_c):
    all_statistics = []
    if size_z > 1 and size_t > 1:
        #raise error that time series and z-stack data is not supported
        raise ValueError("Time series and z-stack data is not supported (yet)")
    elif size_t > 1:
        for t, label in zip(range(size_t), labels):
            for c in range(size_c):
                statistics = cle.statistics_of_labelled_pixels(pixels.getPlane(0, c, t), label)
                statistics = pd.DataFrame(statistics)
                statistics['z'] = 0
                statistics['t'] = t
                statistics['channel'] = c
                all_statistics.append(statistics)   
    elif size_z > 1:
        for z, label in zip(range(size_z), labels):
            for c in range(size_c):
                statistics = cle.statistics_of_labelled_pixels(pixels.getPlane(z, c, 0), label)
                statistics = pd.DataFrame(statistics)
                statistics['z'] = z
                statistics['t'] = 0
                statistics['channel'] = c
                all_statistics.append(statistics)
    else:
        statistics = cle.statistics_of_labelled_pixels(pixels.getPlane(1, 0, 0), labels)
        statistics['z'] = 0
        statistics['t'] = 0
        all_statistics.append(statistics)
    
    # Concatenate all statistics into a single DataFrame
    all_statistics_df = pd.concat(all_statistics, ignore_index=True)
    
    return all_statistics_df


def calculate_norm_factor(slice_data, labels):
    """
    Calculate normalization factor for a single z-slice
    Args:
        slice_data: 2D array of intensities for one channel/slice
        labels: 2D array of nuclear labels for this slice
    Returns:
        float: normalization factor for this slice
    """
    # Get mean background intensity (where labels == 0)
    background = slice_data[labels == 0].mean()
    
    # Get mean foreground intensity (where labels > 0)
    foreground = slice_data[labels > 0].mean()
    
    # Calculate factor to normalize foreground-background difference
    if foreground - background > 0:
        return 1.0 / (foreground - background)
    return 1.0

def normalize_nuclei_intensities(img_data, labels, background_subtract=True):
    """
    Normalize nuclear intensities per z-slice
    Args:
        img_data: 4D array (z,c,y,x) with DAPI, GFP, RFP channels
        labels: 3D array (z,y,x) of nuclear labels
        background_subtract: Whether to perform background subtraction
    Returns:
        4D array of normalized intensities
    """
    norm_data = np.zeros_like(img_data, dtype=np.float32)
    
    # Process each z-slice and channel independently
    for z in range(img_data.shape[0]):
        for c in range(img_data.shape[1]):
            slice_data = img_data[z,c].astype(np.float32)
            slice_labels = labels[z]
            
            if background_subtract:
                # Subtract background (median of non-nucleus regions)
                background = np.median(slice_data[slice_labels == 0])
                slice_data = slice_data - background
                
            # Calculate normalization factor
            norm_factor = calculate_norm_factor(slice_data, slice_labels)
            norm_data[z,c] = slice_data * norm_factor
            
    return norm_data

def classify_cell_cycle(gfp_intensity, rfp_intensity):
    """
    Classify cell cycle phase based on FUCCI reporter intensities
    Args:
        gfp_intensity: Mean GFP intensity in nucleus
        rfp_intensity: Mean RFP intensity in nucleus
    Returns:
        str: Cell cycle phase ('G0', 'G1', 'G1/S', or 'G2/M')
    """
    # These thresholds need to be determined empirically
    gfp_thresh = 0.2  # Example threshold
    rfp_thresh = 0.2
    
    if gfp_intensity < gfp_thresh and rfp_intensity < rfp_thresh:
        return 'G0'
    elif rfp_intensity > rfp_thresh and gfp_intensity < gfp_thresh:
        return 'G1'  
    elif gfp_intensity > gfp_thresh and rfp_intensity > rfp_thresh:
        return 'G1/S'
    else:  # gfp high, rfp low
        return 'G2/M'

class ProcessImage:
    """Class to handle image processing and segmentation using StarDist."""
    
    # Class constants
    SEGMENTATION_NAMESPACE = "stardist.segmentation"
    ROI_NAME = "Stardist Nuclei"
    ROI_DESCRIPTION = "Nuclei segmentation using Stardist"
    IMAGE_DESCRIPTION = "Nuclei segmentation using Stardist"
    
    def __init__(self, conn: Any, image: Any,job_id: Any, model: Any) -> None:
        """
        Initialize ProcessImage instance.
        
        Args:
            conn: OMERO connection object
            image: OMERO image object
            job_id: Unique identifier for the job
            model: StarDist model object
        
        Raises:
            ValueError: If image or connection is invalid
        """
        if not image or not conn:
            raise ValueError("Image and connection must be provided")
        if not job_id:
            job_id = ""    
        self._image = image
        self._conn = conn
        self._pixels = image.getPrimaryPixels()
        self._size_c = image.getSizeC()
        self._size_z = image.getSizeZ()
        self._size_t = image.getSizeT()
        self._image_id = image.getId()
        self._labels = None
        self._polygons = None
        self._model = model
        self._job_id = job_id
        
    @property
    def labels(self) -> np.ndarray:
        """Get segmentation labels."""
        if self._labels is None:
            raise ValueError("Segmentation has not been performed yet")
        return self._labels
    def get_normalized_stack(self) -> np.ndarray:
        """
        Get all channels as normalized numpy array
        Returns:
            4D numpy array (z,c,y,x)
        """
        stack = np.zeros((self._size_z, self._size_c, 
                        self._pixels.getSizeY(), 
                        self._pixels.getSizeX()), dtype=np.float32)
        
        for z in range(self._size_z):
            for c in range(self._size_c):
                stack[z,c] = self._pixels.getPlane(z, c, 0)
                
        return normalize_nuclei_intensities(stack, self._labels)


    def get_image_stack(self) -> np.ndarray:
        """
        Get all channels as  numpy array
        Returns:
            4D numpy array (z,c,y,x)
        """
        stack = np.zeros((self._size_z, self._size_c, 
                        self._pixels.getSizeY(), 
                        self._pixels.getSizeX()), dtype=np.float32)
        
        for z in range(self._size_z):
            for c in range(self._size_c):
                stack[z,c] = self._pixels.getPlane(z, c, 0)
                
        return stack

    def measure_fucci_intensities(self):
        """
        Measure FUCCI reporter intensities in nuclei
        Returns:
            DataFrame with nuclear measurements including normalized intensities
        """
        if not hasattr(self, '_normalized_pixels'):
            self._normalized_pixels = self.get_normalized_stack()
        
        measurements = []
        for z in range(self._size_z):
            # Measure all channels for this z-slice
            stats = cle.statistics_of_labelled_pixels(
                self._normalized_pixels[z], self._labels[z])
            stats = pd.DataFrame(stats)
            stats['z'] = z
            measurements.append(stats)
        
        df = pd.concat(measurements, ignore_index=True)
    
        # Add classification
        df['cell_cycle_phase'] = classify_cell_cycle(
            df['mean_intensity'][df['channel'] == 1],  # GFP
            df['mean_intensity'][df['channel'] == 2])  # RFP
        
        return df    
    def normalize_intensities(self):
        """
        Normalize intensities for all channels
        """
        self._normalized_pixels = self.get_normalized_stack(
            self._pixels, self._labels)    
            
    def measure_intensity(self,norm = False):
        """
        Measure intensities in normalized images
        """
        if norm:
            if not hasattr(self, '_normalized_pixels'):
                self.normalize_intensities()
                
            self.all_statistics = measure_intensity(
                self._normalized_pixels, self._labels, 
                self._size_z, self._size_t, self._size_c)
            
            return self.all_statistics
        else:
            self.all_statistics = measure_intensity(
                self._pixels, self._labels, 
                self._size_z, self._size_t, self._size_c)
            
            return self.all_statistics
    
    def get_measurements_to_df(self):
        """
        Convert measurements to DataFrame
        """
        if not hasattr(self, 'all_statistics'):
            self.measure_intensity()
            
        return self.all_statistics

    def segment_nuclei(self, nucl_channel: int) -> None:
        """
        Segment nuclei in the specified channel.
        
        Args:
            nucl_channel: Channel number for nuclear staining
            
        Raises:
            ValueError: If channel number is invalid
        """
        if not 0 <= nucl_channel < self._size_c:
            raise ValueError(f"Invalid channel number: {nucl_channel}")
         
        try:
            # Check if data is either time series or z-stack, stack with both z and t is not supported
            if self._size_t > 1 and self._size_z > 1:
                raise ValueError("Time series and z-stack data is not supported (yet)")
            if self._size_t > 1:
                self._segment_time_series(nucl_channel)
            elif self._size_z > 1:
                self._segment_3d_image(nucl_channel)
            else:
                self._segment_2d_image(nucl_channel)
        except Exception as e:
            print(f"Segmentation failed: {str(e)}")
            raise
            
    def _segment_time_series(self, channel: int) -> None:
        """Handle time series segmentation."""
        planes = [self._pixels.getPlane(0, channel, t) for t in range(self._size_t)]
        labels_polygons = [self._segment_slice(plane) for plane in planes]
        self._labels, self._polygons = zip(*labels_polygons)
        
        # Reshape labels to match original image dimensions (t, z, c) -> (z, c, t)
        self._labels = np.moveaxis(np.array(self._labels), 0, 2)
        
        # Debugging
        print(f"Labels shape: {self._labels.shape}")
        print(f"Polygons: {self._polygons}")

    def _segment_3d_image(self, channel: int) -> None:
        """Handle 3D image segmentation."""
        planes = [self._pixels.getPlane(z, channel, 0) for z in range(self._size_z)]
        labels_polygons = [self._segment_slice(plane) for plane in planes]
        self._labels, self._polygons = zip(*labels_polygons)
        
    def _segment_2d_image(self, channel: int) -> None:
        """Handle 2D image segmentation."""
        plane = self._pixels.getPlane(0, channel, 0)
        labels_polygons = self._segment_slice(plane)
        self._labels, self._polygons = zip(*[labels_polygons])
        
    def _segment_slice(self, plane: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Segment a single image plane.
        
        Args:
            plane: 2D numpy array representing image plane
            
        Returns:
            Tuple of (labels, polygons)
        """
        try:
            img = normalize(plane)
            return self._model.predict_instances(img)
        except Exception as e:
            print(f"Slice segmentation failed: {str(e)}")
            raise
            
    def save_segmentation_to_omero_as_new_image(self, new_img_name: str, desc: Optional[str] = None) -> None:
        """
        Save segmentation as new OMERO image.
        
        Args:
            new_img_name: Name for the new image
            desc: Description for the new image
            
        Raises:
            ValueError: If segmentation hasn't been performed
        """
        if self._labels is None:
            raise ValueError("No segmentation data available - run segment_nuclei first")
        
        if desc is None:
            desc = self.IMAGE_DESCRIPTION

        try:
            # Convert labels to proper numpy array if needed
            labels_array = np.asarray(self._labels)
            
            # Create new image
            new_img = self._conn.createImageFromNumpySeq(
                plane_gen=iter(labels_array), 
                name=new_img_name,
                sizeZ=self._size_z,
                sizeC=1,
                sizeT=self._size_t,
                description=desc,
                dataset=self._image.getParent()
            )
            
            print(f'Created new Image:{new_img.getId()} Name:"{new_img.getName()}"')
            return new_img.getId()
            
        except Exception as e:
            print(f"Failed to save new image: {str(e)}")
            raise


    def save_segmentation_to_omero_as_attach(self, tmp_dir: str, desc: Optional[str] = None) -> None:
        """Save segmentation as OMERO attachment."""
        if self._labels is None:
            raise ValueError("No segmentation data available - run segment_nuclei first")
        
        if desc is None:
            desc = self.IMAGE_DESCRIPTION

        tmp_path = Path(tmp_dir)
        if not tmp_path.exists():
            tmp_path.mkdir(parents=True)
            
        tif_file = tmp_path / f"{self._image.getName()}_segmentation_{self._job_id}_{self._image_id}.tif"
        
        try:
            # Convert labels to proper numpy array if needed
            labels_array = np.asarray(self._labels)
            
            # Save the entire stack as a TIFF file
            imsave(tif_file, labels_array)
            
            file_annotation_id = ezomero.post_file_annotation(
                self._conn,
                str(tif_file),
                ns=self.SEGMENTATION_NAMESPACE,
                object_type="Image",
                object_id=self._image.getId(),
                description=desc
            )
            print(f'File annotation ID: {file_annotation_id}')
        finally:
            if tif_file.exists():
                tif_file.unlink()
                
    def save_segmentation_to_omero_as_roi(self) -> None:
        """Save segmentation as OMERO ROIs."""
        if not self._polygons:
            raise ValueError("No polygons available - run segmentation first")
            
        all_polygons = self._create_polygon_shapes()
        
        if all_polygons:
            try:
                roi_id = ezomero.post_roi(
                    conn=self._conn,
                    image_id=self._image.getId(),
                    shapes=all_polygons,
                    name=self.ROI_NAME,
                    description=self.ROI_DESCRIPTION
                )
                print(f"Created ROI with ID: {roi_id}")
            except Exception as e:
                print(f"Error creating ROI: {str(e)}")
                raise
        else:
            print("Warning: No valid polygons were created")
            
    def _create_polygon_shapes(self) -> List[dict]:
        """Create polygon shapes from segmentation results."""
        all_polygons = []
        if self._size_t > 1 and self._size_z > 1:
            raise ValueError("Time series and z-stack data is not supported (yet)")
        if self._size_t > 1:
            for t, polygons in enumerate(self._polygons):
                coords_array = polygons['coord']
                # Process each contour in the coordinates array
                for contour_idx in range(coords_array.shape[0]):
                    try:
                        # Extract x,y coordinates for current contour
                        xy_coords = coords_array[contour_idx]
                        # x and y coordinates are flipped from StarDist output
                        points = [(float(y), float(x)) for x, y in zip(xy_coords[0], xy_coords[1])]
                        
                        ezomero_polygon = ezomero.rois.Polygon(
                            points=points,
                            z=None,
                            c=None,
                            t=t,
                            label="nuclei",
                            fill_color=None,
                            stroke_color=None,
                            stroke_width=None
                        )
                        all_polygons.append(ezomero_polygon)
                    
                    except Exception as e:
                        print(f"Error processing contour {contour_idx} at t={t}: {e}")
                        continue
            return all_polygons
        elif self._size_z > 1:
            for z, polygons in enumerate(self._polygons):
                coords_array = polygons['coord']
                # Process each contour in the coordinates array
                for contour_idx in range(coords_array.shape[0]):
                    try:
                        # Extract x,y coordinates for current contour
                        xy_coords = coords_array[contour_idx]
                        # x and y coordinates are flipped from StarDist output
                        points = [(float(y), float(x)) for x, y in zip(xy_coords[0], xy_coords[1])]
                        
                        ezomero_polygon = ezomero.rois.Polygon(
                            points=points,
                            z=z,
                            c=None,
                            t=None,
                            label="nuclei",
                            fill_color=None,
                            stroke_color=None,
                            stroke_width=None
                        )
                        all_polygons.append(ezomero_polygon)
                    
                    except Exception as e:
                        print(f"Error processing contour {contour_idx} at t={t}: {e}")
                        continue
            return all_polygons
        
        elif self._size_z == 1 and self._size_t == 1:
            coords_array = self._polygons['coord']
            # Process each contour in the coordinates array
            for contour_idx in range(coords_array.shape[0]):
                try:
                    # Extract x,y coordinates for current contour
                    xy_coords = coords_array[contour_idx]
                    # x and y coordinates are flipped from StarDist output
                    points = [(float(y), float(x)) for x, y in zip(xy_coords[0], xy_coords[1])]
                    
                    ezomero_polygon = ezomero.rois.Polygon(
                        points=points,
                        z=None,
                        c=None,
                        t=None,
                        label="nuclei",
                        fill_color=None,
                        stroke_color=None,
                        stroke_width=None
                    )
                    all_polygons.append(ezomero_polygon)
                    
                except Exception as e:
                    print(f"Error processing contour {contour_idx}: {e}")
                    continue
            return all_polygons
        else:
            raise ValueError("Unsupported image dimensions")
        
    def visualize_measurements(self, channel: int = 0) -> None:
        """
        Visualize segmentation and measurements using napari
        
        Args:
            channel: Channel to display (default: 0)
        """
        import napari
        from napari.settings import get_settings
        settings = get_settings()
        settings.application.ipy_interactive = False
        
        # Get the image data for specified channel
        image_data = self._pixels.getPlane(0, channel, 0)
        
        # Create viewer and add image layer
        viewer = napari.Viewer()
        viewer.add_image(image_data, name='Image')
        
        if self._labels is not None:
            # Add labels layer
            viewer.add_labels(self._labels[0], name='Nuclei')
            
            # Create shape layer from polygons if available
            if self._polygons is not None:
                coords_array = self._polygons['coord']
                shapes = []
                shape_types = []
                for i in range(coords_array.shape[0]):
                    # Get coordinates for current polygon
                    xy_coords = coords_array[i]
                    # Create polygon points (swap x,y to match napari format)
                    points = np.column_stack((xy_coords[1], xy_coords[0]))
                    shapes.append(points)
                    shape_types.append('polygon')
                
                # Add shapes layer
                viewer.add_shapes(
                    shapes,
                    shape_type=shape_types,
                    edge_color='red',
                    face_color='transparent',
                    name='Polygons'
                )
                
            # Add measurements if available
            if hasattr(self, 'all_statistics'):
                # Get centroids and measurements
                stats = self.all_statistics
                centroids = np.column_stack((
                    stats['centroid_x'],
                    stats['centroid_y']
                ))
                
                # Create points layer with measurements
                viewer.add_points(
                    centroids,
                    properties={
                        'mean_intensity': stats['mean_intensity'],
                        'area': stats['area']
                    },
                    text={
                        'text': '{mean_intensity:.1f}',
                        'size': 8,
                    },
                    name='Measurements'
                )
        
        napari.run()