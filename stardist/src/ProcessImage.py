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


class ProcessImage:
    """Class to handle image processing and segmentation using StarDist."""
    
    # Class constants
    SEGMENTATION_NAMESPACE = "stardist.segmentation"
    ROI_NAME = "Stardist Nuclei"
    ROI_DESCRIPTION = "Nuclei segmentation using Stardist"
    
    def __init__( conn: Any,self, image: Any,job_id: Any, model: Any) -> None:
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
            
    def save_segmentation_to_omero_as_new_image(self, new_img_name: str, desc: str) -> None:
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

    def save_segmentation_to_omero_as_attach(self, tmp_dir: str, desc: str) -> None:
        """Save segmentation as OMERO attachment."""
        if self._labels is None:
            raise ValueError("No segmentation data available - run segment_nuclei first")
        
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