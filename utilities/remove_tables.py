import ezomero
import os

def remove_tables_from_dataset_images(conn, dataset_id, table_name_pattern=None):
    """
    Remove tables from all images in a dataset that match an optional name pattern.
    
    Args:
        conn: OMERO connection object
        dataset_id (int): ID of the dataset
        table_name_pattern (str, optional): Pattern to match table names
    """
    # Get all images in the dataset
    image_ids = ezomero.get_image_ids(conn, dataset=dataset_id)
    print(f"Found {len(image_ids)} images in dataset {dataset_id}")
    
    total_removed = 0
    
    for image_id in image_ids:
        # Get all file annotation IDs for this image
        file_ann_ids = ezomero.get_file_annotation_ids(conn, "Image", image_id)
        
        for ann_id in file_ann_ids:
            try:
                # Get the actual annotation object
                ann = conn.getObject("FileAnnotation", ann_id)
                filename = ann.getFileName()
                print(filename)
            
                if table_name_pattern is None or table_name_pattern in filename:
                    # Delete the annotation
                    conn.deleteObjects("FileAnnotation", [ann_id], wait=True)
                    total_removed += 1
                    print(f"Removed table '{filename}' from image {image_id}")
                    
            except Exception as e:
                print(f"Error processing annotation {ann_id} on image {image_id}: {e}")
    
    print(f"\nTotal tables removed: {total_removed}")

if __name__ == '__main__':
    # Connect to OMERO
    conn = ezomero.connect(
        user=os.environ.get("USER_NAME"),
        password=os.environ.get("PASSWORD"), 
        group=os.environ.get("GROUP"),
        host=os.environ.get("HOST"),
        port=os.environ.get("PORT"),
        secure=True
    )

    # Example usage:
    dataset_id = 1112  # Replace with your dataset ID
    table_pattern = "Nuclei_measurements"  # Optional: specify pattern to match table names
    
    # Remove tables
    remove_tables_from_dataset_images(conn, dataset_id, table_pattern)
    
    # Close connection
    conn.close()