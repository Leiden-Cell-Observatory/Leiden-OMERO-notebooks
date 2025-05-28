import ezomero
import os
import pandas as pd

#simple script to retrieve all tables from an image in OMERO
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

    image_id = 3013
    
    # Get all file annotation IDs for this image
    file_ann_ids = ezomero.get_file_annotation_ids(conn, "Image", image_id)
    print(f"Found {len(file_ann_ids)} file annotations")
    
    # Read each table
    tables = []
    for ann_id in file_ann_ids:
        try:
            table = ezomero.get_table(conn, ann_id)
            if table is not None:
                tables.append(table)
                print(f"Successfully read table from annotation {ann_id}")
        except Exception as e:
            print(f"Error reading table from annotation {ann_id}: {e}")

    print(f"\nSuccessfully read {len(tables)} tables")
    
    # If using pandas, you can examine the tables
    for i, table in enumerate(tables):
        if isinstance(table, pd.DataFrame):
            print(f"\nTable {i+1} shape: {table.shape}")
            print("Columns:", table.columns.tolist())
            print("Preview:")
            print(table.head())