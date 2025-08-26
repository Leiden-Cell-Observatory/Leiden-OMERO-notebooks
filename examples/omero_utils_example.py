#!/usr/bin/env python3
"""
Example usage of OMERO utility functions

This script demonstrates how to use the new OMERO utility functions
for table management, annotation handling, and error recovery.
"""

import sys
import os

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def example_table_management(conn):
    """Example of table management utilities"""
    print("📋 Table Management Examples")
    print("=" * 40)
    
    from omero_annotate_ai.omero.omero_utils import (
        list_user_tables,
        find_table_by_pattern,
        backup_table,
        validate_table_schema
    )
    
    # List tables in a project
    print("1. Listing tables in project 101...")
    tables = list_user_tables(conn, "project", 101)
    
    for table in tables:
        print(f"   - {table['name']} (ID: {table['id']})")
    
    # Find specific table
    print("\n2. Finding tables with 'microsam' in name...")
    table = find_table_by_pattern(conn, "project", 101, "microsam")
    
    if table:
        print(f"   Found: {table['name']}")
        
        # Backup the table
        print("\n3. Creating backup...")
        backup_path = f"/tmp/backup_{table['name']}.csv"
        backup_table(conn, table['id'], backup_path)
        
        # Validate schema
        print("\n4. Validating table schema...")
        expected_columns = ['image_id', 'processed', 'sam_model']
        is_valid, missing = validate_table_schema(conn, table['id'], expected_columns)
        
        if not is_valid:
            print(f"   Missing columns: {missing}")

def example_annotation_management(conn):
    """Example of annotation management utilities"""
    print("\n🏷️ Annotation Management Examples")
    print("=" * 40)
    
    from omero_annotate_ai.omero.omero_utils import (
        list_annotations_by_namespace,
        validate_roi_integrity
    )
    
    # List micro-SAM annotations
    print("1. Listing micro-SAM annotations on image 252...")
    annotations = list_annotations_by_namespace(
        conn, "Image", 252, "openmicroscopy.org/omero/microsam"
    )
    
    for ann in annotations:
        print(f"   - {ann['file_name']} ({ann['description']})")
    
    # Check ROI integrity
    print("\n2. Validating ROI integrity...")
    integrity = validate_roi_integrity(conn, 252)
    
    print(f"   Total ROIs: {integrity['total_rois']}")
    print(f"   Total shapes: {integrity['total_shapes']}")
    print(f"   Valid: {integrity['is_valid']}")

def example_error_handling():
    """Example of error handling utilities"""
    print("\n🔧 Error Handling Examples")
    print("=" * 40)
    
    from omero_annotate_ai.omero.omero_utils import with_retry
    
    # Example function that might fail
    attempt_count = 0
    def unreliable_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception(f"Attempt {attempt_count} failed")
        return f"Success on attempt {attempt_count}"
    
    print("1. Testing retry mechanism...")
    result = with_retry(unreliable_function, max_retries=5)
    print(f"   Result: {result}")

def example_dask_loading(conn):
    """Example of improved dask loading"""
    print("\n📊 Dask Loading Examples")
    print("=" * 40)
    
    from omero_annotate_ai.omero.omero_functions import (
        get_dask_image_single,
        get_dask_image_multiple
    )
    
    # Get some images
    try:
        # Get project and first few images
        project = conn.getObject("Project", 101)
        if project:
            images = []
            for dataset in project.listChildren():
                for image in dataset.listChildren():
                    images.append(image)
                    if len(images) >= 2:  # Just get 2 for demo
                        break
                if len(images) >= 2:
                    break
            
            if images:
                print(f"1. Loading {len(images)} images with dask...")
                
                # Load single image
                single_image = get_dask_image_single(
                    conn, images[0], 
                    timepoints=[0], channels=[0], z_slices=[0]
                )
                
                if single_image is not None:
                    print(f"   Single image loaded: {single_image.shape}")
                
                # Load multiple images
                if len(images) > 1:
                    multiple_images = get_dask_image_multiple(
                        conn, images, 
                        timepoints=[0], channels=[0], z_slices=[0]
                    )
                    print(f"   Multiple images loaded: {len(multiple_images)} images")
                    
                    for i, img in enumerate(multiple_images):
                        print(f"     Image {i}: {img.shape}")
            else:
                print("   No images found in project 101")
    except Exception as e:
        print(f"   Error loading images: {e}")

def main():
    """Main example function"""
    print("🚀 OMERO Utilities Usage Examples")
    print("=" * 50)
    
    # For this example, we'll use mock/None connection
    # In real usage, you would connect to OMERO first:
    # import ezomero
    # conn = ezomero.connect(host, port, user, password, group)
    
    conn = None  # Mock connection for demo
    print("Note: Using mock connection for demonstration")
    print("In real usage, connect to OMERO first with ezomero.connect()")
    
    try:
        # Run examples (these will show error handling with None connection)
        example_table_management(conn)
        example_annotation_management(conn)
        example_error_handling()
        example_dask_loading(conn)
        
        print("\n✅ Example completed!")
        print("\nTo use with real OMERO connection:")
        print("   import ezomero")
        print("   conn = ezomero.connect(host, port, user, password, group)")
        print("   # Then run the examples with real conn")
        
    except Exception as e:
        print(f"❌ Error in examples: {e}")

if __name__ == "__main__":
    main()