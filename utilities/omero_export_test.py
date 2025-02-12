import omero.gateway as gateway

# Connect to OMERO
conn = gateway.BlitzGateway('root', 'omero', host='localhost', port=4064)
conn.connect()

# Get the dataset
dataset = conn.getObject("Dataset", 502)

print(f"Dataset: {dataset.getName()}")

# Check each image and its annotations
for img in dataset.listChildren():
    print(f"\nImage ID: {img.getId()} - {img.getName()}")
    
    for ann in img.listAnnotations():
        print(f"  Annotation Type: {type(ann).__name__}")
        print(f"  Annotation ID: {ann.getId()}")
        
        # Special handling for different annotation types
        if isinstance(ann, gateway.FileAnnotationWrapper):
            orig_file = ann.getFile()
            print(f"    File: {orig_file.getName()}")
            print(f"    Size: {orig_file.getSize()}")
            print(f"    Path: {orig_file.getPath()}")
        elif isinstance(ann, gateway.MapAnnotationWrapper):
            print(f"    Key-Value Pairs: {ann.getValue()}")
            
        print("  ---")

conn.close()
