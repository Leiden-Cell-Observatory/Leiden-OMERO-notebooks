import ezomero
import pandas as pd

# ====== CONFIGURATION ======
OMERO_CREDENTIALS = {
    'user': 'user',
    'password': 'password',
    'host': 'localhost',
    'port': 4064,
    'group': 'users',
    'secure': True
}

PROJECT_IDS = [101]

OUTPUT_FILE = 'omero_images.tsv'
# ==========================

def get_project_images(conn, project_ids):
    image_data = []

    for proj_id in project_ids:
        try:
            project = conn.getObject("Project", proj_id)
            project_name = project.getName() if project else "Unknown Project"
            print(f"\nProcessing Project: {project_name} (ID: {proj_id})")

            dataset_ids = ezomero.get_dataset_ids(conn, project=proj_id)
            print(f"Found {len(dataset_ids)} datasets")

            for ds_id in dataset_ids:
                image_ids = ezomero.get_image_ids(conn, dataset=ds_id)
                print(f"Processing Dataset ID {ds_id}: {len(image_ids)} images")
                print(image_ids)
                for im_id in image_ids:
                    try:
                        image_tuple = ezomero.get_image(conn, image_id=im_id, no_pixels=True)  # Corrected call
                        image = image_tuple[0]  # Get the image object from the tuple

                        if isinstance(image, str):  # Check if image retrieval failed
                            print(f"Error getting Image ID {im_id}: {image}")  # Print the error message from ezomero
                            image_name = "Error retrieving image"
                            original_file = "Error retrieving image"
                        elif image:
                            paths = ezomero.get_original_filepaths(conn, image_id=im_id)
                            original_file = paths[0] if paths else "No file path found"
                            image_name = image.getName()
                        else:  # Should not happen, but for completeness
                            image_name = "Unknown Image"
                            original_file = "Unknown file"

                        image_data.append({
                            'Project_ID': proj_id,
                            'Project_Name': project_name,
                            'Dataset_ID': ds_id,
                            'Image_ID': im_id,
                            'Image_Name': image_name,
                            'Original_File': original_file
                        })

                    except Exception as e:
                        print(f"Unexpected error processing Image ID {im_id}: {str(e)}")

        except Exception as e:
            print(f"Error processing Project ID {proj_id}: {str(e)}")

    return image_data


if __name__ == "__main__":
    conn = ezomero.connect(**OMERO_CREDENTIALS)

    try:
        print(f"Getting images from {len(PROJECT_IDS)} projects")
        image_data = get_project_images(conn, PROJECT_IDS)

        df = pd.DataFrame(image_data)
        df.to_csv(OUTPUT_FILE, sep='\t', index=False)
        print(f"\nSaved {len(image_data)} images to {OUTPUT_FILE}")

    finally:
        conn.close()