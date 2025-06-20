# Example Tests

This folder contains example test scripts that demonstrate real-world usage of the omero-annotate-ai package.

## test_annotation_pipeline.py

A comprehensive end-to-end test that demonstrates the complete micro-SAM annotation workflow:

- Creates a configuration with batch_size=0 and vit_b_lm model
- Connects to a real OMERO server
- Runs the full annotation pipeline on a test dataset
- Uploads ROIs and creates tracking tables
- Demonstrates proper table ID management

### Requirements

- Active OMERO server connection
- Valid `.env` file with OMERO credentials
- Test dataset (Dataset ID 351 by default)
- micro-sam installed via conda

### Usage

```bash
# Ensure you have the package installed
pip install -e .

# Run the test
python example_test/test_annotation_pipeline.py
```

### Key Features Demonstrated

1. **Configuration Creation**: Shows how to create and customize annotation configs
2. **OMERO Integration**: Demonstrates connection and dataset loading
3. **Pipeline Execution**: Full micro-SAM processing workflow
4. **Table Management**: Proper handling of OMERO table updates and ID changes
5. **Error Handling**: Robust error handling and cleanup

This test serves as both a validation tool and a reference implementation for users wanting to understand how to use the package in their own workflows.