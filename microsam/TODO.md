# OMERO MicroSAM Annotation - TODO List

## High Priority (Core Functionality)
### Easy 🟢
- [x] Better way to handle when there are multiple annotation runs on a dataset (now they all get the same names, add date+start time?)
- [ ] Store annotations locally so it could also work with IDR
- [ ] Next to ROIs also attach label map

### Medium 🟡
- [ ] Avoid overlapping patches
- [ ] Better schema to track annotations (replace OMERO table with dedicated schema)

### Difficult 🔴
- [ ] Store all annotations into OMERO (see: https://github.com/computational-cell-analytics/micro-sam/issues/445; in series annotator possible to add commit path with prompts, but they get overwritten)

## Medium Priority (Enhanced Features)
### Easy 🟢
- [ ] Inspect dataset structure functionality
- [ ] Build a little widget in the notebook to select the data settings
- [ ] Handle multiple datasets in single run

### Medium 🟡
- [ ] Use image metadata to pick training data (e.g., balance treated/untreated cells - we have untreated and treated cells, balance training to have equal number of those)
- [ ] Multiple channels support (convert to RGB? current microsam expects single channel)

### Difficult 🔴
- [ ] Classification instead of semantic segmentation (not sure this is directly possible with current microsam annotator, but we can run twice and add a label to the annotations?)

## Low Priority (Quality of Life)
### Easy 🟢
- [ ] Clean up the errors and warnings output from napari
- [ ] Add training metrics

### Medium 🟡
- [ ] Add recovery mode to handle cases when users abort in the middle of a batch annotation session (currently annotations made before closing are preserved, but could be improved with a dedicated recovery workflow)
- [ ] Make annotation process more robust by checking existing local files first and maintaining a local progress tracking table to enable seamless continuation of interrupted annotation sessions
- [ ] Improve ROI creation for 3D volumes to better represent volumetric masks in OMERO

### Difficult 🔴
- [ ] Work with Dask arrays directly in micro-sam

## Export/Integration Features
### Easy 🟢
- [ ] Package training data export functions for:
  - [ ] micro_sam
  - [ ] cellpose

### Medium 🟡
- [ ] Package training data export functions for:
  - [ ] biapy
  - [ ] zerocost
- [ ] Instead of OMERO table use a schema

## Deployment/Infrastructure
### Medium 🟡
- [ ] Host on jupyter server with docker

## Notes
- Recovery mode: Currently annotations made before closing are preserved, but could be improved with a dedicated recovery workflow
- OMERO integration: In series annotator it's possible to add commit path with prompts, but they get overwritten
