# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.


import sys
import numpy as np
import os
import imageio
from tifffile import imread, imwrite
import skimage
import skimage.color
from csbdeep.utils import normalize
from stardist import random_label_cmap
from stardist.models import StarDist2D
from cytomine.models import Job
from biaflows import CLASS_OBJSEG
from biaflows.helpers import BiaflowsJob, prepare_data, upload_data, upload_metrics

def run_startdist(bj,models,img):
    fluo = True
    n_channel = 3 if img.ndim == 3 else 1

    if n_channel == 3:
        if np.array_equal(img[:,:,0], img[:,:,1]) and np.array_equal(img[:,:,0], img[:,:,2]):
            img = skimage.color.rgb2gray(img)
        else:
            fluo = False
    axis_norm = (0,1)
    img = normalize(img, bj.parameters.stardist_norm_perc_low, bj.parameters.stardist_norm_perc_high,axis=axis_norm)
    if fluo:
        labels, details = models[0].predict_instances(img,
                                                       prob_thresh=bj.parameters.stardist_prob_t,
                                                       nms_thresh=bj.parameters.stardist_nms_t)
    else:
        labels, details = models[1].predict_instances(img,
                                                     prob_thresh=bj.parameters.stardist_prob_t,
                                                     nms_thresh=bj.parameters.stardist_nms_t)
    labels = labels.astype(np.uint16)
    bj.job.update(status=Job.RUNNING, progress=30, statusComment="Maxiumum value in labels: {}".format(np.max(labels)))
    return labels

def main(argv):
    base_path = "{}".format(os.getenv("HOME"))
    problem_cls = CLASS_OBJSEG

    with BiaflowsJob.from_cli(argv) as bj:
        bj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialization...")

        # 1. Prepare data for workflow
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, bj, is_2d=False, **bj.flags)
        list_imgs = [image.filepath for image in in_imgs]
        nuc_channel = bj.parameters.nuc_channel
        channels = bj.parameters.channels
        time_series = bj.parameters.time_series
        z_slices = bj.parameters.z_slices

        # 2. Run Stardist model on input images
        bj.job.update(progress=25, statusComment="Launching workflow...")
        bj.job.update(progress=26, statusComment="Processing images with channels is {}, time_series is {}, z_slices is {}".format(channels, time_series, z_slices))

        #Loading pre-trained Stardist model
        np.random.seed(17)

        lbl_cmap = random_label_cmap()
        model_fluo = StarDist2D(None, name='2D_versatile_fluo', basedir='/models/')
        model_he = StarDist2D(None, name='2D_versatile_he', basedir='/models/')
        models = [model_fluo, model_he]
        # handle all possible input cases (x,y), (c,z,x,y), (c,t,x,y), (z,t,x,y), (z,x,y),(c,x,y),(t,x,y),(c,z,t,x,y)
        bj.job.update(progress=30, statusComment="length of list_imgs: {}".format(len(list_imgs)))
        for img_path in list_imgs:
            #check if image is 2D or 3D
            img = imread(img_path)
            dims = img.shape
            labels = np.zeros_like(img)
            bj.job.update(progress=30, statusComment="Number of dimensions in image: {}".format(len(dims)))
            bj.job.update(progress=30, statusComment="shape of full image "+ str(img.shape))

            if len(dims) == 5:
                _, nz, nt = img.shape[:3]
                for z, t in np.ndindex(nz, nt):
                    # Process single xy slice
                    processed_slice = run_startdist(bj,models,img[nuc_channel,z,t])
                    # Store processed slice
                    labels[nuc_channel,z,t] = processed_slice
                metadata = {'axes': 'CZTYX'}
            elif len(dims) == 4:
                if channels and z_slices:
                    _, nz = img.shape[:2]
                    bj.job.update(progress=30, statusComment="nz is "+ str(nz))
                    for z in range(nz):
                        processed_slice = run_startdist(bj,models,img[nuc_channel,z])
                        # Store processed slice
                        labels[nuc_channel,z] = processed_slice
                    metadata = {'axes': 'CZYX'}                    
                elif channels and time_series:
                    _, nt = img.shape[:2]
                    for t in range(nt):
                        processed_slice = run_startdist(bj,models,img[nuc_channel,t])
                        # Store processed slice
                        labels[nuc_channel,t] = processed_slice
                    metadata = {'axes': 'CTYX'}                    
                elif z_slices and time_series:
                    nz , nt = img.shape[:2]
                    for z, t in np.ndindex(nz, nt):
                        # Process single xy slice
                        processed_slice = run_startdist(bj,models,img[z,t])
                        # Store processed slice
                        labels[z,t] = processed_slice
                    metadata = {'axes': 'ZTYX'}                    
            elif len(dims) == 3:
                if z_slices:
                    nz = img.shape[0]
                    for z in range(nz):
                        processed_slice = run_startdist(bj,models,img[z])
                        # Store processed slice
                        labels[z] = processed_slice
                    metadata = {'axes': 'ZYX'}                      
                elif time_series:
                    nt = img.shape[0]
                    for t in np.ndindex(nt):
                        processed_slice = run_startdist(bj,models,img[t])
                        # Store processed slice
                        labels[t] = processed_slice
                    metadata = {'axes': 'TYX'}                      
                elif channels:
                    labels[nuc_channel] = run_startdist(bj,models,img[nuc_channel])
                    metadata = {'axes': 'CYX'}
            elif len(dims) == 2:
                labels = run_startdist(bj,models,img)
                metadata = {'axes': 'YX'}
            bj.job.update(progress=90, statusComment="shape of labels: "+ str(labels.shape))
            imwrite(os.path.join(out_path,os.path.basename(img_path)), labels,ome=True,metadata=metadata,photometric='minisblack')

        # 3. Upload data to BIAFLOWS
        upload_data(problem_cls, bj, in_imgs, out_path, **bj.flags, monitor_params={
            "start": 60, "end": 90, "period": 0.1,
            "prefix": "Extracting and uploading polygons from masks"})
        
        # 4. Compute and upload metrics
        bj.job.update(progress=90, statusComment="Computing and uploading metrics...")
        upload_metrics(problem_cls, bj, in_imgs, gt_path, out_path, tmp_path, **bj.flags)

        # 5. Pipeline finished
        bj.job.update(progress=100, status=Job.TERMINATED, status_comment="Finished.")

if __name__ == "__main__":
    main(sys.argv[1:])
