#!/usr/bin/env python

import cv2
from numpy import empty, nan
import os
import sys
import time
import math
import logging as log

import CMT
import numpy as np
import util

import trax.server
import trax.region
import trax.image

def read_trax_image(image):
    if image.type == trax.image.PATH:
        return cv2.imread(image.path)
    if image.type == trax.image.MEMORY:
        return image.image
    if image.type == trax.image.BUFFER:
        return cv2.imdecode(np.fromstring(image.data, dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
    return None

options = trax.server.ServerOptions(trax.region.RECTANGLE, [trax.image.PATH, trax.image.MEMORY, trax.image.BUFFER])
region = None
tracker = None
with trax.server.Server(options, verbose=True) as server:
    while True:
        request = server.wait()
        if request.type in ["quit", "error"]:
            break
        if request.type == "initialize":
            region = request.region
            im0 = read_trax_image(request.image)
            im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            tl = (region.x, region.y)
            br = (region.x + region.width - 1, region.y + region.height - 1)
            tracker = None
            try:
                 tracker = CMT.CMT()
                 tracker.initialise(im_gray0, tl, br)
            except Exception:
                 tracker = None
        else:
            im = read_trax_image(request.image)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if not tracker == None:
                tracker.process_frame(im_gray)
                if not math.isnan(tracker.bb[0]):
                   region = trax.region.Rectangle(tracker.bb[0], tracker.bb[1], tracker.bb[2], tracker.bb[3])
                else:
                   region = trax.region.Special(0)
            else:
                region = trax.region.Special(0)
        server.status(region)



