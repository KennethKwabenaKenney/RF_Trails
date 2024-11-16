# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:41:34 2024

@author: kenneyke
"""

import os
# Keep using Keras 2
#os.environ['TF_USE_LEGACY_KERAS'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # TODO: uncomment this and restart kernel to not use GPU # check before shipping 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

import tensorflow_decision_forests as tfdf

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
import math
import urllib.request
#%%Limiting the output height in colab
from IPython.core.magic import register_line_magic
from IPython.display import Javascript
from IPython.display import display as ipy_display

# Some of the model training logs can cover the full
# screen if not compressed to a smaller viewport.
# This magic allows setting a max height for a cell.
@register_line_magic
def set_cell_height(size):
  ipy_display(
      Javascript("google.colab.output.setIframeHeight(0, true, {maxHeight: " +
                 str(size) + "})"))

#%% Check the version of TensorFlow Decision Forests
print(tfdf.__version__)

