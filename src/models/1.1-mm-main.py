''' Copyright (C) 2021 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''

# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 
import os
import pandas as pd
import tsfel

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
modulesdir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '__modules__'))
sys.path.append(modulesdir)

%load_ext autoreload
%autoreload 2
## from src.__modules__ import setup
from __modules__ import imports
from __modules__ import process 
from __modules__ import learn 
from __modules__ import evaluate 

# Caching Libraries
import joblib
from shutil import rmtree
location = 'cachedir'
memory = joblib.Memory(location=location, verbose=10)

import numpy as np
np.random.seed(42)

# ------------------------------------------------------------------------- #
#                           Importing Datasets                              #
# ------------------------------------------------------------------------- #
## dev
## test
## train
## 


# ------------------------------------------------------------------------- #
#                           Feature Selection                               #
# ------------------------------------------------------------------------- #
