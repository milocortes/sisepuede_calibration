import functools
import numpy as np
import pandas as pd
import math

# Load LAC-Decarbonization source
import sys
import os

cwd = os.environ['LAC_PATH']
sys.path.append(os.path.join(cwd, 'python'))

import setup_analysis as sa
import support_functions as sf
import sector_models as sm
import argparse

from model_socioeconomic import Socioeconomic
from model_afolu import AFOLU
from model_circular_economy import CircularEconomy
from model_ippu import IPPU

"""
***********************************************************
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
***********************************************************

             FUNCTION_EVALUATION DECORATORS

***********************************************************
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
***********************************************************
"""


# *****************************
# ***********  AFOLU **********
# *****************************

def dec_func_eval_AFOLU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, params):

        # Get output data
        df_model_data_project = calibration.build_get_output_data_AFOLU(params)

        # Build performance for AFOLU            
        output = calibration.build_performance_AFOLU(df_model_data_project)
        
        return output
    return wrapper_decorator


# *****************************
# ****  CircularEconomy *******
# *****************************

def dec_func_eval_CircularEconomy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, params):

        # Get output data
        df_model_data_project = calibration.build_get_output_data_CircularEconomy(params)

        # Build performance for AFOLU            
        output = calibration.build_performance_CircularEconomy(df_model_data_project)
        
        return output
    return wrapper_decorator


# *****************************
# **********  IPPU ************
# *****************************

def dec_func_eval_IPPU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, params):
        
        # Get output data
        df_model_data_project = calibration.build_get_output_data_IPPU(params)

        # Build performance for AFOLU            
        output = calibration.build_performance_IPPU(df_model_data_project)
        

        return output
    return wrapper_decorator

# *****************************
# ***** NonElectricEnergy *****
# *****************************

def dec_func_eval_NonElectricEnergy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, params):
        
        """
        +++ 
        +++ UNDER CONSTRUCTION
        +++
        """

        return output
    return wrapper_decorator

# *****************************
# ******* ElectricEnergy ******
# *****************************

def dec_func_eval_ElectricEnergy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, params):

        """
        +++ 
        +++ UNDER CONSTRUCTION
        +++
        """

        return output
    return wrapper_decorator
