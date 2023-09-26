import warnings
warnings.filterwarnings("ignore")

import functools
import numpy as np
import pandas as pd
import math
import sqlalchemy

# Load LAC-Decarbonization source
import sys
import os

cwd = os.environ['LAC_PATH']
sys.path.append(os.path.join(cwd, 'python'))

import setup_analysis as sa
import support_functions as sf
import argparse

from model_socioeconomic import Socioeconomic
from model_afolu import AFOLU
from model_circular_economy import CircularEconomy
from model_ippu import IPPU
from model_energy import NonElectricEnergy
from model_electricity import ElectricEnergy

"""
***********************************************************
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
***********************************************************

             GET_OUTPUT_FROM_INPUT_DATA DECORATORS

***********************************************************
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
***********************************************************
"""


# *****************************
# ***********  AFOLU **********
# *****************************

def get_output_input_data_AFOLU_dec(func):
    @functools.wraps(func)
    def wrapper_decorator(df_input_data):
        
        model_afolu = AFOLU(sa.model_attributes)
        df_model_data_project = model_afolu.project(df_input_data)

        return df_model_data_project
    return wrapper_decorator


# *****************************
# *****  CircularEconomy ******
# *****************************

def get_output_input_data_CircularEconomy_dec(func):
    @functools.wraps(func)
    def wrapper_decorator(df_input_data):

        model_circecon = CircularEconomy(sa.model_attributes)
        df_output_data = model_circecon.project(df_input_data)

        return df_output_data
    return wrapper_decorator

# *****************************
# ***********  IPPU ***********
# *****************************

def get_output_input_data_IPPU_dec(func):
    @functools.wraps(func)
    def wrapper_decorator(df_input_data):

        model_ippu = IPPU(sa.model_attributes)
        df_output_data = model_ippu.project(df_input_data)

        return df_output_data
    return wrapper_decorator


# *****************************
# *****  NonElectricEnergy ****
# *****************************

def get_output_input_data_NonElectricEnergy_dec(func):
    @functools.wraps(func)
    def wrapper_decorator(df_input_data):
        try:
            model_energy = NonElectricEnergy(sa.model_attributes)                   
            df_output_data = model_energy.project(df_input_data)
        except:
            print("LOG ERROR HERE: CANNOT RUN WITHOUT IPPU")
            df_output_data = None

        return df_output_data
    return wrapper_decorator


# *****************************
# *****  ElectricEnergy *******
# *****************************


def get_output_input_data_ElectricEnergy_dec(func):
    @functools.wraps(func)
    def wrapper_decorator(df_input_data):

        try:
            model_elecricity = ElectricEnergy(sa.model_attributes, 
                            sa.dir_jl, 
                            sa.dir_ref_nemo)

            # create the engine
            engine = sqlalchemy.create_engine(f"sqlite:///{sa.fp_sqlite_nemomod_db_tmp}")

            # run model
            df_output_data = model_elecricity.project(df_input_data, engine)

        except:
            print("LOG ERROR HERE: CANNOT RUN WITHOUT IPPU AND AFOLU")
            df_output_data = None

        return df_output_data
    return wrapper_decorator




# *****************************************************************
# **** AllEnergy : Fugitive emissions from Non-Electric Energy ****
# *****************************************************************


def get_output_input_data_Fugitive_dec(func):
    @functools.wraps(func)
    def wrapper_decorator(df_input_data):

        try:
            model_energy = NonElectricEnergy(sa.model_attributes)
            df_output_data = model_energy.project(df_input_data, subsectors_project = sa.model_attributes.subsec_name_fgtv)
        except:        
            print("LOG ERROR HERE: CANNOT RUN WITHOUT IPPU AND AFOLU")

            df_output_data = None
        
        return df_output_data
    return wrapper_decorator


