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

             GET_CALIBRATED_DATA DECORATORS

***********************************************************
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
***********************************************************
"""

# *****************************
# ***********  AFOLU **********
# *****************************


def get_calibrated_data_AFOLU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, params, print_sector_model = False):
        # Copy original input data
        df_input_data = calibration.all_time_period_input_data.copy()

        # RUN AFOLU SECTOR
        if print_sector_model:
            print("RUN AFOLU SECTOR")
        
        # Build input data for AFOLU sector
        df_input_data = calibration.build_data_AFOLU(df_input_data, params)

        model_afolu = AFOLU(sa.model_attributes)
        df_model_data_project = model_afolu.project(df_input_data)        

        df_input_data = sf.merge_output_df_list([df_input_data, df_model_data_project], sa.model_attributes, "concatenate")

        return df_input_data
    return wrapper_decorator


# *****************************
# ****  CircularEconomy *******
# *****************************

def get_calibrated_data_CircularEconomy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, params, print_sector_model = False):

        # Copy original input data
        df_input_data = calibration.all_time_period_input_data.copy()       

        # RUN CircularEconomy
        if print_sector_model:
            print("RUN CircularEconomy SECTOR")

        # Build input data for CircularEconomy sector
        df_input_data = calibration.build_data_CircularEconomy(df_input_data, params)

        model_circular_economy = CircularEconomy(sa.model_attributes)
        df_model_data_project = model_circular_economy.project(df_input_data)

        df_input_data = sf.merge_output_df_list([df_input_data, df_model_data_project], sa.model_attributes, "concatenate")

        return df_input_data
    return wrapper_decorator


# *****************************
# **********  IPPU ************
# *****************************

def get_calibrated_data_IPPU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, params, print_sector_model = False):
        
        # Copy original input data
        df_input_data = calibration.all_time_period_input_data.copy()       

        # RUN IPPU
        if print_sector_model:
            print("RUN IPPU SECTOR")

        # Build input data for IPPU sector
        df_input_data = calibration.build_data_IPPU(df_input_data, params)
        
        model_ippu = IPPU(sa.model_attributes)
        df_model_data_project = model_ippu.project(df_input_data) 

        df_input_data = sf.merge_output_df_list([df_input_data, df_model_data_project], sa.model_attributes, "concatenate")

        return df_input_data
    return wrapper_decorator

# *****************************
# ***** NonElectricEnergy *****
# *****************************

def get_calibrated_data_NonElectricEnergy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, params, print_sector_model = False):
        
        # Copy original input data
        df_input_data = calibration.all_time_period_input_data.copy()        
        
        # RUN NonElectricEnergy SECTOR
        if print_sector_model:
            print("RUN NonElectricEnergy SECTOR") 

        # Build input data for NonElectricEnergy sector
        df_input_data = calibration.build_data_NonElectricEnergy(df_input_data, params)
        
        try:
            model_energy = sm.NonElectricEnergy(sa.model_attributes)                   
            df_model_data_project = model_energy.project(df_input_data)
        except:
            print("LOG ERROR HERE: CANNOT RUN WITHOUT IPPU")
            df_model_data_project = None

        df_input_data = sf.merge_output_df_list([df_input_data, df_model_data_project], sa.model_attributes, "concatenate")

        return df_input_data
    return wrapper_decorator

# *****************************
# ******* ElectricEnergy ******
# *****************************

def get_calibrated_data_ElectricEnergy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, params, print_sector_model = False):

        # Copy original input data
        df_input_data = calibration.all_time_period_input_data.copy()        

        # RUN ElectricEnergy SECTOR
        if print_sector_model:
            print("RUN ElectricEnergy SECTOR") 

        # Build input data for ElectricEnergy sector
        df_input_data = calibration.build_data_ElectricEnergy(df_input_data, params)
        
        try:
            model_elecricity = sm.ElectricEnergy(sa.model_attributes, 
                            sa.dir_jl, 
                            sa.dir_ref_nemo)

            # create the engine
            engine = sqlalchemy.create_engine(f"sqlite:///{sa.fp_sqlite_nemomod_db_tmp}")

            # run model
            df_model_data_project = model_elecricity.project(df_input_data, engine)

        except:
            print("LOG ERROR HERE: CANNOT RUN WITHOUT IPPU AND AFOLU")
            df_model_data_project = None

        df_input_data = sf.merge_output_df_list([df_input_data, df_model_data_project], sa.model_attributes, "concatenate")

        return df_input_data
    return wrapper_decorator


# *****************************
# ******* Fugitive ******
# *****************************

def get_calibrated_data_Fugitive(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, params, print_sector_model = False):

        # Copy original input data
        df_input_data = calibration.all_time_period_input_data.copy()        

        # RUN Energy SECTOR fugitive emissions from Non-Electric Energy
        if print_sector_model:
            print("RUN Energy SECTOR with fugitive emissions from Non-Electric Energy") 

        # Build input data for Fugitive sector
        df_input_data = calibration.build_data_Fugitive(df_input_data, params)
        
        try:
            model_energy = sm.NonElectricEnergy(sa.model_attributes)
            df_model_data_project = model_energy.project(df_input_data, subsectors_project = sa.model_attributes.subsec_name_fgtv)
        except:        
            print("LOG ERROR HERE: CANNOT RUN WITHOUT IPPU AND AFOLU")

            df_model_data_project = None

        df_input_data = sf.merge_output_df_list([df_input_data, df_model_data_project], sa.model_attributes, "concatenate")

        return df_input_data
    return wrapper_decorator