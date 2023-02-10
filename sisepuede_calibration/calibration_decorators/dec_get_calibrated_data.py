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

             GET_OUTPUT_DATA DECORATORS

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

        if calibration.run_integrated_q:
            # RUN AFOLU SECTOR
            if print_sector_model:
                print("RUN AFOLU SECTOR")

            df_output_data = []
            df_input_data = calibration.build_data_AFOLU(df_input_data, calibration.best_vector["AFOLU"])
            model_afolu = AFOLU(sa.model_attributes)
            df_output_data.append(model_afolu.project(df_input_data))

            # RUN CircularEconomy SECTOR
            if print_sector_model:
                print("RUN CircularEconomy SECTOR")

            model_circecon = sm.CircularEconomy(sa.model_attributes)

            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data[0],
                model_circecon.integration_variables
            )

            df_input_data = calibration.build_data_CircularEconomy(df_input_data, params)

            df_output_data.append(model_circecon.project(df_input_data))

            df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 

            # Build output data frame
            df_model_data_project = sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")

        else:
            # RUN CircularEconomy
            if print_sector_model:
                print("RUN CircularEconomy SECTOR")

            df_input_data = calibration.build_data_CircularEconomy(df_input_data, params)
            model_circular_economy = CircularEconomy(sa.model_attributes)
            df_model_data_project = model_circular_economy.project(df_input_data)


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

        if calibration.run_integrated_q:

             # RUN AFOLU SECTOR
            if print_sector_model:
                print("RUN AFOLU SECTOR")
            
            df_output_data = []
            df_input_data = calibration.build_data_AFOLU(df_input_data, calibration.best_vector["AFOLU"])
            model_afolu = AFOLU(sa.model_attributes)
            df_output_data.append(model_afolu.project(df_input_data))

            # RUN CircularEconomy SECTOR
            if print_sector_model:
                print("RUN CircularEconomy SECTOR")
            
            model_circecon = sm.CircularEconomy(sa.model_attributes)

            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data[0],
                model_circecon.integration_variables
            )

            df_input_data = calibration.build_data_CircularEconomy(df_input_data, calibration.best_vector["CircularEconomy"])

            df_output_data.append(model_circecon.project(df_input_data))

            df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 


            # RUN IPPU SECTOR
            if print_sector_model:
                print("RUN IPPU SECTOR")
            
            model_ippu = sm.IPPU(sa.model_attributes)

            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data[0],
                model_ippu.integration_variables
            )

            df_input_data = calibration.build_data_IPPU(df_input_data, params)

            df_output_data.append(model_ippu.project(df_input_data))
            df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 
    
            # Build output data frame
            df_model_data_project = sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")

           
        else:
        
            # RUN IPPU
            if print_sector_model:
                print("RUN IPPU SECTOR")

            df_input_data = calibration.build_data_IPPU(df_input_data, params)
            model_ippu = IPPU(sa.model_attributes)
            df_model_data_project = model_ippu.project(df_input_data) 


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
        
        # RUN AFOLU SECTOR
        if print_sector_model:
            print("RUN AFOLU SECTOR")
        
        df_output_data = []
        df_input_data = calibration.build_data_AFOLU(df_input_data, calibration.best_vector["AFOLU"])
        model_afolu = AFOLU(sa.model_attributes)
        df_output_data.append(model_afolu.project(df_input_data))

        # RUN CircularEconomy SECTOR
        if print_sector_model:
            print("RUN CircularEconomy SECTOR")
        
        model_circecon = sm.CircularEconomy(sa.model_attributes)

        df_input_data = sa.model_attributes.transfer_df_variables(
            df_input_data,
            df_output_data[0],
            model_circecon.integration_variables
        )

        df_input_data = calibration.build_data_CircularEconomy(df_input_data, calibration.best_vector["CircularEconomy"])

        df_output_data.append(model_circecon.project(df_input_data))

        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 


        # RUN IPPU SECTOR
        if print_sector_model:
            print("RUN IPPU SECTOR")
        
        model_ippu = sm.IPPU(sa.model_attributes)

        df_input_data = sa.model_attributes.transfer_df_variables(
            df_input_data,
            df_output_data[0],
            model_ippu.integration_variables
        )

        df_input_data = calibration.build_data_IPPU(df_input_data, calibration.best_vector["IPPU"])

        df_output_data.append(model_ippu.project(df_input_data))
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 


        # RUN NonElectricEnergy SECTOR
        if print_sector_model:
            print("RUN NonElectricEnergy SECTOR") 
        
        model_energy = sm.NonElectricEnergy(sa.model_attributes)

        df_input_data = sa.model_attributes.transfer_df_variables(
            df_input_data,
            df_output_data[0],
            model_energy.integration_variables_non_fgtv
        )

        df_output_data.append(model_energy.project(df_input_data))
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")]

        # Build output data frame
        df_model_data_project = sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")

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

        # RUN AFOLU SECTOR
        if print_sector_model:
            print("RUN AFOLU SECTOR")
        
        df_output_data = []
        df_input_data = calibration.build_data_AFOLU(df_input_data, calibration.best_vector["AFOLU"])
        model_afolu = AFOLU(sa.model_attributes)
        df_output_data.append(model_afolu.project(df_input_data))

        # RUN CircularEconomy SECTOR
        if print_sector_model:
            print("RUN CircularEconomy SECTOR")
        
        model_circecon = sm.CircularEconomy(sa.model_attributes)

        df_input_data = sa.model_attributes.transfer_df_variables(
            df_input_data,
            df_output_data[0],
            model_circecon.integration_variables
        )

        df_input_data = calibration.build_data_CircularEconomy(df_input_data, calibration.best_vector["CircularEconomy"])

        df_output_data.append(model_circecon.project(df_input_data))

        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 


        # RUN IPPU SECTOR
        if print_sector_model:
            print("RUN IPPU SECTOR")
        
        model_ippu = sm.IPPU(sa.model_attributes)

        df_input_data = sa.model_attributes.transfer_df_variables(
            df_input_data,
            df_output_data[0],
            model_ippu.integration_variables
        )

        df_input_data = calibration.build_data_IPPU(df_input_data, calibration.best_vector["IPPU"])

        df_output_data.append(model_ippu.project(df_input_data))
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 


        # RUN NonElectricEnergy SECTOR
        if print_sector_model:
            print("RUN NonElectricEnergy SECTOR") 
        
        model_energy = sm.NonElectricEnergy(sa.model_attributes)

        df_input_data = sa.model_attributes.transfer_df_variables(
            df_input_data,
            df_output_data[0],
            model_energy.integration_variables_non_fgtv
        )

        df_output_data.append(model_energy.project(df_input_data))
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")]

        # RUN ElectricEnergy SECTOR
        if print_sector_model:
            print("RUN ElectricEnergy SECTOR") 

        model_elecricity = sm.ElectricEnergy(sa.model_attributes, 
                                            sa.dir_jl, 
                                            sa.dir_ref_nemo)


        df_input_data = sa.model_attributes.transfer_df_variables(
            df_input_data,
            df_output_data[0],
            model_elecricity.integration_variables
        )

        # create the engine
        engine = sqlalchemy.create_engine(f"sqlite:///{sa.fp_sqlite_nemomod_db_tmp}")

        df_elec = model_elecricity.project(df_input_data, engine)
        df_output_data.append(df_elec)
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")]

        # Build output data frame
        df_model_data_project = sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")

        return df_input_data
    return wrapper_decorator


# *****************************
# ******* AllEnergy ******
# *****************************

def get_calibrated_data_AllEnergy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, params, print_sector_model = False):

        # Copy original input data
        df_input_data = calibration.all_time_period_input_data.copy()        

        # RUN AFOLU SECTOR
        if print_sector_model:
            print("RUN AFOLU SECTOR")
        
        df_output_data = []
        df_input_data = calibration.build_data_AFOLU(df_input_data, calibration.best_vector["AFOLU"])
        model_afolu = AFOLU(sa.model_attributes)
        df_output_data.append(model_afolu.project(df_input_data))

        # RUN CircularEconomy SECTOR
        if print_sector_model:
            print("RUN CircularEconomy SECTOR")
        
        model_circecon = sm.CircularEconomy(sa.model_attributes)

        df_input_data = sa.model_attributes.transfer_df_variables(
            df_input_data,
            df_output_data[0],
            model_circecon.integration_variables
        )

        df_input_data = calibration.build_data_CircularEconomy(df_input_data, calibration.best_vector["CircularEconomy"])

        df_output_data.append(model_circecon.project(df_input_data))

        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 


        # RUN IPPU SECTOR
        if print_sector_model:
            print("RUN IPPU SECTOR")
        
        model_ippu = sm.IPPU(sa.model_attributes)

        df_input_data = sa.model_attributes.transfer_df_variables(
            df_input_data,
            df_output_data[0],
            model_ippu.integration_variables
        )

        df_input_data = calibration.build_data_IPPU(df_input_data, calibration.best_vector["IPPU"])

        df_output_data.append(model_ippu.project(df_input_data))
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 


        # RUN NonElectricEnergy SECTOR
        if print_sector_model:
            print("RUN NonElectricEnergy SECTOR") 
        
        model_energy = sm.NonElectricEnergy(sa.model_attributes)

        df_input_data = sa.model_attributes.transfer_df_variables(
            df_input_data,
            df_output_data[0],
            model_energy.integration_variables_non_fgtv
        )

        df_input_data = calibration.build_data_AllEnergy(df_input_data, params)

        df_output_data.append(model_energy.project(df_input_data))
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")]

        # RUN ElectricEnergy SECTOR
        if print_sector_model:
            print("RUN ElectricEnergy SECTOR") 

        model_elecricity = sm.ElectricEnergy(sa.model_attributes, 
                                            sa.dir_jl, 
                                            sa.dir_ref_nemo)


        df_input_data = sa.model_attributes.transfer_df_variables(
            df_input_data,
            df_output_data[0],
            model_elecricity.integration_variables
        )

        # create the engine
        engine = sqlalchemy.create_engine(f"sqlite:///{sa.fp_sqlite_nemomod_db_tmp}")

        df_elec = model_elecricity.project(df_input_data, engine)
        df_output_data.append(df_elec)
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")]

        # RUN Energy SECTOR fugitive emissions from Non-Electric Energy
        if print_sector_model:
            print("RUN Energy SECTOR with fugitive emissions from Non-Electric Energy") 

        model_energy = sm.NonElectricEnergy(sa.model_attributes)

        df_input_data = sa.model_attributes.transfer_df_variables(
            df_input_data,
            df_output_data[0],
            model_energy.integration_variables_fgtv
        )

        df_output_data.append(model_energy.project(df_input_data, subsectors_project = sa.model_attributes.subsec_name_fgtv))
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 

        # Build output data frame
        df_model_data_project = sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")

        return df_input_data
    return wrapper_decorator