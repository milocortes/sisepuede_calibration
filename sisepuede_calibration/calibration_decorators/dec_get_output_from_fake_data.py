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
import sector_models as sm
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

             GET_OUTPUT_FROM_FAKE_DATA DECORATORS

***********************************************************
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
***********************************************************
"""


# *****************************
# ***********  AFOLU **********
# *****************************

def get_output_fake_data_AFOLU_dec(func):
    @functools.wraps(func)
    def wrapper_decorator(df_input_data, run_integrated_q):

            if run_integrated_q:
                model_afolu = AFOLU(sa.model_attributes)
                df_model_data_project = model_afolu.project(df_input_data)
            else:
                model_afolu = AFOLU(sa.model_attributes)
                df_model_data_project = model_afolu.project(df_input_data)

            return df_model_data_project
    return wrapper_decorator


# *****************************
# *****  CircularEconomy ******
# *****************************

def get_output_fake_data_CircularEconomy_dec(func):
    @functools.wraps(func)
    def wrapper_decorator(df_input_data, run_integrated_q):

            if run_integrated_q:
                df_output_data = []
                print("\n\tRunning AFOLU")
                # get the model, run it using the input data, then update the output data (for integration)
                model_afolu = sm.AFOLU(sa.model_attributes)
                df_output_data.append(model_afolu.project(df_input_data))

                print("\n\tRunning CircularEconomy")
                model_circecon = sm.CircularEconomy(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_output_data[0],
                    model_circecon.integration_variables
                )

                df_output_data.append(model_circecon.project(df_input_data))

                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 

                # build output data frame
                df_output_data = sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")

            else:
                model_circecon = sm.CircularEconomy(sa.model_attributes)
                df_output_data = model_circecon.project(df_input_data)

            return df_output_data
    return wrapper_decorator

# *****************************
# ***********  IPPU ***********
# *****************************

def get_output_fake_data_IPPU_dec(func):
    @functools.wraps(func)
    def wrapper_decorator(df_input_data, run_integrated_q):

            if run_integrated_q:
                df_output_data = []
                print("\n\tRunning AFOLU")
                model_afolu = sm.AFOLU(sa.model_attributes)
                df_output_data.append(model_afolu.project(df_input_data))

                print("\n\tRunning CircularEconomy")
                model_circecon = sm.CircularEconomy(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_output_data[0],
                    model_circecon.integration_variables
                )

                df_output_data.append(model_circecon.project(df_input_data))

                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 

                print("\n\tRunning IPPU")
                model_ippu = sm.IPPU(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_output_data[0],
                    model_ippu.integration_variables
                )

                df_output_data.append(model_ippu.project(df_input_data))
                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 
     
                # build output data frame
                df_output_data = sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")

            else:
                model_ippu = sm.IPPU(sa.model_attributes)
                df_output_data = model_ippu.project(df_input_data)

            return df_output_data
    return wrapper_decorator


# *****************************
# *****  NonElectricEnergy ****
# *****************************

def get_output_fake_data_NonElectricEnergy_dec(func):
    @functools.wraps(func)
    def wrapper_decorator(df_input_data, run_integrated_q):

            if run_integrated_q:
                df_output_data = []
                print("\n\tRunning AFOLU")
                model_afolu = sm.AFOLU(sa.model_attributes)
                df_output_data.append(model_afolu.project(df_input_data))

                print("\n\tRunning CircularEconomy")
                model_circecon = sm.CircularEconomy(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_output_data[0],
                    model_circecon.integration_variables
                )

                df_output_data.append(model_circecon.project(df_input_data))

                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 

                print("\n\tRunning IPPU")
                model_ippu = sm.IPPU(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_output_data[0],
                    model_ippu.integration_variables
                )

                df_output_data.append(model_ippu.project(df_input_data))
                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 
        
                print("\n\tRunning NonElectricEnergy")
                model_energy = sm.NonElectricEnergy(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_output_data[0],
                    model_energy.integration_variables_non_fgtv
                )

                df_output_data.append(model_energy.project(df_input_data))
                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 

                # build output data frame
                df_output_data = sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")

            else:
                print("LOG ERROR HERE: CANNOT RUN WITHOUT IPPU")

            return df_output_data
    return wrapper_decorator


# *****************************
# *****  ElectricEnergy *******
# *****************************


def get_output_fake_data_ElectricEnergy_dec(func):
    @functools.wraps(func)
    def wrapper_decorator(df_input_data, run_integrated_q):

            if run_integrated_q:
                df_output_data = []
                print("\n\tRunning AFOLU")
                model_afolu = sm.AFOLU(sa.model_attributes)
                df_output_data.append(model_afolu.project(df_input_data))

                print("\n\tRunning CircularEconomy")
                model_circecon = sm.CircularEconomy(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_output_data[0],
                    model_circecon.integration_variables
                )

                df_output_data.append(model_circecon.project(df_input_data))

                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 

                print("\n\tRunning IPPU")
                model_ippu = sm.IPPU(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_output_data[0],
                    model_ippu.integration_variables
                )

                df_output_data.append(model_ippu.project(df_input_data))
                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")]
        
                print("\n\tRunning NonElectricEnergy")
                model_energy = sm.NonElectricEnergy(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_output_data[0],
                    model_energy.integration_variables_non_fgtv
                )

                df_output_data.append(model_energy.project(df_input_data))
                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 

                print("\n\tRunning ElectricEnergy")
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

                df_elec =  model_elecricity.project(df_input_data, engine)
                df_output_data.append(df_elec)
                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")]
                    
                # build output data frame
                df_output_data = sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")

            else:
                print("LOG ERROR HERE: CANNOT RUN WITHOUT IPPU AND AFOLU")

            return df_output_data
    return wrapper_decorator




# *****************************************************
# **** Fugitive emissions from Non-Electric Energy ****
# *****************************************************


def get_output_fake_data_FugitiveNonElectricEnergy_dec(func):
    @functools.wraps(func)
    def wrapper_decorator(df_input_data, run_integrated_q):

            if run_integrated_q:
                df_output_data = []
                print("\n\tRunning AFOLU")
                model_afolu = sm.AFOLU(sa.model_attributes)
                df_output_data.append(model_afolu.project(df_input_data))

                print("\n\tRunning CircularEconomy")
                model_circecon = sm.CircularEconomy(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_output_data[0],
                    model_circecon.integration_variables
                )

                df_output_data.append(model_circecon.project(df_input_data))

                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 

                print("\n\tRunning IPPU")
                model_ippu = sm.IPPU(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_output_data[0],
                    model_ippu.integration_variables
                )

                df_output_data.append(model_ippu.project(df_input_data))
                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")]
        
                print("\n\tRunning NonElectricEnergy")
                model_energy = sm.NonElectricEnergy(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_output_data[0],
                    model_energy.integration_variables_non_fgtv
                )

                df_output_data.append(model_energy.project(df_input_data))
                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] 

                print("\n\tRunning ElectricEnergy")
                model_elecricity = sm.ElectricEnergy(sa.model_attributes, sa.dir_ref_nemo)

                df_input_data = sa.model_attributes.transfer_df_variables(
                        df_input_data,
                        df_output_data[0],
                        model_elecricity.integration_variables
                    )

                # create the engine
                engine = sqlalchemy.create_engine(f"sqlite:///{sa.fp_sqlite_nemomod_db_tmp}")

                df_elec =  model_elecricity.project(df_input_data, engine)
                df_output_data.append(df_elec)
                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")]
        
                print("\n\tRunning NonElectricEnergy - Fugitive Emissions")
                model_energy = sm.NonElectricEnergy(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                    df_input_data,
                    df_output_data[0],
                    model_energy.integration_variables_fgtv
                )

                df_output_data.append(model_energy.project(df_input_data, subsectors_project = sa.model_attributes.subsec_name_fgtv))
                df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] if run_integrated_q else df_output_data

                # build output data frame
                df_output_data = sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")

            else:
                print("LOG ERROR HERE: CANNOT RUN WITHOUT IPPU AND AFOLU")

            return df_output_data
    return wrapper_decorator


