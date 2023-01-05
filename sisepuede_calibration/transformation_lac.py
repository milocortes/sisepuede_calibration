import random
import math
import numpy as np
import time
import pandas as pd

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



class Transformation:
    '''
    ------------------------------------------

    Transformation class

    -------------------------------------------

    # Inputs:
        * df_calib_input_var        - Pandas DataFrame of input calibrated variables
        * country                   - Country of calibration
        * subsector_model           - Subsector model
        * dict_tranformation_files  - Dictionary with necesarly transformation files

    '''

    def __init__(self, df_calib_input_var, country, subsector_model, dict_tranformation_files):
        self.df_data_calib = df_calib_input_var
        self.country = country
        self.subsector_model = subsector_model
        self.dict_tranformation_files = dict_tranformation_files  



    def run_transformations(self):
        # Raw input file
        df_input_data = pd.read_csv(self.dict_tranformation_files["df_input_data"])

        # Combinations of transformations to be run
        df_exp = pd.read_csv(self.dict_tranformation_files["df_exp"])

        # Waste reduction
        df_wr = pd.read_csv(self.dict_tranformation_files["waste_reduction"])

        # Liquid waste-sanitation and treatment 
        df_st = pd.read_csv(self.dict_tranformation_files["sanitation"])

        # Solid waste-sanitation and treatment 
        df_sot = pd.read_csv(self.dict_tranformation_files["solid"])


        #list all policies
        policies = np.unique(df_exp['Policy_ID'])

        df_all = []

        for policy in policies:
            print("+++++++++++++++++++++++++")
            print(f"           {policy}     ")
            print("+++++++++++++++++++++++++")
            #load transformation tables
            tt = list(df_exp[df_exp['Policy_ID']==policy]['waste_reduction'])[0]
            st = list(df_exp[df_exp['Policy_ID']==policy]['sanitation'])[0]
            sot = list(df_exp[df_exp['Policy_ID']==policy]['solid'])[0]

            df_wrp = df_wr[df_wr['TransformationName']==tt].reset_index(drop = True)
            df_stp = df_st[df_st['TransformationName']==st].reset_index(drop = True)
            df_sotp = df_sot[df_sot['TransformationName']==sot].reset_index(drop = True)


            df_input_new = df_input_data.copy()
            df_dcp = self.df_data_calib.copy()

            df_input_new = df_input_new.reset_index(drop = True)
            df_dcp = df_dcp.reset_index(drop = True)
            #now substitue values in df_dcp en df_input_data
            #do all columns names are in in the input file
            #subset columns names that we need to replace
            target_vars = df_dcp.columns[df_dcp.columns.isin(df_input_new.columns)][1:]
            #print(df_input_data[target_vars[0]])
            #print(df_input_new[target_vars[0]])
            
            df_input_new[target_vars] = df_dcp[target_vars].copy()

            #print(df_input_data[target_vars[0]])
            #print(df_input_new[target_vars[0]])
            #now modify input table to reflect transformations impact

            #waste reduction
            target_Tvars = df_wrp.columns[df_wrp.columns.isin(df_input_new.columns)][1:]
            
            #j= 'factor_waso_waste_per_capita_scalar_metal'
            df_input_new[target_Tvars] = df_wrp[target_Tvars].copy()

            #sanitation
            target_Svars = df_stp.columns[df_stp.columns.isin(df_input_new.columns)][1:]
    
            #j= 'frac_wali_ww_domestic_rural_treatment_path_advanced_aerobic'
            df_input_new[target_Svars] = df_stp[target_Svars].copy()

            #solid waste
            target_SOvars = df_sotp.columns[df_sotp.columns.isin(df_input_new.columns)][1:]
            
            #j= 'factor_waso_waste_per_capita_scalar_metal'
            df_input_new[target_SOvars] = df_sotp[target_SOvars].copy()

            #now we used df_input_new to run the model
            #set initial attributes of model circular economy
            model_circecon = CircularEconomy(sa.model_attributes)
            #run the model
            df_output = model_circecon.project(df_input_new)
            #add ids
            df_output['policy'] = policy
            df_all.append(df_output)
    
        #save final table with results
        df_all = pd.concat(df_all, ignore_index = True)

        return df_all
