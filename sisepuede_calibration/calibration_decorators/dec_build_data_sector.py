import functools
import numpy as np
import pandas as pd
import math

# Load LAC-Decarbonization source
import sys
import os

cwd = os.environ['LAC_PATH']
sys.path.append(os.path.join(cwd, 'python'))

from data_functions_mix_lndu_transitions_from_inferred_bounds import MixedLNDUTransitionFromBounds


"""
***********************************************************
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
***********************************************************

             DECORATORS FOR BUILD DATA PER SECTOR

***********************************************************
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
***********************************************************
"""

# *****************************
# ***********  AFOLU **********
# *****************************


def data_AFOLU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, df_input_data, params):

        #df_input_data = df_input_data.iloc[calibration.cv_training]

        calib_bounds_AFOLU = calibration.df_calib_bounds_afolu.query("sector == 'AFOLU'").reset_index(drop = True)

        agrupa = calib_bounds_AFOLU.groupby("group")
        group_list = calib_bounds_AFOLU["group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group == 0:
                index_var_group = calib_bounds_AFOLU["variable"].iloc[agrupa.groups[group]]
                df_input_data[index_var_group] =  df_input_data[index_var_group]*params[:-(total_groups-1)]
                #df_input_data[index_var_group] = df_input_data[index_var_group].apply(lambda x: round(x, calibration.precition))
            index_var_group = calib_bounds_AFOLU["variable"].iloc[agrupa.groups[group]]
            df_input_data[index_var_group] =  df_input_data[index_var_group]*params[group-total_groups]
            #df_input_data[index_var_group] = df_input_data[index_var_group].apply(lambda x: round(x, calibration.precition))

        agrupa = calib_bounds_AFOLU.groupby("norm_group")
        group_list = calib_bounds_AFOLU["norm_group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group != 0:
                pij_vars = calib_bounds_AFOLU["variable"].iloc[agrupa.groups[group]].to_list()
                total_grupo = df_input_data[pij_vars].sum(1)
                for pij_var_ind in pij_vars:
                    df_input_data[pij_var_ind] = df_input_data[pij_var_ind]/total_grupo
                    df_input_data[pij_var_ind] = df_input_data[pij_var_ind].apply(lambda x : round(x, calibration.precition))


        # Do something after
        return df_input_data
    return wrapper_decorator

def data_matrix_pij_AFOLU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, df_input_data, params):

        #df_input_data = df_input_data.iloc[calibration.cv_training]

        calib_bounds_AFOLU = calibration.df_calib_bounds_afolu.query("sector == 'AFOLU'").reset_index(drop = True)

        agrupa = calib_bounds_AFOLU.query("group!=999").groupby("group")
        group_list = calib_bounds_AFOLU.query("group!=999")["group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group == 0:
                index_var_group = calib_bounds_AFOLU["variable"].iloc[agrupa.groups[group]]
                df_input_data[index_var_group] =  df_input_data[index_var_group]*params[:-(total_groups-1)-1]
                #df_input_data[index_var_group] = df_input_data[index_var_group].apply(lambda x: round(x, calibration.precition))
            index_var_group = calib_bounds_AFOLU["variable"].iloc[agrupa.groups[group]]
            df_input_data[index_var_group] =  df_input_data[index_var_group]*params[group-total_groups-1]
            #df_input_data[index_var_group] = df_input_data[index_var_group].apply(lambda x: round(x, calibration.precition))

        mixer = MixedLNDUTransitionFromBounds(eps = 0.0001)# eps is a correction threshold for transition matrices
        prop_pij = mixer.mix_transitions(params[-1],calibration.country)

        # SELECCIONAMOS LOS VALORES DE 2010 A 2015. Esto va a cambiar eventualmente. 
        # PARCHE PROVISIONAL
        #prop_pij = prop_pij.query(f"year> {calibration.year_init-3}").reset_index(drop=True)
        prop_pij = prop_pij.iloc[11:17].reset_index(drop=True)

        if df_input_data.shape[0] == prop_pij.shape[0]:
            df_input_data[prop_pij.columns[1:]] = prop_pij[prop_pij.columns[1:]]
        else:
            prop_pij_completa = prop_pij.copy()

            while not prop_pij_completa.shape[0] ==  df_input_data.shape[0]:
                ultimo_dato = pd.DataFrame({i:[j] for i,j in zip(prop_pij.iloc[-1].index, prop_pij.iloc[-1].values)}) 
                prop_pij_completa = pd.concat([prop_pij_completa, ultimo_dato], ignore_index = True)
            
            df_input_data[prop_pij_completa.columns[1:]] = prop_pij_completa[prop_pij_completa.columns[1:]]
            df_input_data = df_input_data.reset_index(drop = True)
        # Do something after
        return df_input_data
    return wrapper_decorator



# *****************************
# ****  CircularEconomy *******
# *****************************


def data_CircularEconomy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, df_input_data, params):

        # Get years in training set
        #df_input_data = df_input_data.iloc[calibration.cv_training]
        # Update time period (values needs >= 0)
        df_input_data["time_period"] = range(df_input_data.shape[0])

        df_input_data[calibration.calib_targets["CircularEconomy"]] = df_input_data[calibration.calib_targets["CircularEconomy"]]*np.array(params)

        # Do something after
        return df_input_data
    return wrapper_decorator


# *****************************
# **********  IPPU ************
# *****************************


def data_IPPU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, df_input_data, params):

        # Get years in training set
        #df_input_data = df_input_data.iloc[calibration.cv_training]
        # Update time period (values needs >= 0)
        df_input_data["time_period"] = range(df_input_data.shape[0])

        df_input_data[calibration.calib_targets["IPPU"]] = df_input_data[calibration.calib_targets["IPPU"]]*np.array(params)

        # Do something after
        return df_input_data
    return wrapper_decorator

# *****************************
# ***** NonElectricEnergy *****
# *****************************

def data_NonElectricEnergy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, df_input_data, params):

        """
        +++ 
        +++ UNDER CONSTRUCTION
        +++
        """

        return df_input_data
    return wrapper_decorator


# *****************************
# ******* ElectricEnergy ******
# *****************************

def data_ElectricEnergy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, df_input_data, params):

        """
        +++ 
        +++ UNDER CONSTRUCTION
        +++
        """
        
        return df_input_data
    return wrapper_decorator

# *****************************
# ******* AllEnergy ******
# *****************************

def data_AllEnergy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration, df_input_data, params):

        # Get years in training set
        #df_input_data = df_input_data.iloc[calibration.cv_training]
        # Update time period (values needs >= 0)
        df_input_data["time_period"] = range(df_input_data.shape[0])

        df_input_data[calibration.calib_targets["AllEnergy"]] = df_input_data[calibration.calib_targets["AllEnergy"]]*np.array(params)

        # Do something after
        return df_input_data
    return wrapper_decorator
