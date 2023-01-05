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
    def wrapper_decorator(calibration,params):

        df_input_data = calibration.df_input_var.copy()
        df_input_data = df_input_data.iloc[calibration.cv_training]

        agrupa = calibration.df_calib_bounds.groupby("group")
        group_list = calibration.df_calib_bounds["group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group == 0:
                index_var_group = calibration.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
                df_input_data[index_var_group] =  df_input_data[index_var_group]*params[:-(total_groups-1)]
                #df_input_data[index_var_group] = df_input_data[index_var_group].apply(lambda x: round(x, calibration.precition))
            index_var_group = calibration.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
            df_input_data[index_var_group] =  df_input_data[index_var_group]*params[group-total_groups]
            #df_input_data[index_var_group] = df_input_data[index_var_group].apply(lambda x: round(x, calibration.precition))

        agrupa = calibration.df_calib_bounds.groupby("norm_group")
        group_list = calibration.df_calib_bounds["norm_group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group != 0:
                pij_vars = calibration.df_calib_bounds["variable"].iloc[agrupa.groups[group]].to_list()
                total_grupo = df_input_data[pij_vars].sum(1)
                for pij_var_ind in pij_vars:
                    df_input_data[pij_var_ind] = df_input_data[pij_var_ind]/total_grupo
                    df_input_data[pij_var_ind] = df_input_data[pij_var_ind].apply(lambda x : round(x, calibration.precition))


        # Do something after
        return df_input_data
    return wrapper_decorator

def data_matrix_pij_AFOLU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration,params):

        df_input_data = calibration.df_input_var.copy()
        df_input_data = df_input_data.iloc[calibration.cv_training]

        agrupa = calibration.df_calib_bounds.query("group!=999").groupby("group")
        group_list = calibration.df_calib_bounds.query("group!=999")["group"].unique()
        total_groups = len(group_list)

        for group in group_list:
            group = int(group)
            if group == 0:
                index_var_group = calibration.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
                df_input_data[index_var_group] =  df_input_data[index_var_group]*params[:-(total_groups-1)-2]
                #df_input_data[index_var_group] = df_input_data[index_var_group].apply(lambda x: round(x, calibration.precition))
            index_var_group = calibration.df_calib_bounds["variable"].iloc[agrupa.groups[group]]
            df_input_data[index_var_group] =  df_input_data[index_var_group]*params[group-total_groups-2]
            #df_input_data[index_var_group] = df_input_data[index_var_group].apply(lambda x: round(x, calibration.precition))

        mixer = MixedLNDUTransitionFromBounds(eps = 0.0001)# eps is a correction threshold for transition matrices
        prop_pij = mixer.mix_transitions(params[-1],calibration.country)
        prop_pij = prop_pij.query(f"year> {calibration.year_init-3}").reset_index(drop=True)
        df_input_data[prop_pij.columns[1:]] = prop_pij[prop_pij.columns[1:]]
        # Do something after
        return df_input_data
    return wrapper_decorator



# *****************************
# ****  CircularEconomy *******
# *****************************


def data_CircularEconomy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration,params):

        # Copy original input data
        df_input_data = calibration.df_input_var.copy()
        # Get years in training set
        df_input_data = df_input_data.iloc[calibration.cv_training]
        # Update time period (values needs >= 0)
        df_input_data["time_period"] = range(df_input_data.shape[0])

        df_input_data[calibration.calib_targets["CircularEconomy"]] = df_input_data[calibration.calib_targets["CircularEconomy"]]*np.array(params)

        """
        vars_adecua = [i for i in df_input_data.columns if "frac_wali_ww_domestic_urban" in i]

        total = df_input_data[vars_adecua].sum(1)

        for var in vars_adecua:
            df_input_data[var] = df_input_data[var]/total
        """
        # Do something after
        return df_input_data
    return wrapper_decorator
