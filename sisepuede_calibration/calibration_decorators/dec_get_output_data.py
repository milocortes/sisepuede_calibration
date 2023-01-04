import functools
import numpy as np
import pandas as pd
import math



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


def get_output_data_AFOLU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration,df_model_data_project):
        item_val_afolu = {}
        item_val_afolu_total_item_fao = {}
        item_val_afolu_total_item_fao_observado = {}

        item_val_afolu_percent_diff = {}

        acumula_total = (calibration.df_co2_emissions.groupby(["Area_Code","Year"]).sum().reset_index(drop=True).Value/1000).to_numpy()

        for item, vars in calibration.var_co2_emissions_by_sector[calibration.subsector_model].items():
            if vars:
                item_val_afolu_total_item_fao[item] = df_model_data_project[vars].sum(1)  
                item_val_afolu_total_item_fao_observado[item] = (calibration.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000
                item_val_afolu[item] = (item_val_afolu_total_item_fao[item] - item_val_afolu_total_item_fao_observado[item])**2
                #item_val_afolu_percent_diff[item] = (df_model_data_project[vars].sum(1) / ((calibration.df_co2_emissions.query("Item_Code=={}".format(item)).drop_duplicates(subset='Year', keep="first").reset_index().Value)/1000))*100
                item_val_afolu_percent_diff[item] = ((item_val_afolu_total_item_fao[item] - item_val_afolu_total_item_fao_observado[item])/item_val_afolu_total_item_fao_observado[item])*100

        co2_df = pd.DataFrame(item_val_afolu)
        co2_df_percent_diff = pd.DataFrame(item_val_afolu_percent_diff)
        calibration.percent_diff = co2_df_percent_diff
        calibration.error_by_item = co2_df
        calibration.item_val_afolu_total_item_fao = item_val_afolu_total_item_fao
        co2_df_total = co2_df.sum(1)
    return wrapper_decorator


# *****************************
# ****  CircularEconomy *******
# *****************************

def get_output_data_CircularEconomy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration,df_model_data_project):
        print("get_output_data_CircularEconomy")
    return wrapper_decorator