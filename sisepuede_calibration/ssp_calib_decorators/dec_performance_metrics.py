import functools
import numpy as np
import pandas as pd
import math



"""
***********************************************************
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
***********************************************************

            PERFORMANCE METRICS DECORATORS 

***********************************************************
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
***********************************************************
"""

# *****************************
# ***********  AFOLU **********
# *****************************

def performance_AFOLU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration,df_model_data_project):
        #cycle, trend = statsm.tsa.filters.hpfilter((calib["value"].values/1000), 1600)
        #trend = calibration.df_co2_emissions.value
        #trend = [i/1000 for i in trend]

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

        co2_df_observado = pd.DataFrame(item_val_afolu_total_item_fao_observado)

        ponderadores = (co2_df_observado.mean().abs()/co2_df_observado.mean().abs().sum()).apply(math.exp)   
        co2_df_total = (ponderadores*co2_df).sum(1)


        output = np.sum(co2_df_total)

        # Do something after
        return output
    return wrapper_decorator


# *****************************
# ****  CircularEconomy *******
# *****************************


def performance_CircularEconomy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration,df_model_data_project):

        #co2_df_total = np.array(df_model_data_project[calibration.var_co2_emissions_by_sector["CircularEconomy"]].sum(1)) - np.array(df_model_data_project["emission_co2e_co2_waso_incineration"])
        co2_df_total = np.array(df_model_data_project[calibration.var_co2_emissions_by_sector["CircularEconomy"]].sum(1))
        co2_historical = np.array(calibration.df_co2_emissions["value"].tolist())*(1/1000)

        output = np.mean(np.mean(( co2_df_total - co2_historical )**2))

        return output
    return wrapper_decorator


# *****************************
# ***********  IPPU ***********
# *****************************


def performance_IPPU(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration,df_model_data_project):

        co2_df_total = np.array(df_model_data_project[calibration.var_co2_emissions_by_sector["IPPU"]].sum(1))
        co2_historical = np.array(calibration.df_co2_emissions["value"].tolist())*(1/1000)

        output = np.mean(np.mean(( co2_df_total - co2_historical )**2))

        return output
    return wrapper_decorator


# *****************************
# ******  AllEnergy ***********
# *****************************

def performance_AllEnergy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration,df_model_data_project):

        energy_crosswalk_estimado = {}
        energy_crosswalk_observado = {}
        energy_crosswalk_error = {}

        for subsector, sisepuede_vars in calibration.var_co2_emissions_by_sector["AllEnergy"].items():
            energy_crosswalk_estimado[subsector] = df_model_data_project[sisepuede_vars].sum(1).reset_index(drop = True) 
            energy_crosswalk_observado[subsector] = calibration.df_co2_emissions.query(f"subsector_sisepuede == '{subsector}'")[["value"]].sum(1).reset_index(drop = True)
            energy_crosswalk_error[subsector] = (energy_crosswalk_estimado[subsector] - energy_crosswalk_observado[subsector])**2

        co2_df = pd.DataFrame(energy_crosswalk_error)
        calibration.error_by_item = co2_df
        calibration.error_by_sector_energy = energy_crosswalk_error
        co2_df_total = co2_df.sum(1)

        co2_df_observado = pd.DataFrame(energy_crosswalk_observado)

        ponderadores = (co2_df_observado.mean().abs()/co2_df_observado.mean().abs().sum()).apply(math.exp)   
        co2_df_total = (ponderadores*co2_df).sum(1)

        output = np.sum(co2_df_total)
            
        return output
    return wrapper_decorator

