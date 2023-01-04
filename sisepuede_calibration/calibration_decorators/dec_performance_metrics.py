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

        if calibration.cv_calibration:
            """
            co2_df_total = np.array([co2_df_total[i] for i in calibration.cv_training])
            output = np.mean(co2_df_total)
            """
            co2_df_total = [co2_df_total[i] for i in calibration.cv_training]
            output = np.sum(co2_df_total)

        else:
            output = np.sum(co2_df_total)
            """
            if any((np.mean(calibration.percent_diff[["5058","6750"]]).to_numpy()>30).flatten()):
                return output + 10000000
            else:
            """

        # Do something after
        return output
    return wrapper_decorator


# *****************************
# ****  CircularEconomy *******
# *****************************


def performance_CircularEconomy(func):
    @functools.wraps(func)
    def wrapper_decorator(calibration,df_model_data_project):

        subsectors = ['emission_co2e_subsector_total_wali','emission_co2e_subsector_total_waso', 'emission_co2e_subsector_total_trww']

        co2_df_total = np.array(df_model_data_project[subsectors].sum(1))
        co2_historical = np.array(calibration.df_co2_emissions["value"].tolist())*(1/1000)

        if calibration.cv_calibration:

            output = np.mean([(co2_df_total[i] - co2_historical[i])**2 for i in calibration.cv_training])
            
        else:
            output = np.mean(np.mean(( co2_df_total - co2_historical )**2))

        return output
    return wrapper_decorator

