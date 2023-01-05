import random
import math
import numpy as np
import time
import pandas as pd

# Load Optimization Algorithms source
from sisepuede_calibration.optimization_algorithms import *

# Load Decorators for calibration process
from sisepuede_calibration.calibration_decorators.calib_decorators import *

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

class RunModel:
    '''
    ------------------------------------------

    RunModel class

    -------------------------------------------

    # Inputs:
        * df_input_var              - Pandas DataFrame of input variables
        * country                   - Country of calibration
        * calib_targets             - Variables that will be calibrated
        * t_times                   - List with the period of simulation
        * subsector_model           - Subsector model
        * downstream                - Flag that says if the model is running integrated (i.e. with the output of latest subsector model )
        * best_vector

    '''

    def __init__(self, year_init, year_end, df_input_var, country, subsector_model, calib_targets,df_calib_bounds,all_time_period_input_data = None,downstream = False):
        self.year_init = year_init 
        self.year_end = year_end
        self.df_input_var = df_input_var
        self.country = country
        self.calib_targets = {}
        self.calib_targets[subsector_model] = calib_targets
        self.subsector_model = subsector_model
        self.df_calib_bounds = df_calib_bounds
        self.downstream = downstream
        self.best_vector = {'AFOLU' : None, 'CircularEconomy' : None, 'IPPU': None}
        self.all_time_period_input_data = all_time_period_input_data


    """
    ---------------------------------
    set_calib_targets method
    ---------------------------------

    Description: The method receive the subsector model to set the calibration targets

    # Inputs:
             * subsector_model              - Subsector model name
             * calib_targets_model          - Calibration targets
    """
    def set_calib_targets(self,subsector_model,calib_targets_model):
        self.calib_targets[subsector_model] = calib_targets_model



    """
    ---------------------------------
    set_best_vector method
    ---------------------------------

    Description: The method receive the subsector model to set the best vector found in
                 the calibration

    # Inputs:
             * subsector_model          - Subsector model name
             * vector                   - vector
    """
    def set_best_vector(self,subsector_model,vector):
        self.best_vector[subsector_model] = vector

    # +++++++++++++++++++++++++++++++++++++++
    # Set decorators for build data by sector
    # +++++++++++++++++++++++++++++++++++++++

    @data_matrix_pij_AFOLU
    def build_data_AFOLU(self, params):
        pass

    @data_CircularEconomy
    def build_data_CircularEconomy(self, params):
        pass

    @data_CircularEconomy_all_period
    def build_data_CircularEconomy_all_period(self, params):
        pass

    # +++++++++++++++++++++++++++++++++++++++
    # Set decorators for get output data by sector
    # +++++++++++++++++++++++++++++++++++++++

    @get_output_data_AFOLU
    def build_get_output_data_AFOLU(self, df_model_data_project):
        pass

    @get_output_data_CircularEconomy
    def build_get_output_data_CircularEconomy(self, df_model_data_project):
        pass


    """
    ---------------------------------
    get_output_data method
    ---------------------------------

    Description: The method recive params, run the subsector model,
                 and compute output data of specific subsector model

    # Inputs:
             * params                   - calibration vector

    # Output:
             * df_model_data_project    - output data of specific subsector model
    """

    def get_output_data(self, params):


        if self.subsector_model == "AFOLU":
            #print("\n\tAFOLU")
            df_input_data = self.build_data_AFOLU(params)
            model_afolu = AFOLU(sa.model_attributes)
            df_model_data_project = model_afolu.project(df_input_data)
            self.build_get_output_data_AFOLU(df_model_data_project)
            
        if self.subsector_model == "CircularEconomy":
            #print("\n\tRunning CircularEconomy")
            df_input_data = self.build_data_CircularEconomy(params)
            model_circular_economy = CircularEconomy(sa.model_attributes)
            df_model_data_project = model_circular_economy.project(df_input_data)

        if self.subsector_model == "IPPU":
            if not self.downstream:
                #print("\n\tRunning IPPU")
                model_ippu = IPPU(sa.model_attributes)
                df_model_data_project = model_ippu.project(df_input_data)
            else:
                # First run CircularEconomy model
                model_circular_economy = CircularEconomy(sa.model_attributes)
                df_input_data = self.df_input_var.copy()
                df_input_data[self.calib_targets["CircularEconomy"]] = df_input_data[self.calib_targets["CircularEconomy"]] * self.best_vector["CircularEconomy"]
                output_circular_economy = model_circular_economy.project(df_input_data)

                # Build the new df_input_data with the output of CircularEconomy model
                model_ippu = IPPU(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                                df_input_data,
                                output_circular_economy,
                                model_ippu.integration_variables
                            )
                # Run IPPU model
                df_input_data[var_no_change_over_time] = df_input_data[var_no_change_over_time].mean()*np.array(params)[index_var_no_change_over_time]

                if list(var_change_over_time):
                    df_input_data[var_change_over_time] = df_input_data[var_change_over_time]*np.array(params)[index_var_change_over_time]

                df_model_data_project = model_ippu.project(df_input_data)

        if self.subsector_model == "ElectricEnergy":
            model_afolu = sm.AFOLU(sa.model_attributes)
            df_output_data = model_afolu.project(df_input_data)


            model_ippu = sm.IPPU(sa.model_attributes)

            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data,
                model_ippu.integration_variables
            )

            df_ippu_out = model_ippu.project(df_input_data)

            df_output_data = pd.merge(df_ippu_out, df_output_data)

            model_energy = sm.NonElectricEnergy(sa.model_attributes)

            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data,
                model_energy.integration_variables_non_fgtv
            )
                        
            df_output_data = model_energy.project(df_input_data)
                        

        return df_model_data_project

    """
    ---------------------------------
    build_calibrated_data method
    ---------------------------------

    Description: The method recive params and build the calibrated
                 data of specific subsector model.

    # Inputs:
             * params                   - calibration vector
             * df_input_data            - fake data complete

    # Output:
             * df_calibrated_data       - calibrated data
    """

    def get_output_data_from_df_input_data(self, df_input_data):

        if self.subsector_model == "AFOLU":
            #print("\n\tAFOLU")
            model_afolu = AFOLU(sa.model_attributes)
            df_model_data_project = model_afolu.project(df_input_data)
            
        if self.subsector_model == "CircularEconomy":
            #print("\n\tRunning CircularEconomy")
            model_circular_economy = CircularEconomy(sa.model_attributes)
            df_model_data_project = model_circular_economy.project(df_input_data)

                        
        return df_model_data_project


class CalibrationModel(RunModel):
    '''
    ------------------------------------------

    CalibrationModel class

    -------------------------------------------

    # Inputs:
        * df_input_var              - Pandas DataFrame of input variables
        * country                   - Country of calibration
        * calib_targets             - Variables that will be calibrated
        * df_calib_bounds           - Pandas DataFrame with the calibration scalar ranges
        * t_times                   - List with the period of simulation
        * df_co2_emissions          - Pandas DataFrame of co2_emissions
        * subsector_model           - Subsector model
        * cv_calibration            - Flag that says if the calibration runs cross validation
        * cv_training               - List with the periods of the training set
        * cv_test                   - List with the periods of the test set
        * downstream                - Flag that says if the model is running integrated (i.e. with the output of latest subsector model )

    # Output
        * mse_all_period            - Mean Squared Error for all periods
        * mse_training              - Mean Squared Error for training period
        * mse_test                  - Mean Squared Error for test period
        * calib_vector              - List of calibration scalar
    '''

    def __init__(self,  year_init, year_end, df_input_var, country, subsector_model, calib_targets, df_calib_bounds, all_time_period_input_data,
                 df_co2_emissions, co2_emissions_by_sector = {}, cv_calibration = False, cv_training = [], cv_test = [], cv_run = 0, id_mpi = 0,downstream = False,weight_co2_flag = False, weight_co2 = [],precition = 6):
        super(CalibrationModel, self).__init__(year_init, year_end, df_input_var, country, subsector_model, calib_targets,df_calib_bounds,all_time_period_input_data,downstream = False)
        self.df_co2_emissions = df_co2_emissions
        self.cv_calibration = cv_calibration
        self.cv_training = cv_training
        self.cv_test = cv_test
        self.var_co2_emissions_by_sector = {'CircularEconomy' : ["emission_co2e_subsector_total_wali","emission_co2e_subsector_total_waso","emission_co2e_subsector_total_trww"],
                                            'IPPU': ['emission_co2e_subsector_total_ippu'],
                                            'AFOLU' : co2_emissions_by_sector}
        self.cv_run = cv_run
        self.id_mpi = id_mpi
        self.fitness_values = {'AFOLU' : [], 'CircularEconomy' : [], 'IPPU' : []}
        self.weight_co2_flag = weight_co2_flag
        self.weight_co2 = np.array(weight_co2)
        self.item_val_afolu_total_item_fao = None
        self.precition = precition

    def get_calib_var_group(self,grupo): 
        return self.df_calib_bounds.query("group =={}".format(grupo))["variable"].to_list()

    # +++++++++++++++++++++++++++++++++++++++++++++++++
    # Set decorators for performance metrics by sector
    # +++++++++++++++++++++++++++++++++++++++++++++++++

    @performance_AFOLU
    def build_performance_AFOLU(self, df_model_data_project):
        pass

    @performance_CircularEconomy
    def build_performance_CircularEconomy(self, df_model_data_project):
        pass

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Set decorators for performance metrics in test set by sector
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @performance_test_AFOLU
    def build_performance_test_AFOLU(self, df_model_data_project):
        pass

    @performance_test_CircularEconomy
    def build_performance_test_CircularEconomy(self, df_model_data_project):
        pass

    """
    ---------------------------------
    objective_a method
    ---------------------------------

    Description: The method recive params, run the subsector model,
                 and compute the MSE by the specific period of simulation

    # Inputs:
             * params       - calibration vector

    # Output:
             * output       - MSE value

    """
    def objective_a(self, params):

        if self.subsector_model == "AFOLU":
            #print("\n\tAFOLU")
            df_input_data = self.build_data_AFOLU(params)
            model_afolu = AFOLU(sa.model_attributes)
            df_model_data_project = model_afolu.project(df_input_data)
            output = self.build_performance_AFOLU(df_model_data_project)
        if self.subsector_model == "CircularEconomy":
            #print("\n\tRunning CircularEconomy")
            df_input_data = self.build_data_CircularEconomy(params)
            model_circular_economy = CircularEconomy(sa.model_attributes)
            df_model_data_project = model_circular_economy.project(df_input_data)
            output = self.build_performance_CircularEconomy(df_model_data_project)

        if self.subsector_model == "IPPU":
            if not self.downstream:
                #print("\n\tRunning IPPU")
                model_ippu = IPPU(sa.model_attributes)
                df_model_data_project = model_ippu.project(df_input_data)
            else:
                # First run CircularEconomy model
                model_circular_economy = CircularEconomy(sa.model_attributes)
                df_input_data = self.df_input_var.copy()
                df_input_data[self.calib_targets["CircularEconomy"]] = df_input_data[self.calib_targets["CircularEconomy"]] * self.best_vector["CircularEconomy"]
                output_circular_economy = model_circular_economy.project(df_input_data)

                # Build the new df_input_data with the output of CircularEconomy model
                model_ippu = IPPU(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                                df_input_data,
                                output_circular_economy,
                                model_ippu.integration_variables
                            )
                # Run IPPU model
                df_input_data[var_no_change_over_time] = df_input_data[var_no_change_over_time].mean()*np.array(params)[index_var_no_change_over_time]

                if list(var_change_over_time):
                    df_input_data[var_change_over_time] = df_input_data[var_change_over_time]*np.array(params)[index_var_change_over_time]

                df_model_data_project = model_ippu.project(df_input_data)

        if self.subsector_model == "ElectricEnergy":
            model_afolu = sm.AFOLU(sa.model_attributes)
            df_output_data = model_afolu.project(df_input_data)


            model_ippu = sm.IPPU(sa.model_attributes)

            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data,
                model_ippu.integration_variables
            )

            df_ippu_out = model_ippu.project(df_input_data)

            df_output_data = pd.merge(df_ippu_out, df_output_data)

            model_energy = sm.NonElectricEnergy(sa.model_attributes)

            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data,
                model_energy.integration_variables_non_fgtv
            )
                        
            df_output_data = model_energy.project(df_input_data)
                        


        return output

    """
    --------------------------------------------
    f method
    --------------------------------------------

    Description : fitness function

    # Inputs:
             * X        - calibration vector

    # Output:
             * output   - MSE value
    """
    def f(self, X):
        return self.objective_a(X)


    """
    ---------------------------------
    get_mse_test method
    ---------------------------------

    Description: The method recive params, run the subsector model,
                 and compute the MSE by the specific period of simulation

    # Inputs:
             * params       - calibration vector

    # Output:
             * output       - MSE value
    """

    def get_mse_test(self, params):

        if self.subsector_model == "AFOLU":
            #print("\n\tAFOLU")
            df_input_data = self.build_data_AFOLU(params)
            model_afolu = AFOLU(sa.model_attributes)
            df_model_data_project = model_afolu.project(df_input_data)
            output = self.build_performance_test_AFOLU(df_model_data_project)

        if self.subsector_model == "CircularEconomy":
            #print("\n\tRunning CircularEconomy")
            df_input_data = self.build_data_CircularEconomy(params)
            model_circular_economy = CircularEconomy(sa.model_attributes)
            df_model_data_project = model_circular_economy.project(df_input_data)
            output = self.build_performance_test_CircularEconomy(df_model_data_project)

        if self.subsector_model == "IPPU":
            if not self.downstream:
                #print("\n\tRunning IPPU")
                model_ippu = IPPU(sa.model_attributes)
                df_model_data_project = model_ippu.project(df_input_data)
            else:
                # First run CircularEconomy model
                model_circular_economy = CircularEconomy(sa.model_attributes)
                df_input_data = self.df_input_var.copy()
                df_input_data[self.calib_targets["CircularEconomy"]] = df_input_data[self.calib_targets["CircularEconomy"]] * self.best_vector["CircularEconomy"]
                output_circular_economy = model_circular_economy.project(df_input_data)

                # Build the new df_input_data with the output of CircularEconomy model
                model_ippu = IPPU(sa.model_attributes)

                df_input_data = sa.model_attributes.transfer_df_variables(
                                df_input_data,
                                output_circular_economy,
                                model_ippu.integration_variables
                            )
                # Run IPPU model
                df_input_data[var_no_change_over_time] = df_input_data[var_no_change_over_time].mean()*np.array(params)[index_var_no_change_over_time]

                if list(var_change_over_time):
                    df_input_data[var_change_over_time] = df_input_data[var_change_over_time]*np.array(params)[index_var_change_over_time]

                df_model_data_project = model_ippu.project(df_input_data)

        if self.subsector_model == "ElectricEnergy":
            model_afolu = sm.AFOLU(sa.model_attributes)
            df_output_data = model_afolu.project(df_input_data)


            model_ippu = sm.IPPU(sa.model_attributes)

            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data,
                model_ippu.integration_variables
            )

            df_ippu_out = model_ippu.project(df_input_data)

            df_output_data = pd.merge(df_ippu_out, df_output_data)

            model_energy = sm.NonElectricEnergy(sa.model_attributes)

            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data,
                model_energy.integration_variables_non_fgtv
            )
                        
            df_output_data = model_energy.project(df_input_data)
                        

        return output


    def run_calibration(self,optimization_method,population, maxiter, param_algo):
        '''
        ------------------------------------------

        run_calibration method

        -------------------------------------------

        # Inputs:
            * optimization_method       - Optimization method. Args: genetic_binary, differential_evolution,differential_evolution_parallel, pso


        # Output
            * mse_all_period            - Mean Squared Error for all periods
            * mse_training              - Mean Squared Error for training period
            * mse_test                  - Mean Squared Error for test period
            * calib_vector              - List of calibration scalar
        '''

        if optimization_method == "genetic_binary":
            # Optimize the function
            print("------------------------------------------------")
            print("         GENETIC BINARY      ")
            print("------------------------------------------------")
            start_time = time.time()

            print("--------- Start Cross Validation: {} on Node {}. Model: {}".format(self.cv_run,self.id_mpi,self.subsector_model))

            n_variables = len(self.calib_targets[self.subsector_model])
            i_sup_vec = [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "max_35"].item()  for i in self.calib_targets[self.subsector_model]]
            i_inf_vec = [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "min_35"].item()  for i in self.calib_targets[self.subsector_model]]
            precision = param_algo["precision"]
            pc = param_algo["pc"]
            binary_genetic = BinaryGenetic(population,n_variables,i_sup_vec,i_inf_vec,precision,maxiter,pc)
            self.fitness_values[self.subsector_model], self.best_vector[self.subsector_model] ,mejor_valor= binary_genetic.run_optimization(self.f)

            print("--------- End Cross Validation: {} on Node {}\nOptimization time:  {} seconds ".format(self.cv_run ,self.id_mpi,(time.time() - start_time)))
            print(" Cross Validation: {} on Node {}. MSE Training : {}".format(self.cv_run ,self.id_mpi,mejor_valor))
            mse_test = self.get_mse_test(self.best_vector[self.subsector_model])
            print(" Cross Validation: {} on Node {}. MSE Test : {}".format(self.cv_run ,self.id_mpi,mse_test))

        if optimization_method == "differential_evolution":
            print("------------------------------------------------")
            print("         DIFERENTIAL EVOLUTION      ")
            print("------------------------------------------------")

            start_time = time.time()

            print("--------- Start Cross Validation: {} on Node {}. Model: {}".format(self.cv_run,self.id_mpi,self.subsector_model))

            # 1 - Define the upper and lower bounds of the search space
            n_dim =  len(self.calib_targets[self.subsector_model])              # Number of dimensions of the problem
            lb = [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "min_35"].item()  for i in self.calib_targets[self.subsector_model]]  # lower bound for the search space
            ub =  [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "max_35"].item()  for i in self.calib_targets[self.subsector_model]] # upper bound for the search space
            lb = np.array(lb)
            ub = np.array(ub)

            # 2 - Define the parameters for the optimization
            max_iters = maxiter  # maximum number of iterations

            # 3 - Parameters for the algorithm
            pc = 0.7 # crossover probability
            pop_size = population   # number of individuals in the population

            # 4 - Define the cost function
            f_cost = self.f

            # 5 - Run the DE algorithm
            # call DE
            self.fitness_values[self.subsector_model], self.best_vector[self.subsector_model] ,mejor_valor = DE(f_cost,pop_size, max_iters,pc, lb, ub, step_size = 0.8)
            print("--------- End Cross Validation: {} on Node {}\nOptimization time:  {} seconds ".format(self.cv_run ,self.id_mpi,(time.time() - start_time)))
            print(" Cross Validation: {} on Node {}. MSE Training : {}".format(self.cv_run ,self.id_mpi,mejor_valor[0]))
            mse_test = self.get_mse_test(self.best_vector[self.subsector_model])
            print(" Cross Validation: {} on Node {}. MSE Test : {}".format(self.cv_run ,self.id_mpi,mse_test))

        if optimization_method == "differential_evolution_parallel":
            
            print("------------------------------------------------")
            print("         PARALLEL DIFERENTIAL EVOLUTION      ")
            print("------------------------------------------------")

            start_time = time.time()

            print("--------- Start Cross Validation: {} on Node {}. Model: {}".format(self.cv_run,self.id_mpi,self.subsector_model))

            # 1 - Define the upper and lower bounds of the search space
            n_dim =  len(self.calib_targets[self.subsector_model])              # Number of dimensions of the problem
            lb = [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "min_35"].item()  for i in self.calib_targets[self.subsector_model]]  # lower bound for the search space
            ub =  [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "max_35"].item()  for i in self.calib_targets[self.subsector_model]] # upper bound for the search space
            lb = np.array(lb)
            ub = np.array(ub)

            # 2 - Define the parameters for the optimization
            max_iters = maxiter  # maximum number of iterations

            # 3 - Parameters for the algorithm
            pc = 0.7 # crossover probability
            pop_size = population   # number of individuals in the population

            # 4 - Define the cost function
            f_cost = self.f

            # 5 - Run the DE algorithm
            # call DE
            self.fitness_values[self.subsector_model], self.best_vector[self.subsector_model] ,mejor_valor = DE_par(f_cost,pop_size, max_iters,pc, lb, ub, step_size = 0.4)
            print("--------- End Cross Validation: {} on Node {}\nOptimization time:  {} seconds ".format(self.cv_run ,self.id_mpi,(time.time() - start_time)))
            print(" Cross Validation: {} on Node {}. MSE Training : {}".format(self.cv_run ,self.id_mpi,mejor_valor[0]))
            mse_test = self.get_mse_test(self.best_vector[self.subsector_model])
            print(" Cross Validation: {} on Node {}. MSE Test : {}".format(self.cv_run ,self.id_mpi,mse_test))

        if optimization_method == "pso":
            
            print("------------------------------------------------")
            print("         PARTICLE SWARM OPTIMIZATION      ")
            print("------------------------------------------------")

            start_time = time.time()

            print("--------- Start Cross Validation: {} on Node {}. Model: {}".format(self.cv_run,self.id_mpi,self.subsector_model))

            # 1 - Define the upper and lower bounds of the search space
            n_dim =  len(self.calib_targets[self.subsector_model])              # Number of dimensions of the problem
            lb = [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "min_35"].item()  for i in self.calib_targets[self.subsector_model]]  # lower bound for the search space
            ub =  [self.df_calib_bounds.loc[self.df_calib_bounds["variable"] == i, "max_35"].item()  for i in self.calib_targets[self.subsector_model]] # upper bound for the search space
            lb = np.array(lb)
            ub = np.array(ub)

            # 2 - Define the parameters for the optimization
            max_iters = maxiter  # maximum number of iterations

            # 3 - Parameters for the algorithm
            pop_size = population   # number of individuals in the population
            # Cognitive scaling parameter
            α = param_algo["alpha"]
            # Social scaling parameter
            β = param_algo["beta"]

            # velocity inertia
            w = 0.5
            # minimum value for the velocity inertia
            w_min = 0.4
            # maximum value for the velocity inertia
            w_max = 0.9

            # 4 - Define the cost function
            f_cost = self.f

            # 5 - Run the PSO algorithm
            # call DE
            self.fitness_values[self.subsector_model], self.best_vector[self.subsector_model] ,mejor_valor = PSO(f_cost,pop_size, max_iters, lb, ub,α,β,w,w_max,w_min)
            print("--------- End Cross Validation: {} on Node {}\nOptimization time:  {} seconds ".format(self.cv_run ,self.id_mpi,(time.time() - start_time)))
            print(" Cross Validation: {} on Node {}. MSE Training : {}".format(self.cv_run ,self.id_mpi,mejor_valor))
            mse_test = self.get_mse_test(self.best_vector[self.subsector_model])
            print(" Cross Validation: {} on Node {}. MSE Test : {}".format(self.cv_run ,self.id_mpi,mse_test))
