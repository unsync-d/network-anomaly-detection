#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import _pickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from telethon import TelegramClient, sync
from datetime import datetime
import itertools
import requests
import os
import time
import sys
import json
from os import sched_getaffinity

GENERAL_RESULTS_PATH = r'../../results/'

DATASET_PATH = r'../../data/dataset_dist.csv'

OUTLIERS_PATH = r'../../outliers/MOD_final'

# SPLITS_OUTER_CV=10
SPLITS_OUTER_CV=2
# SPLITS_INNER_CV=5
SPLITS_INNER_CV=3
PARALLELIZATION = len(sched_getaffinity(0))

def UnderSampler(X, y):
    rus = RandomUnderSampler()
    X_res,y_res = rus.fit_resample(X, y)

    return X_res,y_res

def stringify_params(param):
    return str(param[0]) + "_" + str(param[1]) + "_" + str(param[2])

def myGSCV(X_train, y_train, params, n_splits=5):
    best_score = 0
    best_parameters = []

    cv_result_dict = {'params': [],
                      'mean_fit_times': []}
    for i in range(0,n_splits):
        cv_result_dict['fold_{}_scores'.format(i)] = []
    
    cv_result_dict['mean_scores'] = []
    cv_result_dict['scores_std'] = []

    # Por cada combinación de parámetros a probar...
    for param in params:
        ### Recuperamos outliers del archivo correspondente
        # OP -> Outliers Path
        filepath = OUTLIERS_PATH
        it = 0
        outliers_holder = {}

        with open(filepath, "rb") as jsonfile:
            outliers_holder = pickle.load(jsonfile)


        this_model = RandomForestClassifier(n_estimators=param[0],
                                            criterion=param[1],
                                            max_features=param[2], n_jobs=PARALLELIZATION)
        this_param_fit_times = []
        this_param_scores = []
        this_features = []

        cv_inner = StratifiedKFold(n_splits=n_splits)


        n_split = 0
        for in_train_ix, in_test_ix in cv_inner.split(X_train, y_train):
            ## Separación de conjuntos
            
            X_in_train, X_in_test = X_train.iloc[in_train_ix, :], X_train.iloc[in_test_ix, :]
            y_in_train, y_in_test = y_train.iloc[in_train_ix, :], y_train.iloc[in_test_ix, :]
            
            X_in_train.reset_index(drop=True, inplace=True)
            y_in_train.reset_index(drop=True, inplace=True)

            print("         STARTING INNER OUTLIER DETECTION: PARAMS({}) IT({})".format(param,n_split))
            sys.stdout.flush()
            print("            TRAIN SET ROWS BEFORE OD: {}".format(len(X_in_train.index)))
            sys.stdout.flush()

            # print("            ROWS TAKEN BY OD: {} MOD".format(len(outliers_holder['MOD'])))
            sys.stdout.flush()

            ## Detección de outliers en entrenamiento
            # mod_X_train = X_in_train.iloc[X_in_train.index.difference(outliers_holder['MOD'])]
            mod_X_train = X_in_train[~X_in_train['row_id'].isin(outliers_holder['MOD'])]
            # mod_y_train = y_in_train.iloc[y_in_train.index.difference(outliers_holder['MOD'])] 
            mod_y_train = y_in_train[~X_in_train['row_id'].isin(outliers_holder['MOD'])]
            
            # No debería ser necesario en el último conjunto de datos
            mod_X_train.drop(['row_id'], axis = 1, inplace=True)
            
            mod_X_train.reset_index(drop=True, inplace = True)
            mod_y_train.reset_index(drop=True, inplace = True)


            print("            TRAIN SET ROWS AFTER OD: {}\n".format(len(mod_X_train.index)))
            sys.stdout.flush()

            print("         STARTING INNER UNDERSAMPLING: PARAMS({}) IT({})".format(param,n_split))
            sys.stdout.flush()
            print("            TRAIN SET ROWS BEFORE SAMPLING: {}".format(len(mod_X_train.index)))
            sys.stdout.flush()

            # Undersampling entrenamiento
            mod_X_train, mod_y_train = UnderSampler(mod_X_train, mod_y_train)
            print("            TRAIN SET ROWS AFTER SAMPLING: {}\n".format(len(mod_X_train.index)))
            sys.stdout.flush()
            print("         INNER MODEL TRAINING: PARAMS({}) IT({})\n".format(param,n_split))
            sys.stdout.flush()

            # Entrenamiento del modelo
            start = time.process_time()
            this_model.fit(mod_X_train, mod_y_train.values.ravel())
            end = time.process_time()
            
            # Obtención y devolución de métricas
            this_param_fit_times.append(end - start)
            
            # No debería hacer falta en el último conjunto de datos
            X_in_test.drop(['row_id'], axis = 1, inplace=True)
            pred = this_model.predict(X_in_test)

            # Append score to the respective parameter-fold records
            # score = accuracy_score(y_in_test['Target'], pred)
            score = f1_score(y_in_test['Target'], pred)
            this_param_scores.append(score)
            cv_result_dict['fold_{}_scores'.format(n_split)].append(score)
            
            n_split = n_split + 1
        print("\n\n")
        
        # Obtención de métricas por combinación de parámetros
        cv_result_dict['mean_fit_times'].append(np.array(this_param_fit_times).mean())
        temp_score = np.array(this_param_scores).mean()
        
        if temp_score > best_score:
            best_score = temp_score
            best_parameters = param
        
        cv_result_dict['mean_scores'].append(temp_score)
        cv_result_dict['scores_std'].append(np.array(this_param_scores).std())
        cv_result_dict['params'].append(param)

    
    # Devolvemos mejores métricas
    cv_result_dict['best_score'] = best_score
    cv_result_dict['best_params'] = {'n_estimators':best_parameters[0],'criterion':best_parameters[1],'max_features':best_parameters[2]}
    return cv_result_dict


def main():
    pd.options.mode.chained_assignment = None  # default='warn'
    ## Establecemos la "semilla raiz" para garantizar reproductibilidad 
    np.random.seed(809)

    ## Recuperación del conjunto de datos

    ### X =  pd.read_csv(DATASET_PATH).head(20000)
    ### y =  pd.read_csv(TARGET_PATH).head(20000)
    X = pd.read_csv(DATASET_PATH).head(200000)
    ###y = pd.read_csv(TARGET_PATH)
    y = X[['Target']]
    X.drop(['Target'], inplace=True, axis=1)
    

    ## Definición del nombre de archivos de resultados
    RESULTS_FOLDER = datetime.now().strftime(r'RF_LOADED_O_RESULTADOS_EJECUCION_%Y_%m_%d_%H_%M_%S')
    RESULTS_PATH = datetime.now().strftime(r'../../results/{}/it_{}_rf_%Y_%m_%d_%H_%M_%S.json')
    LOG_FILE_PATH = datetime.now().strftime(r'../../logs/log_%Y_%m_%d_%H_%M_%S.log')
    try:
        os.makedirs(GENERAL_RESULTS_PATH + "/" + RESULTS_FOLDER)
    except OSError as e:
        raise

    ### Redireccionamos salida estándar al archivo de log 
    log_file = open(LOG_FILE_PATH, "w")
    old_stdout = sys.stdout
    sys.stdout = log_file


    ## Repeticiones para probar consistencia con diferentes semillas
    for global_iteration in range(0,2):
        outer_results_dict = {}
        for i in range(0, SPLITS_OUTER_CV):
            outer_results_dict['out_fold_{}_scores'.format(i)] = []
            outer_results_dict['out_fold_{}_inner_GSCV_results'.format(i)] = []


        ## Calculamos las combinaciones de parámetros

        # params_n_estimators = list(range(800,1500,100))
        params_n_estimators = [800,900,1000,1200]
        params_criterion = ['gini']
        # params_max_features= list(range(5,30,7))
        params_max_features= [6]

        params = []
        params.append(params_n_estimators)
        params.append(params_criterion)
        params.append(params_max_features)

        params = list(itertools.product(*params))


        ## Iniciamos Outer CV
        cv_outer = StratifiedKFold(n_splits=SPLITS_OUTER_CV)
        splitN = 0

        print("*************************")
        print("ITERATION {} IS STARTING.".format(global_iteration))
        print("*************************")
        for train_ix, test_ix in cv_outer.split(X,y):
            ## Recuperamos outliers del archivo donde los guardamos
            filepath = OUTLIERS_PATH

            outliers_holder = []

            with open(filepath, "rb") as jsonfile:
                outliers_holder = pickle.load(jsonfile)

            print("   SPLIT {} IS STARTING.".format(splitN))
            sys.stdout.flush()
            X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
            y_train, y_test = y.iloc[train_ix, :], y.iloc[test_ix, :]

            X_train.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)

            print("      GSCV IS STARTING.")
            sys.stdout.flush()
            # Inner GSCV
            results = myGSCV(X_train, y_train, n_splits=SPLITS_INNER_CV, 
                            params=params)
            # EOF Inner GSCV
            
            print("      STARTING OUTTER OUTLIER DETECTION:")
            sys.stdout.flush()
            print("         TRAIN SET ROWS BEFORE OD: {}".format(len(X_train.index)))
            sys.stdout.flush()


            ## Detección de otliers en entrenamiento
            # mod_X_train = X_train.iloc[X_train.index.difference(outliers_holder['MOD'])]
            mod_X_train = X_train[~X_train['row_id'].isin(outliers_holder['MOD'])]
            # mod_y_train = y_train.iloc[y_train.index.difference(outliers_holder['MOD'])] 
            mod_y_train = y_train[~X_train['row_id'].isin(outliers_holder['MOD'])]

            # No debería ser necesario con el nuevo conjunto
            mod_X_train.drop(['row_id'], axis = 1, inplace=True)

            mod_X_train.reset_index(drop=True, inplace = True)
            mod_y_train.reset_index(drop=True, inplace = True)

            print("         TRAIN SET ROWS AFTER OD: {}\n".format(len(mod_X_train.index)))
            sys.stdout.flush()

            print("      UNDERSAMPLING IS STARTING.")
            sys.stdout.flush()
            print("         TRAIN SET ROWS BEFORE SAMPLING: {}".format(len(mod_X_train.index)))
            sys.stdout.flush()

            ## Undersampling train
            mod_X_train, mod_y_train = UnderSampler(mod_X_train, mod_y_train)
            print("         TRAIN SET ROWS AFTER SAMPLING: {}\n".format(len(mod_X_train.index)))
            sys.stdout.flush()

            print("      BEST MODEL TRAINING IS STARTING.")
            sys.stdout.flush()
            best_model = RandomForestClassifier()
            best_model.set_params(**results['best_params'])
            best_model.fit(mod_X_train, mod_y_train.values.ravel())

            ## Predecimos con el modelo
            # No debería ser necesario con el nuevo conjunto
            # X_test.drop(['startDateTime', 'stopDateTime'], axis = 1, inplace=True)
            X_test.drop(['row_id'], axis = 1, inplace=True)
            yhat = best_model.predict(X_test)

            ## Calculamos CM
            conf_matrix = confusion_matrix(y_test, yhat)
 
            ## Guardamos resultados
            #outer_results = {'out_fold_{}_scores'.format(splitN): conf_matrix.tolist(),
            #                    'out_fold_{}_inner_GSCV_results'.format(splitN): results}

            outer_results_dict['out_fold_{}_scores'.format(splitN)].append(conf_matrix.tolist())
            outer_results_dict['out_fold_{}_inner_GSCV_results'.format(splitN)].append(results)



            splitN = splitN + 1
            
            print("\n-----------------------------------------------------------------\n\n\n")
            sys.stdout.flush()



        with open(RESULTS_PATH.format(RESULTS_FOLDER, global_iteration), "w") as jsonfile:
            json.dump(outer_results_dict, jsonfile)

    sys.stdout = old_stdout
    log_file.close()

if __name__ == "__main__":
    main()


