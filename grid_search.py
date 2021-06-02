# coding:utf-8
"""
Objective & search space definitions for hyper-parameter grid searching of different models

Created   :   2,  7, 2020
Revised   :   2,  7, 2020
All rights reserved
"""
__author__ = 'dawei.leng'
import config
from ligand_based_VS_train import train

def objective_4v4(trial, args):
    model_config            = config.model_4v4_config()
    model_config.block_num  = trial.suggest_int('block_num', 4, 6)
    model_config.input_dim  = trial.suggest_categorical('input_dim', [75, 128, 256])
    model_config.hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    model_config.aggregation_methods               = trial.suggest_categorical('aggregation_methods', [['max', 'sum'], ['max', 'mean'], ['max'], ['mean'], ['sum'],['max', 'sum', 'att'], ['att'], ['max', 'mean', 'att']])
    model_config.multiple_aggregation_merge_method = trial.suggest_categorical('multiple_aggregation_merge_method', ['cat', 'sum'])
    model_config.multiple_readout_merge_method     = trial.suggest_categorical('multiple_readout_merge_method', ['cat', 'sum'])
    model_config.affine_before_merge               = trial.suggest_categorical('affine_before_merge', [False, True])
    model_config.node_feature_update_method        = trial.suggest_categorical('node_feature_update_method', ['cat', 'sum', 'rnn', 'max'])
    model_config.readout_methods                   = trial.suggest_categorical('readout_methods', [['rnn-max-sum'], ['sum'], ['mean'], ['max'], ['att'], ['max', 'sum'], ['max', 'mean'], ['rnn-mean-max'], ['rnn-max-sum', 'mean'], ['rnn-mean-max', 'att'], ['rnn-max-sum', 'att'], ['rnn-mean-max', 'sum'], ['rnn-max-sum', 'mean', 'att']])
    model_config.add_dense_connection = trial.suggest_categorical('add_dense_connection', [True, False])  # whether add dense connection among the blocks
    model_config.pyramid_feature      = trial.suggest_categorical('pyramid_feature', [True, False])
    model_config.slim                 = trial.suggest_categorical('slim', [True, False])

    args.config = model_config
    trainlog = train(args)
    if args.class_num > 1:
        if args.select_by_aupr:
            objective = trainlog.best_aupr
        else:
            objective = -trainlog.best_ER_test
    else:
        if args.select_by_corr:
            objective = trainlog.best_corr
        else:
            objective = -trainlog.best_rmse_test

    return objective

