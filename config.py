# coding:utf-8
"""
Parameter configurations for different experiments
Created  :   7, 11, 2019
Revised  :   7, 11, 2019
Author   :  David Leon (dawei.leng@ghddi.org)
All rights reserved
-------------------------------------------------------------------------------
"""
__author__ = 'dawei.leng'


class CONFIG(object):
    def __init__(self):
        super().__init__()
        self.version = None


class model_4v4_config2(CONFIG):
    def __init__(self):
        super().__init__()
        self.version                           = '1.0.1, 10-13-2020'
        self.block_num                         = 5
        self.input_dim                         = 75
        self.hidden_dim                        = 256
        self.degree_wise                       = False
        self.max_degree                        = 26
        self.aggregation_methods               = ['max', 'sum']
        self.multiple_aggregation_merge_method = 'sum'
        self.multiple_readout_merge_method     = 'cat'
        self.affine_before_merge               = False
        self.node_feature_update_method        = 'cat'
        self.readout_methods                   = ['max']
        self.add_dense_connection              = False
        self.pyramid_feature                   = True
        self.slim                              = True
        self.norm_method                       = 'newbn'  # 'bn', 'ln', 'wn', 'bn_notrack', 'none'
        self.classifier_dim                    = self.hidden_dim

model_HAGNet_config = model_4v4_config = model_4v4_config2

class model_4v4_config1(CONFIG):
    def __init__(self):
        super().__init__()
        self.block_num                         = 5
        self.input_dim                         = 75
        self.hidden_dim                        = 256
        self.degree_wise                       = False
        self.max_degree                        = 26
        self.aggregation_methods               = ['max', 'mean']
        self.multiple_aggregation_merge_method = 'sum'
        self.multiple_readout_merge_method     = 'sum'
        self.affine_before_merge               = False
        self.node_feature_update_method        = 'rnn'
        self.readout_methods                   = ['rnn-max-sum']
        self.add_dense_connection              = True  # whether add dense connection among the blocks
        self.pyramid_feature                   = True
        self.slim                              = True
        self.norm_method                       = 'newbn'  # 'bn', 'ln', 'wn', 'bn_notrack', 'none'
        self.classifier_dim                    = self.hidden_dim  # self.hidden_dim


class model_4v4_config_baseline(CONFIG):
    def __init__(self):
        super().__init__()
        self.block_num                         = 4
        self.input_dim                         = 75
        self.hidden_dim                        = 256
        self.degree_wise                       = False
        self.max_degree                        = 26
        self.aggregation_methods               = ['sum']
        self.multiple_aggregation_merge_method = 'cat'
        self.multiple_readout_merge_method     = 'cat'
        self.affine_before_merge               = False
        self.node_feature_update_method        = 'sum'
        self.readout_methods                   = ['sum']
        self.add_dense_connection              = False  # whether add dense connection among the blocks
        self.pyramid_feature                   = False
        self.slim                              = False
        self.norm_method                       = 'newbn'  # 'bn', 'ln', 'wn', 'bn_notrack', 'none'
        self.classifier_dim                    = self.hidden_dim  # self.hidden_dim
