# coding:utf-8
"""
Production model definitions for compound virtual screening based on ligand properties
Created  :   6, 11, 2019
Revised  :
Author   :  David Leon (dawei.leng@ghddi.org)
All rights reserved
-------------------------------------------------------------------------------
"""

__author__ = 'dawei.leng'

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_ext.module import BatchNorm1d, BatchNorm
from torch.nn.utils import weight_norm
import pytorch_ext.functional as EF
import numpy as np, warnings, copy
from graph_ops import GraphConv, GraphReadout

# experimental & obsolete models
try:
    import torch_scatter
    torch_scatter_available = True
except ImportError:
    torch_scatter_available = False
    warnings.warn('torch-scatter package is not available, please use `neighbor_op` instead ')

__all__ = [
    'model_HAGNet',
    'model_4v4',
    'Model_Agent',
]

#--- Model Agent ---#
import config as model_config
from pytorch_ext.util import gpickle, get_file_md5
import data_loader
from functools import partial

def _get_device(device):
    """
    :param device: int or instance of torch.device
    """
    if isinstance(device, torch.device):
        torch_device = device
    elif device < 0:
        torch_device = torch.device('cpu')
    else:
        torch_device = torch.device('cuda:%d' % device)
    return torch_device

class Model_Agent(object):
    """
    Model agent for handling different model specific events: IO, forward, loss, predict, etc.
    """
    def __init__(self,
                 device=-1,                  # int or instance of torch.device or list of these two
                 model_ver='4v4',
                 output_dim=2,
                 task='classification',      # {'classification', 'regression'}
                 config=model_config.model_4v4_config(),
                 model_file=None,
                 label_file=None,
                 atom_dict_file=None,
                 load_weights_strict=True,
                 load_all_state_dicts=False, # if there're multiple state dicts in `model_file`, load them sequentially
                 **kwargs):                  # reserved, not used for now
        super().__init__()
        self.model_ver            = model_ver
        self.output_dim           = output_dim
        self.model_file           = model_file
        self.label_file           = label_file
        self.atom_dict_file       = atom_dict_file
        self.model_file_md5       = None
        self.label_file_md5       = None
        self.atom_dict_file_md5   = None
        self.task                 = task
        self.config               = config
        self.atom_dict            = None
        self.label_dict           = None
        self.model                = None
        self.batch_data_loader    = None
        self.load_weights_strict  = load_weights_strict
        self.load_all_state_dicts = load_all_state_dicts
        if self.load_all_state_dicts is True:
            self.load_weights_strict = False

        if isinstance(device, list) or isinstance(device, tuple):
            self.device = [_get_device(dv) for dv in device]
        else:
            self.device = _get_device(device)
        assert self.task in {'classification', 'regression'}
        self.initialize()

    @staticmethod
    def load_weight(model_file, device=torch.device('cpu')):
        model_data = torch.load(model_file, map_location=device)
        return model_data

    def dump_weight(self, filename):
        model_data_to_save = dict()
        model_data_to_save['state_dict'] = self.model.state_dict()
        model_data_to_save['config']     = self.config.__dict__
        torch.save(model_data_to_save, filename)

    def _load_model_state(self, model, model_file):
        model_data = self.load_weight(model_file)
        # --- for back compatibility ---#
        if 'state_dict' in model_data:
            state_dict = model_data['state_dict']
            assert model_data['config'] == self.config.__dict__, "Model config not match, config loaded from model file = %s" % model_data['config']
        else:
            state_dict = model_data

        if isinstance(state_dict, list):
            if not self.load_all_state_dicts:
                state_dict = [state_dict[0]]
        else:
            state_dict = [state_dict]
        for sdict in state_dict:
            missing_keys, unexpected_keys = model.load_state_dict(sdict, strict=self.load_weights_strict)
            if len(missing_keys) > 0:
                warnings.warn('missing weights = {%s}' % ', '.join(missing_keys))
            if len(unexpected_keys) > 0:
                warnings.warn('unexpected weights = {%s} ' % ', '.join(unexpected_keys))
        return model

    def initialize(self):
        if self.atom_dict_file is not None:
            self.atom_dict_file_md5 = get_file_md5(self.atom_dict_file)
            self.atom_dict, _, _ = gpickle.load(self.atom_dict_file)
            if '<unk>' not in self.atom_dict:
                self.atom_dict['<unk>'] = len(self.atom_dict)

        batch_data_loader = getattr(data_loader, 'load_batch_data_%s' % self.model_ver)
        if self.model_ver == '4v4':
            self.batch_data_loader = partial(batch_data_loader, atom_dict=self.atom_dict)
            self.model = model_4v4(num_embedding=len(self.atom_dict),
                                   output_dim=self.output_dim,  # class num
                                   **self.config.__dict__)
        else:
            raise ValueError('self.model_ver = %s not supported' % self.model_ver)

        if not isinstance(self.device, list):
            self.model = self.model.to(self.device)

        if self.model_file is not None:
            if isinstance(self.model_file, list):     # model ensemble, support suspended
                model_base = self.model
                self.model = [model_base]
                for i in range(len(self.model_file)-1):
                    model = copy.deepcopy(model_base)
                    model.to(self.device)
                    self.model.append(model)
                self.model_file_md5 = []
                for i, model_file in enumerate(self.model_file):
                    self.model_file_md5.append(get_file_md5(model_file))
                    model = self._load_model_state(self.model[i], model_file)
                    self.model[i] = model
            else:
                self.model_file_md5 = get_file_md5(self.model_file)
                self.model = self._load_model_state(self.model, self.model_file)

        if self.label_file is not None:
            self.label_file_md5 = get_file_md5(self.label_file)
            self.label_dict     = gpickle.load(self.label_file)

    def forward(self, batch_data, calc_loss=False, **kwargs):
        if 'dropout' in kwargs:
            dropout = kwargs['dropout']
        else:
            dropout = 0
        if self.model_ver in {'4v4'}:
            X, edges, membership, *Y = batch_data
            degree_slices = None
            X_tensor = torch.from_numpy(X).to(self.device)
            edges = torch.from_numpy(edges).to(self.device)
            membership = torch.from_numpy(membership).to(self.device)
            scorematrix, graphs = self.model.forward(X_tensor, edges, membership, dropout, degree_slices=degree_slices)
        else:
            raise ValueError('model_ver = %s not supported' % self.model_ver)
        if calc_loss:
            Y = Y[0]
            device = self.device[0] if isinstance(self.device, list) else self.device
            Y_tensor = torch.from_numpy(Y).to(device)
            if 'class_weight' in kwargs:
                class_weight = kwargs['class_weight']
                class_weight = torch.from_numpy(class_weight).to(device)
            else:
                class_weight = None
            if 'sample_weight' in kwargs:
                sample_weight = kwargs['sample_weight']
                sample_weight = torch.from_numpy(sample_weight).to(device)
            else:
                sample_weight = None
            if self.task == 'classification':
                if sample_weight is None:
                    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
                    loss = criterion(scorematrix, Y_tensor)
                else:
                    criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='none')
                    loss = criterion(scorematrix, Y_tensor)
                    loss = loss * sample_weight
                    loss = loss.mean()
            else:
                raise ValueError('Support removed')

            return scorematrix, loss
        return scorematrix, graphs

    def predict(self, batch_data, **kwargs):
        self.model.eval()
        scorematrix, *_ = self.forward(batch_data, **kwargs)
        if self.task == 'classification':
            scorematrix        = F.softmax(scorematrix, dim=1)
            best_ps, best_idxs = torch.max(scorematrix, dim=1)
            predicted_idxs     = best_idxs.detach().cpu().numpy()
            predicted_scores   = best_ps.detach().cpu().numpy()
            if self.label_dict is not None:
                predicted_labels = [self.label_dict[idx] for idx in predicted_idxs]
            else:
                predicted_labels = [None] * len(predicted_idxs)
            return predicted_scores, predicted_idxs, predicted_labels
        else:   # regression
            raise ValueError('Support removed')

#--- model_4 series
class model_4v4(nn.Module):
    """
    HAG-Net: hybrid aggregation graph network
    """
    def __init__(self,
                 num_embedding=0,
                 block_num=5,
                 input_dim=75,
                 hidden_dim=256,
                 output_dim=2,     # class num
                 degree_wise=False,
                 max_degree=1,
                 aggregation_methods=('max', 'sum'),
                 multiple_aggregation_merge_method='sum',
                 affine_before_merge=False,
                 node_feature_update_method='rnn',
                 readout_methods=('rnn-sum-max',),
                 multiple_readout_merge_method='sum',
                 add_dense_connection=True,  # whether add dense connection among the blocks
                 pyramid_feature=True,
                 slim=True,
                 **kwargs
                 ):
        super().__init__()
        self.num_embedding = num_embedding
        if num_embedding > 0:
            self.emb0   = nn.Embedding(num_embeddings=num_embedding, embedding_dim=input_dim)
        self.block_num                  = block_num
        self.degree_wise                = degree_wise
        self.max_degree                 = max_degree
        self.aggregation_methods        = aggregation_methods
        self.multiple_aggregation_merge = multiple_aggregation_merge_method
        self.readout_methods            = readout_methods
        self.add_dense_connection       = add_dense_connection
        self.pyramid_feature            = pyramid_feature
        self.slim                       = slim
        self.classifier_dim             = input_dim
        self.blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.blocks.append(GraphConv(input_dim=input_dim, output_dim=input_dim,
                                         aggregation_methods=aggregation_methods,
                                         multiple_aggregation_merge_method=multiple_aggregation_merge_method,
                                         affine_before_merge=affine_before_merge,
                                         update_method=node_feature_update_method,
                                         degree_wise=degree_wise,
                                         max_degree=max_degree,
                                         backbone='default',
                                         **kwargs,
                                         ))
        self.readout_ops = nn.ModuleList()
        if self.pyramid_feature:
            readout_block_num = self.block_num + 1
        else:
            readout_block_num = 1
        self.readout_block_num = readout_block_num
        for i in range(readout_block_num):
            self.readout_ops.append(GraphReadout(readout_methods=self.readout_methods, input_dim=input_dim,
                                                 multiple_readout_merge_method=multiple_readout_merge_method,
                                                 affine_before_merge=affine_before_merge,
                                                 degree_wise=degree_wise,
                                                 max_degree=max_degree,
                                                 **kwargs))
            if self.slim:
                break

        readout_dim = input_dim * readout_block_num
        if self.degree_wise:
            readout_dim *= (self.max_degree + 1)
        if 'classifier_dim' in kwargs:
            self.classifier_dim = kwargs['classifier_dim']
        self.dense0 = nn.Linear(in_features=readout_dim, out_features=hidden_dim)
        self.dense1 = nn.Linear(in_features=hidden_dim, out_features=self.classifier_dim)
        self.dense2 = nn.Linear(in_features=self.classifier_dim, out_features=output_dim)

        if 'norm_method' in kwargs:
            norm_method = kwargs['norm_method'].lower()
        else:
            norm_method = 'bn'
        if norm_method == 'ln':
            self.bn0 = nn.LayerNorm(normalized_shape=hidden_dim)
            self.bn1 = nn.LayerNorm(normalized_shape=hidden_dim)
        elif norm_method == 'bn_notrack':
            self.bn0 = BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
            self.bn1 = BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
        elif norm_method == 'none':
            self.bn0 = None
            self.bn1 = None
        elif norm_method == 'wn':
            self.dense0 = weight_norm(self.dense0, name='weight', dim=0)
            self.dense1 = weight_norm(self.dense1, name='weight', dim=0)
            self.bn0 = None
            self.bn1 = None
        elif norm_method == 'newbn':
            self.bn0 = BatchNorm(input_shape=(None, hidden_dim))
            self.bn1 = BatchNorm(input_shape=(None, hidden_dim))
        else:  # default = bn
            self.bn0 = BatchNorm1d(num_features=hidden_dim, momentum=0.01)
            self.bn1 = BatchNorm1d(num_features=hidden_dim, momentum=0.01)

    def forward(self, x, edges=None, membership=None, dropout=0.0, degree_slices=None):
        """
        :param x: (node_num,) int64 if embedding is enabled; (node_num, feature_dim) else
        :param edges: (2, edge_num), int64, each column in format of (neighbor_node, center_node)
        :param membership: (node_num,), int64, representing to which graph the i_th node belongs
        :param dropout: dropout value
        :param degree_slices: (max_degree_in_batch, 2), each row in format of (start_idx, end_idx), in which '*_idx' corresponds
                              to edges indices; i.e., each row is the span of edges whose center node is of the same degree,
                              required when self.degree_wise = True, otherwise leave it to None
        :return: x, (batch_size, class_num)
        """
        if self.num_embedding > 0:
            x = self.emb0(x)
        #--- aggregation ---#
        if self.pyramid_feature:
            hiddens = [x]
        block_input = x
        for i in range(self.block_num):
            # block_input = EF.dropout(block_input, p=dropout, shared_axes=(1,), rescale=False, fill_value=0.0) if (self.training and dropout > 0.0) else block_input
            # if self.training and dropout > 0.0:
                # block_input = EF.dropout(block_input, p=dropout, rescale=True)
                # block_input = F.dropout(block_input, p=dropout, training=self.training)
            x = self.blocks[i](x=block_input, edges=edges,
                               include_self_in_neighbor=False,
                               dropout=dropout,
                               degree_slices=degree_slices)
            if self.add_dense_connection:
                block_input = block_input + x
            else:
                block_input = x
            if self.pyramid_feature:
                hiddens.append(x)
        #--- readout ---#
        if self.pyramid_feature:
            graph_representations = []
            for i in range(self.block_num+1):
                idx = 0 if self.slim else i
                hidden = EF.dropout(hiddens[i], p=dropout, shared_axes=(1,), rescale=False, fill_value=0) if (self.training and dropout > 0.0) else hiddens[i]
                pooled = self.readout_ops[idx](hidden, membership, degree_slices=degree_slices)
                graph_representations.append(pooled)
            x = torch.cat(graph_representations, dim=1)
        else:
            if self.training and dropout > 0.0:
                x = EF.dropout(x, p=dropout, shared_axes=(1,), rescale=False, fill_value=0)
            x = self.readout_ops[0](x, membership, degree_slices=degree_slices, dropout=dropout)

        #--- classification ---#
        graphs = x
        x = self.dense0(x)
        x = self.bn0(x)
        x = F.gelu(x)
        if self.training and dropout > 0.0:
            x = EF.dropout(x, p=dropout * 2, rescale=True)
        x = self.dense1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        if self.training and dropout > 0.0:
            x = EF.dropout(x, p=dropout * 2, rescale=True)
        x = self.dense2(x)
        return x, graphs

model_HAGNet = model_4v4
