# coding:utf-8
"""
Data loader for different experiments
Created  :   7, 12, 2019
Revised  :   7, 12, 2019
Author   :  David Leon (dawei.leng@ghddi.org)
All rights reserved
-------------------------------------------------------------------------------
"""
__author__ = 'dawei.leng'

import os
import numpy as np, psutil, multiprocessing, threading
from ligand_based_VS_data_preprocessing import convert_SMILES_to_graph

def load_batch_data_4v4(samples, batch_sample_idx, atom_dict=None, add_self_connection=False, aux_data_list=None):
    """
    batch data loader for model_4v4
    :param samples: list of SMILES strings or returns from `convert_SMILES_to_graph()` or mixture of them
    :param batch_sample_idx:
    :param atom_dict: required when samples contain SMILES string
    :param add_self_connection: whether append self-connection in adjacency matrix
    :param aux_data_list: list of auxiliary data
    :return: X, edges, membership, degree_slices (optional), *aux_data (optional)
             X: (node_num,), index of each node according to `atom_dict`, int64
             edges: (2, edge_num), int64, each column in format of (neighbor_node, center_node)
             membership: (node_num,), int64, representing to which graph the i_th node belongs
             *aux_data: list of auxiliary data organized in one batch
    """
    batch_size = len(batch_sample_idx)
    tokenized_sequences = []
    edges = []
    start_idxs = [0]
    membership = []
    for i in range(batch_size):
        s = samples[batch_sample_idx[i]]
        if isinstance(s, str):
            xs, es = convert_SMILES_to_graph(SMILES=s, atom_dict=atom_dict)
        else:
            xs, es = s
        tokenized_sequences.extend(xs)
        n = len(xs)
        membership.extend([i for _ in range(n)])
        start_idxs.append(n + start_idxs[i])
        edges.extend([(node_n + start_idxs[i], node_c + start_idxs[i]) for (node_n, node_c) in es])
        if add_self_connection:
            edges.extend([(j + start_idxs[i], j + start_idxs[i]) for j in range(n)])

    X = np.array(tokenized_sequences, dtype=np.int64)  # (node_num,), index of each node, int64
    edges = np.array(edges).transpose().astype(np.int64)  # (2, n_pair), each column denotes an edge, from node i to node j, int64
    membership = np.array(membership, dtype=np.int64)  # (node_num,)                        # (node_num,)

    batch_data = [X, edges, membership]

    if aux_data_list is not None:
        for item in aux_data_list:
            batch_data.append(item[batch_sample_idx])

    return batch_data

def feed_sample_batch(batch_data_loader,
                      features,
                      aux_data_list=None,
                      data_queue=None,
                      max_epoch_num=0,
                      batch_size=64, batch_size_min=None,
                      shuffle=False,
                      use_multiprocessing=False,
                      epoch_start_event=None,
                      epoch_done_event=None
                      ):

    me_process = psutil.Process(os.getpid())
    sample_num = features.shape[0]
    if batch_size_min is None:
        batch_size_min = batch_size
    batch_size_min = min([batch_size_min, batch_size])
    done, epoch = False, 0
    while not done:
        if use_multiprocessing:
            if me_process.parent() is None:     # parent process is dead
                raise RuntimeError('Parent process is dead, exiting')
        if epoch_start_event is not None:
            epoch_start_event.wait()
            epoch_start_event.clear()
        if epoch_done_event is not None:
            epoch_done_event.clear()
        if shuffle:
            index = np.random.choice(range(sample_num), size=sample_num, replace=False)
        else:
            index = np.arange(sample_num)
        index = list(index)
        end_idx = 0
        while end_idx < sample_num:
            current_batch_size = np.random.randint(batch_size_min, batch_size + 1)
            start_idx = end_idx
            end_idx = min(start_idx + current_batch_size, sample_num)
            batch_sample_idx = index[start_idx:end_idx]
            batch_data = batch_data_loader(features, batch_sample_idx=batch_sample_idx, aux_data_list=aux_data_list)
            data_queue.put(batch_data)

        if epoch_done_event is not None:
            epoch_done_event.set()
        epoch += 1
        if max_epoch_num > 0:
            if epoch >= max_epoch_num:
                done = True

def sync_manager(workers_epoch_start_events, workers_epoch_done_events):
    while 1:
        for event in workers_epoch_done_events:
            event.wait()
            event.clear()
        for event in workers_epoch_start_events:
            event.set()

class Data_Loader_Manager(object):
    def __init__(self,
                 batch_data_loader,
                 data_queue,
                 data,
                 shuffle=False,
                 batch_size=128,
                 batch_size_min=None,
                 worker_num=1,
                 use_multiprocessing=False,
                 auto_rewind=0,
                 name=None,
                 ):
        """
        :param auto_rewind: int, 0 means no auto-rewinding, >0 means auto-rewinding for at most n epochs
        """
        super().__init__()
        self.batch_data_loader          = batch_data_loader
        self.data_queue                 = data_queue
        self.data                       = data
        self.shuffle                    = shuffle
        self.batch_size_max             = batch_size
        self.batch_size_min             = self.batch_size_max if batch_size_min is None else batch_size_min
        self.worker_num                 = worker_num
        self.use_multiprocessing        = use_multiprocessing
        self.auto_rewind                = auto_rewind
        self.name                       = name
        self.workers                    = []
        self.workers_epoch_start_events = []
        self.workers_epoch_done_events  = []

        self.batch_size_min = min(self.batch_size_max, self.batch_size_min)

        if use_multiprocessing:
            Worker = multiprocessing.Process
            Event  = multiprocessing.Event
        else:
            Worker = threading.Thread
            Event  = threading.Event
        sample_num = data[0].shape[0]
        X, *aux_data_list = data  # X.shape = (n_sample, 2), np-array of tuples (nodes, neighbor_list), Y = ground_truth, shape=(n_sample,) / None

        startidx, endidx, idxstep = 0, 0, sample_num // worker_num

        for i in range(worker_num):
            worker_epoch_start_event = Event()
            worker_epoch_done_event  = Event()
            worker_epoch_start_event.set()
            worker_epoch_done_event.clear()
            self.workers_epoch_start_events.append(worker_epoch_start_event)
            self.workers_epoch_done_events.append(worker_epoch_done_event)

            startidx = i * idxstep
            if i == worker_num - 1:
                endidx = sample_num
            else:
                endidx = startidx + idxstep
            aux_data_list_worker = None
            if aux_data_list is not None or len(aux_data_list) > 0:
                aux_data_list_worker = []
                for item in aux_data_list:
                    aux_data_list_worker.append(item[startidx:endidx])
            data_proc = Worker(target=feed_sample_batch,
                               args=(batch_data_loader,
                                     X[startidx:endidx], aux_data_list_worker, data_queue,
                                     auto_rewind, batch_size, batch_size_min, shuffle,
                                     use_multiprocessing,
                                     worker_epoch_start_event,
                                     worker_epoch_done_event
                                     ),
                               name='%s_thread_%d' % (name, i))
            data_proc.daemon = True
            self.workers.append(data_proc)

        if auto_rewind > 0:
            sync_manager_proc = Worker(target=sync_manager,
                                       args=(self.workers_epoch_start_events,
                                             self.workers_epoch_done_events,
                                             ),
                                       name='%s_sync_manager' % name)
            sync_manager_proc.daemon = True
            self.workers.append(sync_manager_proc)

        for proc in self.workers:
            proc.start()
            print('%s started' % proc.name)

    def rewind(self):
        for event in self.workers_epoch_done_events:
            event.wait()
            event.clear()
        for event in self.workers_epoch_start_events:
            event.set()

    def close(self):
        for proc in self.workers:
            proc.terminate()


