# coding:utf-8
"""
Data pre-processing for ligand based compound virtual screening.
Created  :   6, 11, 2019
Revised  :   11, 28, 2019
Author   :  David Leon (dawei.leng@ghddi.org), Pascal Guo (jinjiang.guo@ghddi.org)
All rights reserved
"""
#------------------------------------------------------------------------------------------------

__author__ = 'dawei.leng, jinjiang.guo'

import pandas as pd, numpy as np, warnings, os, sys
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from matplotlib import pyplot as plt
from pytorch_ext.util import gpickle, sys_output_tap, get_time_stamp
# from featurizer import featurize_A0

def dataset_shuffle(dataset, shuffle_idxs=None):
    """
    Shuffle dataset
    :param dataset: list or dict of data items (np.ndarray) which has the 1st dimension for sample index
    :param shuffle_idxs: if not given, random shuffle will be executed
    :return:
    """

    if shuffle_idxs is None:
        if isinstance(dataset, dict):
            sample_num = dataset['features'].shape[0]
        else:
            sample_num = dataset[0].shape[0]
        shuffle_idxs = np.arange(sample_num)
        np.random.shuffle(shuffle_idxs)
    else:
        sample_num = len(shuffle_idxs)

    if isinstance(dataset, dict):
        for k in dataset:
            if dataset[k] is None or not isinstance(dataset[k], np.ndarray) or dataset[k].shape[0] != sample_num:
                if k in {'sample_weights', 'class_distribution', 'scaffold_dict'}:  # reserved keys
                    pass
                else:
                    warnings.warn('dataset[%s] will not be shuffled due to invalid type/shape' % k)
            else:
                dataset[k] = dataset[k][shuffle_idxs]
        return dataset
    else:
        dataset_shuffled = []
        for i, item in enumerate(dataset):
            if item is None or not isinstance(item, np.ndarray) or item.shape[0] != sample_num:
                dataset_shuffled.append(item)
                warnings.warn('dataset[%d] will not be shuffled due to invalid type/shape' % i)
            else:
                dataset_shuffled.append(item[shuffle_idxs])
        return dataset_shuffled

def balance_resample(dataset, class_labels, class_num=None, expected_class_distribution=None, expected_sample_num=None, shuffled=True):
    """
    Balance resampling of given dataset.
    :param dataset: list of data items (np.ndarray) which has the 1st dimension for sample index
    :param class_labels:
    :param class_num: if None, `np.max(class_labels) + 1` will be used
    :param expected_class_distribution: tuple of floats, required to sum to 1.0
    :param expected_sample_num: if None, the same number of samples will be returned
    :param shuffled: whether shuffle the returned result, default True
    :return: dataset balanced
    """
    original_sample_num = dataset[0].shape[0]
    if expected_sample_num is None:
        expected_sample_num = original_sample_num
    if class_num is None:
        class_num = np.max(class_labels) + 1
    expected_class_distribution /= expected_class_distribution.sum()
    n_partition = int_partition_by_ratio(expected_sample_num, expected_class_distribution, minimum=1)
    new_idxs = []
    idx_range = np.arange(original_sample_num)
    for i in range(class_num):
        class_idxs = idx_range[class_labels==i]
        if len(class_idxs) > 0:
            new_class_idxs = np.random.choice(class_idxs, n_partition[i], replace=True)
            new_idxs.append(new_class_idxs)
    new_idxs = np.concatenate(new_idxs)
    dataset_balanced = []
    for item in dataset:
        if len(item.shape) > 1:
            dataset_balanced.append(item[new_idxs, ::])
        else:
            dataset_balanced.append(item[new_idxs])
    if shuffled:
        return dataset_shuffle(dataset_balanced)
    return dataset_balanced

def int_partition_by_ratio(n, partition_ratio, minimum=1):
    """
    Partition an integer `n` to multiple part by given ratios
    :param n:
    :param partition_ratio: tuple, must sum to 1.0 exactly
    :param minimum: the minimum value of a partition
    :return: integer array which sum to n exactly and each element >= `minimum`
    """
    partition_ratio = np.asarray(partition_ratio)
    partition_ratio = partition_ratio/partition_ratio.sum()  # make sure `partition_ratio` sums to 1.0
    n_partition = np.round(n * partition_ratio).astype(np.int)

    #--- check minimum criterion ---#
    idxs = np.arange(len(n_partition))
    mask = n_partition < minimum
    idxs_to_increase = idxs[mask]
    for i in idxs_to_increase:
        n_partition[i] += minimum - n_partition[i]
    #--- check summation criterion ---#
    while n_partition.sum() > n:
        idx = n_partition.argmax()
        n_partition[idx] -= 1
    return n_partition

def dataset_partition_by_class(dataset, class_labels, partition_ratio=(0.8, 0.2)):
    """
    Partition dataset with specified ratio for each class
    :param dataset: list or dict of data items (np.ndarray) to be partitioned, each array should have the 1st dimension for the sample number
    :param class_labels: it's required that class label starts from 0
    :param partition_ratio: list of float, required to sum to 1.0
    :return: list of datasets corresponding to `partition_ratio`
    """
    class_num   = max(class_labels) + 1
    sample_num  = len(class_labels)
    sample_idxs = np.arange(sample_num)
    partition_idxs_for_each_class = []
    for i in range(class_num):
        mask = class_labels == i
        sample_idxs_one_class = sample_idxs[mask]
        n_partition = int_partition_by_ratio(len(sample_idxs_one_class), partition_ratio, minimum=1)
        partition_idxs_one_class = []
        idx = 0
        for n in n_partition:
            partition_idxs_one_class.append(sample_idxs_one_class[idx:idx+n])
            idx += n
        partition_idxs_for_each_class.append(partition_idxs_one_class)

    dataset_partitions = []

    for i in range(len(partition_ratio)):
        if isinstance(dataset, dict):
            dataset_partition = dict()
        else:
            dataset_partition = []
        idxs = []
        for j in range(class_num):
            idxs.append(partition_idxs_for_each_class[j][i])
        idxs = np.concatenate(idxs)
        idxs.sort()    # keep the original order

        if isinstance(dataset, dict):
            for k in dataset:
                if dataset[k] is None or not isinstance(dataset[k], np.ndarray) or dataset[k].shape[0] != sample_num:
                    if k in {'sample_weights', 'class_distribution', 'scaffold_dict'}:  # reserved keys
                        pass
                    else:
                        warnings.warn('dataset[%s] will not be partitioned due to invalid type/shape' % k)
                    dataset_partition[k] = dataset[k]
                else:
                    dataset_partition[k] = dataset[k][idxs]
        else:
            for i, item in enumerate(dataset):
                if item is None or not isinstance(item, np.ndarray) or item.shape[0] != sample_num:
                    warnings.warn('dataset[%d] will not be partitioned due to invalid type/shape' % i)
                    dataset_partition.append(item)
                else:
                    dataset_partition.append(item[idxs])

        dataset_partitions.append(dataset_partition)

    return dataset_partitions

def calc_class_distribution(class_labels, class_num=None):
    """
    Calculate class distribution
    :param class_labels:
    :param class_num:
    :return: normalized class distribution
    """
    if class_num is None:
        class_num = np.max(class_labels) + 1
    sample_num = len(class_labels)
    class_distribution = np.zeros(class_num)
    for i in range(class_num):
        class_distribution[i] = (class_labels==i).sum()/sample_num
    return class_distribution

def calc_data_distribution(class_labels, class_num=None):
    """
    Calculate data distribution for regression problem
    :param class_labels:
    :param class_num:
    :return: normalized class distribution
    """
    # class_num = int(class_num)
    if class_num is None:
        class_num = np.max(class_labels) + 1
    sample_num = len(class_labels)
    data_distribution = np.zeros(class_num+1)
    for i in range(class_num+1):
        if i == 0:
            data_distribution[i] = (class_labels<=i).sum()/sample_num
            print('label range: <= %d, sample_number %d, total_sample_number %d' %(i, (class_labels<=i).sum(), sample_num))
        elif i < class_num:
            data_distribution[i] = ((class_labels<=i).sum()-(class_labels<=i-1).sum())/sample_num
            print('label range: (%d, %d], sample_number %d, total_sample_number %d' %(i-1, i, (class_labels<=i).sum()-(class_labels<=i-1).sum(), sample_num))
        else:
            data_distribution[i] = (class_labels>i-1).sum()/sample_num
            print('label range: (%d, %d], sample_number %d, total_sample_number %d' %(i-1, i, (class_labels>i-1).sum(), sample_num))
    return data_distribution

def canonicalize_molecule(molecule, addH=True):
    """
    canonicalize a molecule
    :param molecule: 'mol' object from RDKit
    :param addH: whether add back hydrogen atoms
    :return: molecule canonicalized, 'mol' object from RDKit
    """
    if addH:
        molecule = Chem.AddHs(molecule)  # add back all hydrogen atoms
    order    = Chem.CanonicalRankAtoms(molecule)  # to get a canonical form here
    molecule = Chem.RenumberAtoms(molecule, order)
    return molecule

def canonicalize_smiles(smiles_string, addH=True):
    """
    Canonicalize a SMILES string
    :param smiles_string:
    :return:
    """
    molecule = Chem.MolFromSmiles(smiles_string)
    molecule = canonicalize_molecule(molecule, addH)
    smiles   = Chem.MolToSmiles(molecule, isomericSmiles=True, kekuleSmiles=False,
                                canonical=True,
                                allBondsExplicit=True, allHsExplicit=True)
    return smiles

def dataset_summary(dataset):
    for k, v in dataset.items():
        if hasattr(v, 'shape'):
            print('    [%s].shape = %s' % (k, str(v.shape)))
        elif v is None:
            print('    [%s] = None' % k)
        elif isinstance(v, list):
            print('    [%s] = %s' % (k, v))
        elif isinstance(v, dict):
            print('    [%s].len = %d' % (k, len(v)))
        else:
            print('    [%s] = ' % k, v)

def convert_SMILES_to_graph(SMILES=None, canonical_mol=None, atom_dict=None, version='G0'):
    """
    Convert SMILES/canonicalized molecule to undirected graph
    :param SMILES: SMILES string
    :param canonical_mol: canonicalized molecule, as returned by `canonicalize_molecule()`, if given, input of `SMILES` will be ignored
    :param atom_dict: dict
    :param version: string
    :return: (nodes, edges), in which nodes is a list of integer tokenized with `atom_dict`, `edges` is a list of edge with format of tuple (i,j)
    """
    if version == 'G0':
        edges = []
        if canonical_mol is not None:
            molecule = canonical_mol
        else:
            s = SMILES
            molecule = Chem.MolFromSmiles(s)
            molecule = canonicalize_molecule(molecule, addH=True)

        tokenized_seq = []
        for atom in molecule.GetAtoms():
            if atom.GetSymbol() not in atom_dict:
                if '<unk>' in atom_dict:
                    tokenized_seq.append(atom_dict['<unk>'])
                else:
                    tokenized_seq.append(len(atom_dict))  # OOV
            else:
                tokenized_seq.append(atom_dict[atom.GetSymbol()])
        edge_list = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds()]
        edge_list_reverse = [(j, i) for (i, j) in edge_list]
        edges.extend(edge_list)
        edges.extend(edge_list_reverse)   # add symmetric edge
        return tokenized_seq, edges
    else:
        raise NotImplementedError('version = %s not supported' % version)

def data_preprocess(args):
    #--- initialization ---#
    csv_file               = args.csv_file
    label_col_name         = args.label_col_name
    smiles_col_name        = args.smiles_col_name
    id_col_name            = args.id_col_name
    sample_weight_col_name = args.sample_weight_col_name
    task                   = args.task.lower()
    use_chirality          = args.use_chirality.lower() in {'true', 'yes'}
    addH                   = args.addH.lower() in {'true', 'yes'}
    feature_ver            = args.feature_ver.lower()
    partition_ratio        = args.partition_ratio
    save_root_folder       = args.save_root_folder

    if save_root_folder is None:
        save_root_folder = os.path.join(os.getcwd(), 'dataset')
    if not os.path.exists(save_root_folder):
        os.makedirs(save_root_folder)

    stdout_tap      = sys_output_tap(sys.stdout)
    stderr_tap      = sys_output_tap(sys.stderr)
    sys.stdout      = stdout_tap
    sys.stderr      = stderr_tap
    time_stamp      = get_time_stamp()
    atom_dict, _, _ = gpickle.load(args.atom_dict_file)
    assert task in {'classification', 'regression', 'prediction'}, 'task = %s not supported' % task
    assert feature_ver in {'0','1', '2'}, "feature_ver = %s not supported" % feature_ver
    _, basename = os.path.split(os.path.abspath(csv_file))
    basename_without_ext  = os.path.splitext(basename)[0]
    save_folder = os.path.join(save_root_folder, basename_without_ext)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if args.doc is None:
        featurized_file = os.path.join(save_folder, basename_without_ext+'_%s_%s@%s.gpkl' % (task, feature_ver, time_stamp,))
        stdout_log_file = os.path.join(save_folder, basename_without_ext+'_%s_preprocessing_stdout@%s.txt' % (task, time_stamp,))
        stderr_log_file = os.path.join(save_folder, basename_without_ext+'_%s_preprocessing_stderr@%s.txt' % (task, time_stamp,))
    else:
        featurized_file = os.path.join(save_folder, basename_without_ext + '_%s_%s_%s@%s.gpkl' % (args.doc, task, feature_ver, time_stamp,))
        stdout_log_file = os.path.join(save_folder, basename_without_ext + '_%s_%s_preprocessing_stdout@%s.txt' % (args.doc, task, time_stamp,))
        stderr_log_file = os.path.join(save_folder, basename_without_ext + '_%s_%s_preprocessing_stderr@%s.txt' % (args.doc, task, time_stamp,))

    stdout_tap_log  = open(stdout_log_file, 'at')
    stderr_tap_log  = open(stderr_log_file, 'at')
    stdout_tap.set_file(stdout_tap_log)
    stderr_tap.set_file(stderr_tap_log)
    print('Task = %s, feature_ver = %s' % (task, feature_ver))
    print(args.__dict__)

    #--- load csv file ---#
    if csv_file.endswith('.tab'):
        data_frame = pd.read_csv(csv_file, header=0, delimiter='\t')    # add support for TDC *.tab files
    else:
        data_frame = pd.read_csv(csv_file, header=0)
    sample_num = data_frame.shape[0]
    print("Column names got from csv file = %s" % str(data_frame.columns.values))
    print("Total sample number = %d" % sample_num)
    data_frame = data_frame.replace(np.nan, "", regex=True)
    # data_frame = data_frame[~data_frame[label_col_name].isin([""])] # delete rows which contain null value
    pd.options.display.float_format = '{:,.3f}'.format        # limit output to 3 decimal places.
    print('\nData brief peek\n----------------------------------')
    print(data_frame.describe())

    #--- handle columns ---#
    if smiles_col_name not in data_frame:
        raise ValueError('SMILES column name "%s" not found in csv file' % smiles_col_name)
    else:
        SMILES_list = data_frame[smiles_col_name].to_numpy()
    if label_col_name in data_frame:
        labels = data_frame[label_col_name]
        labels = pd.to_numeric(labels).to_numpy()
    else:
        labels = None
        if task in {'classification', 'regression'}:
            raise ValueError('label column name "%s" not found in csv file' % label_col_name)
        if label_col_name is not None and len(label_col_name) > 0:
            warnings.warn('label column name "%s" not found in csv file' % label_col_name)
    if id_col_name in data_frame:
        IDs = data_frame[id_col_name].to_numpy()
    else:
        IDs = None
        if id_col_name is not None and len(id_col_name) > 0:
            warnings.warn('ID column name "%s" not found in csv file' % id_col_name)
    if sample_weight_col_name in data_frame:
        sample_weights = data_frame[sample_weight_col_name].to_numpy().astype(np.float32)
    else:
        sample_weights = None
        if sample_weight_col_name is not None and len(sample_weight_col_name) > 0:
            warnings.warn('sample weight column name "%s" not found in csv file' % sample_weight_col_name)

    #--- check class distribution if labels given ---#
    class_distribution = None
    if labels is not None:
        if task == 'classification':
            print('Labels are treated in classification mode')
            labels = labels.astype(np.int64)
            label_min, label_max = labels.min(), labels.max()
            if label_min != 0:
                warnings.warn('Class label should start from 0, but label_min = %d' % label_min)
            if args.class_num is not None:
                class_num = args.class_num
                print('class num is specified as %d' % class_num)
            else:
                class_num = label_max - label_min + 1
                print('class num is inferred as %d' % class_num)
            if args.label_min is not None:
                labels -= int(args.label_min)
            else:
                labels -= label_min
            class_distribution = calc_class_distribution(labels, class_num)
            print('\nlabel_min = %d, label_max = %d' % (label_min, label_max))
            print('class distribution = ', class_distribution)
            print('class %d number = %d' % (0, sample_num*class_distribution[0]))
            print('class %d number = %d' % (1, sample_num*class_distribution[1]))
        else:
            print('Labels are treated in regression mode')
            labels = labels.astype(np.float32)
            label_min, label_max = labels.min(), labels.max()
            print('\nlabel_min = %d, label_max = %d' % (label_min, label_max))
            label_min = int(label_min)
            label_max = int(label_max)
            data_section = label_max-label_min+1
            labels -= label_min

    #--- feature extraction ---#
    print('\nExtracting initial features...')
    molecules     = []
    mol_features  = []
    mask          = np.ones(sample_num, dtype=np.bool)
    for i in range(sample_num):
        smiles_string = SMILES_list[i]
        # smiles_string = 'O=C1N[S](=O)(=O)C2=C1C=CC=C2'
        # print(smiles_string)
        if len(smiles_string) <= 0:
            molecule = None
        else:
            molecule = Chem.MolFromSmiles(smiles_string)   # return a `Mol` object from rdkit
        if molecule is None or len(molecule.GetBonds()) <= 0:
            mask[i] = False
            warnings.warn('Invalid SMILES @%d = %s' % (i, smiles_string))
        else:
            molecule = canonicalize_molecule(molecule, addH=addH)
            molecules.append(molecule)
            for atom in molecule.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol not in atom_dict:
                    warnings.warn('OOV atom = %s encounter @ SMILES[%d] = %s' % (symbol, i, smiles_string))
            if feature_ver == '0':
                mol_feature = featurize_A0(molecule, use_chirality=use_chirality)  # (nodes, neighbours_list)
            elif feature_ver == '1':
                mol = Chem.MolFromSmiles(smiles_string)
                Chem.Kekulize(mol)
                s_kekulized = Chem.MolToSmiles(mol, kekuleSmiles=True)
                mol_feature = s_kekulized
            elif feature_ver == '2':
                xs, es = convert_SMILES_to_graph(canonical_mol=molecule, atom_dict=atom_dict, version='G0')
                mol_feature = (xs, es)
            else:
                raise NotImplementedError('feature_ver = %s not supported' % feature_ver)
            mol_features.append(mol_feature)
        if (i+1) % 500 == 0:
            print('%d of %d done' % (i+1, sample_num))
    invalid_num = sample_num - mask.sum()
    if invalid_num > 0:
        print('Total %d invalid SMILES encountered' % invalid_num)
    molecules      = np.asarray(molecules)
    features       = np.asarray(mol_features, dtype=object)
    labels         = labels[mask] if labels is not None else None
    IDs            = IDs[mask] if IDs is not None else None
    sample_weights = sample_weights[mask] if sample_weights is not None else None
    SMILES_list    = SMILES_list[mask]

    dataset = dict()
    dataset['features']           = features
    dataset['labels']             = labels
    dataset['SMILESs']            = SMILES_list
    dataset['IDs']                = IDs
    dataset['sample_weights']     = sample_weights
    dataset['class_distribution'] = class_distribution
    gpickle.dump(dataset, featurized_file)
    print('featurization result saved to %s' % featurized_file)
    if isinstance(dataset, dict):
        print('Dataset summary:')
        dataset_summary(dataset)

    #--- split train/test set if required---#
    trainset_file, testset_file = None, None
    if task in {'classification', 'regression'}:
        if 1.0 > partition_ratio > 0.0:
            print('Splitting trainset & testset ...')
            #--- shuffle samples before splitting ---#
            print('    shuffling samples before splitting')
            dataset_shuffled = dataset_shuffle((SMILES_list, features, labels, IDs, sample_weights))

            SMILES_list, features, labels, IDs, sample_weights = dataset_shuffled

            partition_ratio = [1.0 - partition_ratio, partition_ratio]
            dataset = (features, labels, SMILES_list, IDs, sample_weights)
            if task == 'classification':
                trainset, testset = dataset_partition_by_class(dataset, labels, partition_ratio)
            else:
                trainset, testset = dataset_partition_for_regress(dataset, partition_ratio)
            trainset_sample_num = trainset[0].shape[0]
            testset_sample_num  = testset[0].shape[0]
            print('\ntrainset size = %d, testset size = %d' % (trainset_sample_num, testset_sample_num))
            trainset_file = os.path.join(save_folder, '%s_%s_%s_trainset_%d@%s.gpkl' % (basename_without_ext, task, feature_ver, trainset_sample_num,  time_stamp))
            testset_file  = os.path.join(save_folder, '%s_%s_%s_testset_%d@%s.gpkl'  % (basename_without_ext, task, feature_ver, testset_sample_num,  time_stamp))

            dataset_train = dict()
            features_train, labels_train, SMILES_list_train, IDs_train, sample_weights_train = trainset
            dataset_train['features']       = features_train
            dataset_train['labels']         = labels_train
            dataset_train['SMILESs']        = SMILES_list_train
            dataset_train['IDs']            = IDs_train
            dataset_train['sample_weights'] = sample_weights_train

            dataset_test = dict()
            features_test, labels_test, SMILES_list_test, IDs_test, sample_weights_test = testset
            dataset_test['features']       = features_test
            dataset_test['labels']         = labels_test
            dataset_test['SMILESs']        = SMILES_list_test
            dataset_test['IDs']            = IDs_test
            dataset_test['sample_weights'] = sample_weights_test

            gpickle.dump(dataset_train, trainset_file)
            gpickle.dump(dataset_test,  testset_file)
            if isinstance(dataset_train, dict):
                print('trainset summary:')
                dataset_summary(dataset_train)
                print('\ntestset summary:')
                dataset_summary(dataset_test)

    #--- tally character set ---#
    if 0:
        print('Tally character set')
        char_freq = dict()
        for SMILES in SMILES_list:
            SMILES = canonicalize_smiles(SMILES)
            for c in SMILES:
                if c not in char_freq:
                    char_freq[c] = 1
                else:
                    char_freq[c] += 1
        print('charset size = ', len(char_freq))
        charset = list(char_freq.keys())
        charset.sort()
        charset_dict = {charset[i]:i for i in range(len(charset))}
        charset_inv_dict = {i: charset[i] for i in range(len(charset))}
        print(charset_dict)
        gpickle.dump((charset_dict, charset_inv_dict, char_freq), '%s_charset_canonicalized.gpkl' % basename)

    #--- tally SMILES length ---#
    if 0:
        print('Tally SMILES length')
        smiles_lens = [len(canonicalize_smiles(SMILES)) for SMILES in SMILES_list]
        print('max len = %d' % max(smiles_lens))
        print('min len = %d' % min(smiles_lens))
        print('ave len = %0.1f' % np.mean(smiles_lens))
        print('len std = %0.1f' % np.std(smiles_lens))

    #--- check whether conflict data exists ---#
    if 0:
        SMILES_dict = dict()
        conflict_num, duplicate_num = 0, 0
        for i in range(sample_num):
            s = SMILES_list[i]
            if s not in SMILES_dict:
                SMILES_dict[s] = labels[i]
            else:
                if labels[i] == SMILES_dict[s]:
                    # print('Duplicate@%d: %s' % (i, s))
                    duplicate_num += 1
                else:
                    SMILES_dict[s] = max(SMILES_dict[s], labels[i])
                    print('Conflict@%d (gt=%d, %d): %s' % (i, labels[i], SMILES_dict[s], s))
                    conflict_num += 1
        print('Total duplicate = %d' % duplicate_num)
        print('Total conflict  = %d' % conflict_num)
        gpickle.dump(SMILES_dict, '%s_SMILES_gt_dict.gpkl' % basename)


    #--- clean ---#
    sys.stdout = stdout_tap.stream
    sys.stderr = stderr_tap.stream
    stdout_tap_log.close()
    stderr_tap_log.close()

    #--- return values ---#
    rval = [featurized_file]
    if trainset_file is not None:
        rval.append(trainset_file)
        rval.append(testset_file)
    return rval


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-csv_file',               default=None, type=str, help='input csv file')
    argparser.add_argument('-smiles_col_name',        default='cleaned_smiles', type=str, help='column name for SMILES, required')
    argparser.add_argument('-label_col_name',         default='label', type=str, help='column name for label, optional')
    argparser.add_argument('-id_col_name',            default='ID', type=str, help='column name for sample ID, optional')
    argparser.add_argument('-sample_weight_col_name', default='sample weight', type=str, help='column name for sample weight, optional')
    argparser.add_argument('-task',              default='classification', type=str, help='classification/regression/prediction')

    argparser.add_argument('-feature_ver',       default='2', type=str, help='{1|2|0}, different featurization method, 1 = kekulized smiles, 2 = graph, 0 = deepchem fingerprint')
    argparser.add_argument('-use_chirality',     default='true', type=str, help='whether consider chirality for feature ver 0 and scaffold analysis')
    argparser.add_argument('-addH',              default='true', type=str, help='whether add back hydrogen atoms when canonicalize molecule')

    argparser.add_argument('-partition_ratio',   default=0, type=float, help='percent of samples for testing, any value <= 0 or >= 1 will disable the dataset splitting procedure')
    argparser.add_argument('-atom_dict_file',    default='atom_dict.gpkl', type=str, help='path of atom dict file')
    argparser.add_argument('-save_root_folder',  default=None, type=str)


    argparser.add_argument('-label_min',  default=None, type=float)
    argparser.add_argument('-class_num',  default=None, type=int)
    argparser.add_argument('-doc',        default=None, type=str)

    args = argparser.parse_args()
    rval = data_preprocess(args)
    print(rval)

    print('\nAll done~')
