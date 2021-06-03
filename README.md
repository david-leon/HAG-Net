# HAG-Net
![HAG-Net Structure]((doc/HAG-Net_structure.png?raw=true))

## Contents
  * [Introduction](#introduction)
    * [Dependency](#dependency)
    * [Repo Structure](#repo-structure)
  * [Data Preprocessing](#data-preprocessing)
    * [Data Format](#data-format)
  * [Training](#training)
    * [Class Weighting](#class-weighting)
    * [Sample Weighting](#sample-weighting)
    * [Early Stop](#early-stop)
    * [Resume Training](#resume-training)
    * [Multi-fold Cross Validation](#multi-fold-cross-validation)
    * [Hyperparameter Tuning](#hyperparameter-tuning)
    * [Comparative Re-run](#comparative-re-run)

## Introduction
Code for ["Enhance Information Propagation for Graph Neural Network by Heterogeneous Aggregations"](https://arxiv.org/abs/2102.04064), also lite version of package for ligand-based virtual screening.

### Dependency
|  Package                                         |    Description                                                      |
| ------------------------------------------------ | ------------------------------------------------------------------  |
| python                                           | cpython, >=3.6.0                                                    |
| pytorch                                          | >=1.4.0                                                             |
| [pytorch_ext](https://github.com/david-leon/Pytorch_Ext) | install by `pip install git+https://github.com/david-leon/Pytorch_Ext.git` |
| numpy, scipy, matplotlib, psutil                 |  |
| rdkit                                            | required by `ligand_based_VS_data_preprocessing.py` and `data_loader.py`, install by `conda install -c rdkit rdkit` |
| torch-scatter                                    | >= 2.0.4 |
| sklearn                                          | required by `metrics.py` |
| optuna                                           | required by `grid_search.py` |
 
### Repo Structure

|  Folder/File                                     |    Description                                                      |
| ------------------------------------------------ | ------------------------------------------------------------------  |
|  docs                                            |   documentation                                                     |
|  ligand_based_VS_data_preprocessing.py           |   data preprocessig                                                 |
|  ligand_based_VS_train.py                        |   training                                                          |
|  ligand_based_VS_model.py                        |   production model definitions                                      |
|  config.py                                       |   experiment configurations                                         |
|  data_loader.py                                  |   batch data loader                                                 |
|  graph_ops.py                                    |   generic graph operations                               |
|  metrics.py                                      |   performance metric definitions                                    |
|  grid_search.py                                  |   hyperparameter tuning via grid search                             |
|  util                                            |   misc utility scripts  |

## Data Preprocessing
For input data preprocessing, use `ligand_based_VS_data_preprocessing.py`. Check
```
python ligand_based_VS_data_preprocessing.py --help
```
for argument details.

|  arg                 |    Description                                                       |
| -------------------- | -------------------------------------------------------------------  |
| -csv_file            | input csv file, each line seperated by `,`, encoded by UTF8 scheme   |
| -smiles_col_name     | column name for SMILES, default = `cleaned_smiles` |
| -label_col_name      | column name for labels, default = `label` |
| -id_col_name         | column name for sample IDs, default = `ID` |
| -sample_weight_col_name | column name for sample weights, default = `sample weight` |
| -task                | classification / prediction. If your dataset is for inference, set `-task` to `prediction` |
| -feature_ver         | featurizer version, default = `2`. `2` means graph output, `1` means kekulized SMILES output and `0` means DeepChem's fingerprint | 
| -use_chirality       | whether consider chirality for feature ver `0` and scaffold analysis |
| -addH                | whether add back hydrogen atoms when canonicalize molecule |
| -partition_ratio     | percent of samples for testing/validation, any value <= 0 or >= 1 will disable the dataset partition procedure, default = 0. Samples will always be shuffled before partitioned | 
| -atom_dict_file      | the atom dict file used for training, it's also used here for abnormal sample checking |

During the preprocessing, each input SMILES string will be validated, if invalid, a runtime warning will be raised. There will be two log files saved along with the preprocessing output file, named as `<dataset>_preprocessing_stdout@<time_stamp>.txt` and `<dataset>_preprocessing_stderr@<time_stamp>.txt`. Check any warning/error message in `<dataset>_preprocessing_stderr@<time_stamp>.txt` and check other runtime info in `<dataset>_preprocessing_stdout@<time_stamp>.txt`

### Data Format
Preprocessed data, as well as trainset & testset for training & evaluation, will be saved into `.gpkl` files with `dict` format as follows:

|  key                | Required |    Description                                                          |
| ------------------- | -------- | ----------------------------------------------------------------------  |
| features            |  Y       | extracted features. When `-feature_ver` is `1`, this field is the same with `SMILESs` below  |
| labels              |  Y       | sample ground truth labels, = `None` if there's no groud truth          |
| SMILESs             |  Y       | original SMILES strings for each sample                                 |
| IDs                 |  Y       | sample ID, = `None` if there's no sample ID data. If `None`, during training or evaluation, ID will be generated automatically for each sample by its index number, starting from 0. |
| sample_weights      |  Y       | sample weights, = `None` if there's no sample weight data               |
| class_distribution  |  N       | class distribution statistics if sample labels are given, else = `None` |


## Training
For model training, use `ligand_based_VS_train.py`.  
Check
```
python ligand_based_VS_train.py --help
```
for argument details.

|  arg                 |    Description                                                       |
| -------------------- | -------------------------------------------------------------------  |
| -device              | int, -1 means CPU, >=0 means GPU                                     |
| -model_file          | file path of initial model weights, default = `None`                 |
| -model_ver           | model version, default = `4v4`                                       |
| -class_num           | class number, default = `2` |      |
| -optimizer           | `sgd`/`adam`/`adadelta`, default is an SGD optimizer with lr = 1e-2 and momentum = 0.1, you can change learning rate via `-lr` argument |                                                    |
| -trainset            | file path of train set, saved in `.gpkl` format as in data preprocessing step, refer to [Data Format](#data-format) for data format details |
| -testset             | either <br/> 1) file path of test set, saved in `.gpkl` format as in data preprocessing step, refer to [Data Format](#data-format) for data format details <br/> 2) float in range (0, 1.0), this value will be used as partition ratio <br/> note in [Multi-fold Cross Valication](#multi-fold-cross-validation), this input will be ignored |
| -batch_size <br/> -batch_size_min | the max & min sample number for a training batch. You can set them to different values to enable variable training batch size, this will help mitigate batch size effect if your model's performance is correlated to the size of training batch |
| -batch_size_test     | sample number for a test batch, this size is fixed               | 
| -ER_start_test  | float, ER threshold below which the periodic test will be executed, for classification task  |
| -max_epoch_num       | max training epoch number |
| -save_model_per_epoch_num | float in (0.0, 1.0], default = `1.0`, means after training this portion of all training samples a checkpoint model file will be saved |
| -test_model_per_epoch_num | float in (0.0, 1.0], default = `1.0`, means after training this portion of all training samples one round of test will be excuted     |
| -save_root_folder    | root folder for training results saving, by default = `None`, <br/> **for plain training**  a `train_results` folder under the current direction will be used <br/> **for multi-fold cross validation**, a folder with name convention `train_results\<m>_fold_<task>_model_<model_ver>@<time_stamp>` will be used, in which `<m>` is the number of fold, `<task>` = `classification` , `<model_ver>` is the model version string, `<time_stamp>` is the time stamp string meaning when the running is started <br/> **for grid search**,  a folder with name convention `train_results\grid_search_<model_ver>@<time_stamp>` will be used |
| -save_folder         | folder under the `save_root_folder`, under which one `train()` run results will be actually saved, by default = `None`, folder with name convention `[<prefix>]_ligand_VS_<task>_model_<model_ver>_pytorch@<time_stamp>` will be used, in which `<prefix>` is specified by input arg `-prefix` as below | 
| -train_loader_worker | number of parallel loader worker for train set, default = `1`, increase this value if data loader takes more time than one batch training  |
| -test_loader_worker  | number of parallel loader worker for test set, default = `1`, increase this value if data loader takes more time than one batch testing | 
| -weighting_method    | method for different class weighting, default = `0`, refer to [Class Weighting](#class-weighting) for details |
| -prefix              | a convenient str flag to help you differentiate between different runs, by default = `None`, a prefix string `[model_<model_ver>]` will be used. This prefix string will be placed at the beginning of each on-screen output during training |
| -config              | str, specify the `CONFIG` class name within `config.py`, the corresponding config set will be used for the training run. By default = `None`, the `model_<model_ver>_config` set will be used automatically |
| -select_by_aupr <br/> -select_by_corr | `true` / `false`, if `true`, the best model will be selected by `AuPR` metric on test set; else, `ER` metric will be used <br/> `true` / `false`, if `true`, the best model will be selected by `CoRR` metric on test set; else, `RMSE` metric will be used
| -verbose             | int, verbose level, more high more concise the on-screen output is|
| -alpha               | list of float in range (0.0, 1.0], default=`(0.01, 0.02, 0.05, 0.1)`, specifying at top-alpha portions of the test set the relative enrichment factor will be calculated  |
| -alpha_topk          | list of int, default=`(100, 200, 500, 1000)`, specifying at top-k samples of the test set the relative enrichment factor will be calculated  |
| -**score_levels**        | list of float in range (0.0, 1.0), default=`(0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)`, relative enrichment factor will be calculated for samples with predicted score above each level |
| -early_stop          | default = `0`, set it to value > 0 to enable early stop, refer to [Early Stop](#early-stop) for details |
| -mfold               | an integer or file path of `mfold_indexs.gpkl`, default = None, refer to [Multi-fold Cross Validation](#multi-fold-cross-validation) section for details |
| -grid_search         | default = `0`, set it to value > 0 to enable hyperparameter tuning, refer to [Hyperparameter Tuning](#hyperparameter-tuning) for details |
| -**srcfolder**           | folder path for re-run, refer to [Comparative Re-run](#comparative-re-run) for details, default = `None` |
| -lr                  | learning rate for optimizer |

### Class Weighting
  * For classification task training, 3 different class weighting methods are supported:
    1) method 0: default method, use uniform weighting, set input arg as `-weighting_method 0`
    2) method 1: class weight equals to inverse of class distribution of training samples, set input arg as `-weighting_method 1`
    3) method 2: class weight equals to inverse of square root of class distribution of training samples, set input ars as `-weighting_method 2`

### Sample Weighting
  * Note the metrics calculated on train set don't count in sample weights except for the `loss` metric; test set does not support sample weights at all; for meaningful test metric results, you should organize test samples with the same weights together, and sperately run test on test sets with different weights.

### Early Stop
  * Enable early stop for training with input arg `-early_stop m`, in which `m` is the maximum number of consective test rounds without better metric got, after this number the training process will be stopped. Set `-early_stop 0` to disable early stop.

### Resume Training
  * For plain training, there're two alternative ways to resume an interrupted training process:
    1) start a new training with input arg `-model_file`, this will pratically initialize the model with weights from the `model_file`
    2) or, start a new training with input arg `-save_folder`, the training will retrieve a) the last run epoch and b) the latest model file saved in `save_folder` (logs and model files will still be saved into this folder); in this case input arg `-model_file` will be ignored.

### Multi-fold Cross Validation
  * Enable multi-fold cross validation with input arg `-mfold m`, in which `m` is the number of folds or the path of specified `mfold_indexs.gpkl` file if you want to use the same sample splitting indices among different runs.
  * When multi-fold cross validation is activated, the input train set will be divided into `m` parts, at each fold one part will be used as test set, other parts will be used as train set. Test set specified by input arg `-testset` must be `None` now.
  * Train results & log will be saved in seperate folder for each fold
  * Multi-fold cross validation is mainly designed for small data scenarios, `m` is recommended to be in range [3, 5]
  * Parallel running: multi-fold cross validation supports local parallel running
    1) First start a multi-fold training as usual, get the `save_root_folder` of this run
    2) Start one or more multi-fold trainings with all the settings the same with the first run, and explicitly set the input arg `-save_root_folder` to the root folder as above
    3) By **local**, it means the parallel syncronization is done through the local file system. On HPC or cloud enviroment, different training processes can run on different computing nodes as long as they share the same file system.
    4) arg `-shuffle_trainset` only affect the first training process.
   
### Hyperparameter Tuning
  * Auto hyperparameter tuning is supported by grid search. Enable grid search by arg `-grid_search n` in which `n` is the maximum number of trial. Set it to `0` to diable the grid search.
  * Grid search and multi-fold cross validation should not be activated simultaneously.
  * Best hyperparameter combinations will be saved into `best_trial_xxx.json` file under the training `save_root_folder`.
  * Currently hyperparameter search spaces for `model_4v4` and `model_4v7` are defined, check out the details in [`grid_search.py`](grid_search.py); note these search spaces are pretty large, mainly defined for model structure optimization, not for per dataset tuning. If you intend to do per dataset tuning, these search spaces should be narrowed down dramatically. Contact the authors if you're not capable of modifying the default search space definitions.
  * Resume running: to resume an interrupted grid search process, just start a new grid search process with input arg `-save_root_folder` set to the previous run
  * Parallel running: grid search supports local parallel running
    1) First, start a grid search as usual, get the `save_root_folder` of this run, remember that the number of trials specified by input arg `-grid_search` should be reduced now (probably = # total_planned_trials / # parallel_runs)
    2) Start one or more grid search with all the settings the same with the first run (except for `-grid_search`, you can set it to any number appropriate), and explicitly set the input arg `-save_root_folder` to the root folder as above
    3) By **local**, it means the parallel syncronization is done through the local file system. On HPC or cloud enviroment, different grid search processes can run on different computing nodes as long as they share the same file system.

### Comparative Re-run
  * For convenient re-run for comparison purpose, you can set `-srcfolder`  to the folder path of a previous run, the training program will load trainset/testset as well as `mfold_indexs.gpkl` file from there.
  * For plain training, trainset and testset will be loaded from `-srcfolder` for the new run
  * For [multi-fold cross validation](#multi-fold-cross-validation), both `trainset.gpkl` and `mfold_indexs.gpkl` will be loaded from `-srcfolder` for the new run
  * For [bundle training](#bundle-training), order of each run as well as trainset/testset for each run will be loaded from `-srcfolder`, and the new bundle training will be executed with the same order loaded.
