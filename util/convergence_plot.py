# coding:utf-8
"""
Convergence curve plot
Created  :   6, 26, 2019
Revised  :   6, 26, 2019
Author   :  David Leon (dawei.leng@ghddi.org)
All rights reserved
-------------------------------------------------------------------------------
"""
__author__ = 'dawei.leng'
import os, numpy as np
if os.name == 'posix':
    import matplotlib
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy as sp
import scipy.stats
from pytorch_ext.util import gpickle

def median_smooth(xs, w=1):
    """
    Smooth 1D curve by median filtering
    :param xs: instance of list or np.array
    :param w:  median filter window size
    :return: smoothed xs
    """
    xs_smooth = []
    n = len(xs)
    for i in range(n):
        start = i - w
        end = i + w
        if start < 0 or end > n - 1:
            x = xs[i]
        else:
            x = np.median(xs[start:end+1])
        xs_smooth.append(x)
    return xs_smooth

def compute_curve_variance(xs, w=5):
    xs_smooth = median_smooth(xs, w=w)
    dif = xs - xs_smooth
    std = np.sqrt(np.mean(dif**2))
    return std

def convergence_statistics(points, extreme='max', window=100):
    """
    compute convergence statistics
    :param points: list or np array
    :param extreme: 'max' or 'min'
    :param window:
    :return: (idx, mstd, ave_around_extreme, ave_last_window)
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    if extreme == 'max':
        idx = points.argmax()
    else:
        idx = points.argmin()
    idx_left  = max(idx - window//2, 0)
    idx_right = min(idx_left + window, len(points))
    ave_around_extreme  = points[idx_left:idx_right].mean()
    mstd_around_extreme = compute_curve_variance(points[idx_left:idx_right])
    ave_last_window     = points[-window:].mean()
    mstd_last_window    = compute_curve_variance(points[-window:])
    return idx, ave_around_extreme, mstd_around_extreme, ave_last_window, mstd_last_window

def _plot_ER(ERs, max_point=None):
    plt.plot(ERs[:max_point, 0], 'r')
    plt.plot(ERs[:max_point, 1], 'b')
    plt.legend(['ER_test', 'ER_train'])
    ER_test = ERs[:max_point, 0]
    idx, ave_around_extreme, mstd_around_extreme, ave_last_window, mstd_last_window = convergence_statistics(ER_test, extreme='min')
    plt.title('ER (min=%0.2f @%d, ave=%0.2f, mstd=%0.1f' % (ER_test[idx], idx, ave_around_extreme, mstd_around_extreme))
    plt.grid(True)
    plt.xlabel('test intervals')
    plt.ylabel('error rate')

def _plot_AuPR(AuPRs, max_point=None):
    record_num, n = AuPRs.shape
    for i in range(n):
        plt.plot(AuPRs[:max_point, i])
    # AuPR_hmean = sp.stats.hmean(AuPRs[:max_point, :], axis=1)
    AuPR_hmean = []
    for AuPR0, AuPR1 in AuPRs:
        if np.isnan(AuPR0) or np.isnan(AuPR1):
            h = np.nan
        else:
            h = sp.stats.hmean((AuPR0, AuPR1))
        AuPR_hmean.append(h)
    plt.plot(AuPR_hmean)
    plt.legend(['AuPR[%d]' % i for i in range(n)] + ['hmean'])
    AuPR_hmean = np.array(AuPR_hmean)
    idx, ave_around_extreme, mstd_around_extreme, ave_last_window, mstd_last_window = convergence_statistics(AuPR_hmean*100, extreme='max')
    plt.title('AuPR (max=%0.2f @%d, ave=%0.2f, mstd=%0.1f)' % (AuPR_hmean[idx]*100, idx, ave_around_extreme, mstd_around_extreme))
    plt.grid(True)
    plt.autoscale(enable=True, axis='x', tight=True)

def _plot_AuROC(AuROCs, max_point=None):
    record_num, n = AuROCs.shape
    for i in range(n):
        plt.plot(AuROCs[:max_point, i])
    plt.legend(['AuROC[%d]' % i for i in range(n)])
    plt.title('AuROC curves')
    plt.grid(True)
    plt.autoscale(enable=True, axis='x', tight=True)

def _plot_PRF1(PRFs_per_class, class_idx, max_point=None):
    PRFs = PRFs_per_class[class_idx]
    PRFs = np.array(PRFs)
    plt.plot(PRFs[:, 0][:max_point])
    plt.plot(PRFs[:, 1][:max_point])
    plt.plot(PRFs[:, 2][:max_point])
    plt.legend(['P', 'R', 'F1'])
    plt.title('P/R/F1 of class_%s' % class_idx)
    plt.grid(True)
    plt.autoscale(enable=True, axis='x', tight=True)

def _plot_enrichment_curves(relative_enrichments, max_point=None, prefix='top-p'):
    positive_ratio = None
    for key in relative_enrichments:
        relative_enrichment = []
        N = len(relative_enrichments[key])
        M = N if max_point is None else max_point
        if M < 0:
            M = M + N
        for i in range(M):
            positive_ratio = relative_enrichments[key][i][1]
            relative_enrichment.append(relative_enrichments[key][i][2])
        plt.plot(relative_enrichment)
    plt.legend(list(relative_enrichments.keys()))
    if positive_ratio is not None:
        if positive_ratio == 0.0:
            plt.title('%s enrichment curves (sup = +inf)' % (prefix, ))
        else:
            plt.title('%s enrichment curves (sup = %0.2f)' % (prefix, (100.0/positive_ratio)))
    else:
        plt.title('%s percent enrichment curves' % prefix)
    plt.grid(True)
    plt.autoscale(enable=True, axis='x', tight=True)

def _plot_weighted_avg_PRF1(weighted_avg_PRFs, max_point=None):
    weighted_avg_PRFs = np.array(weighted_avg_PRFs)
    plt.plot(weighted_avg_PRFs[:, 0][:max_point])
    plt.plot(weighted_avg_PRFs[:, 1][:max_point])
    plt.plot(weighted_avg_PRFs[:, 2][:max_point])
    plt.legend(['weighted avg. P', 'weighted avg. R', 'weighted avg. F1'])
    plt.title('Weighted Average Performance over all classes')
    plt.grid(True)
    plt.autoscale(enable=True, axis='x', tight=True)

def _plot_macro_avg_PRF1(macro_avg_PRFs, max_point=None):
    macro_avg_PRFs = np.array(macro_avg_PRFs)
    plt.plot(macro_avg_PRFs[:, 0][:max_point])
    plt.plot(macro_avg_PRFs[:, 1][:max_point])
    plt.plot(macro_avg_PRFs[:, 2][:max_point])
    plt.legend(['macro avg. P', 'macroavg. R', 'macro avg. F1'])
    plt.title('Macro Average Performance over all classes')
    plt.grid(True)
    plt.autoscale(enable=True, axis='x', tight=True)

def _parse_enrichment_line(line, prefix='Alpha_enrichment'):
    idx = line.find(prefix)
    substrs = line[idx + len(prefix) + 2:].split('@')
    key = substrs[1]
    substrs2 = substrs[0][1:-1].split(',')
    positive_ratio = None
    if len(substrs2) == 3:
        key_acc, positive_ratio, relative_enrichment = substrs2
        positive_ratio = float(positive_ratio)
    else:
        key_acc, relative_enrichment = substrs2  # for back-compatibility
    key_acc = float(key_acc)
    relative_enrichment = float(relative_enrichment)
    return key, key_acc, positive_ratio, relative_enrichment

def plot_convergence_curve(log_file, folder=None, max_point=None, bbox_tight=False, save_fig=False, noshow=False, summary=True):
    """
    Plot convergence metric curves. By default the figures will be displayed on the screen, not saved into disk.
    You can use this function as log file parser by setting `save_fig=False` and `noshow=True`
    :param log_file: path of the *.log file
    :param folder:   in which the figures will be saved
    :param max_point: maximum limit of the x-axis
    :param save_fig: whether save the figures into disk
    :param noshow:  whether disable plotting the figures on the screen
    :return: (AuROCs, AuPRs, PRFs_per_class, alpha_enrichments)
             AuROCs, AuPRs : both numpy array with shape (record_num, class_num)
             PRFs_per_class, alpha_enrichments: both dict
    """
    if log_file is None:
        raise ValueError('log_file is not specified')
    elif not os.path.exists(log_file):
        raise ValueError('log_file = %s does not exist' % log_file)
    else:
        pass
    folder0, basename = os.path.split(os.path.abspath(log_file))
    if folder is None:
        folder = folder0

    train_process_info      = []
    AuROCs                  = []
    AuPRs                   = []
    PRFs_per_class          = dict()
    alpha_enrichments       = dict()
    topk_enrichments        = dict()
    score_level_enrichments = dict()
    weighted_avg_PRFs       = []
    macro_avg_PRFS          = []
    with open(log_file, mode='rt', encoding='utf8') as f:
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                if 'ER_test' in line and 'ER_train' in line:
                    substrs = line[22:].split(',')
                    ER_test = float(substrs[0].split('=')[1])
                    ER_train = float(substrs[1].split('=')[1])
                    batch = float(substrs[2].split('=')[1])
                    time_cost = float(substrs[3].split('=')[1][:-4])
                    speed = float(substrs[4].split('=')[1][:-10])
                    train_process_info.append([ER_test, ER_train, batch, time_cost, speed])
                if 'AuROCs' in line and 'AuPRs' in line:
                    line = line.replace('，', ',')
                    startidx = line.find('[') + 1
                    endidx   = line.find(']')
                    substrs  = line[startidx:endidx].split(',')
                    aurocs = [float(s) for s in substrs]
                    line = line[endidx+1:]
                    startidx = line.find('[') + 1
                    endidx   = line.find(']')
                    substrs  = line[startidx:endidx].split(',')
                    auprs = [float(s) for s in substrs]
                    AuROCs.append(aurocs)
                    AuPRs.append(auprs)
                if 'class_' in line and 'P = ' in line and 'R = ' in line:
                    startidx = line.find('class_')
                    classidx = line[startidx+6]
                    startidx = line.find('P = ')
                    endidx = line.find(',')
                    P = float(line[startidx:endidx].split('=')[1])
                    line = line[endidx+1:]
                    startidx = line.find('R = ')
                    endidx = line.find(',')
                    R = float(line[startidx:endidx].split('=')[1])
                    line = line[endidx+1:]
                    startidx = line.find('F1 = ')
                    endidx = line.find(',')
                    F1 = float(line[startidx:endidx].split('=')[1])
                    if classidx in PRFs_per_class:
                        PRFs_per_class[classidx].append((P, R, F1))
                    else:
                        PRFs_per_class[classidx] = [(P, R, F1)]
                if 'Alpha_enrichment' in line:
                    key, key_acc, positive_ratio, relative_enrichment = _parse_enrichment_line(line, prefix='Alpha_enrichment')
                    if key not in alpha_enrichments:
                        alpha_enrichments[key] = [[key_acc, positive_ratio, relative_enrichment]]
                    else:
                        alpha_enrichments[key].append([key_acc, positive_ratio, relative_enrichment])
                if 'Topk_enrichment' in line:
                    key, key_acc, positive_ratio, relative_enrichment = _parse_enrichment_line(line, prefix='Topk_enrichment')
                    if key not in topk_enrichments:
                        topk_enrichments[key] = [[key_acc, positive_ratio, relative_enrichment]]
                    else:
                        topk_enrichments[key].append([key_acc, positive_ratio, relative_enrichment])
                if 'Score_level_enrichment' in line:
                    key, key_acc, positive_ratio, relative_enrichment = _parse_enrichment_line(line, prefix='Score_level_enrichment')
                    if key not in score_level_enrichments:
                        score_level_enrichments[key] = [[key_acc, positive_ratio, relative_enrichment]]
                    else:
                        score_level_enrichments[key].append([key_acc, positive_ratio, relative_enrichment])
                if 'weighted_avg_' in line:
                    startidx = line.find('weighted_avg_P = ')
                    endidx = line.find(',')
                    waP = float(line[startidx:endidx].split('=')[1])
                    line = line[endidx+1:]
                    startidx = line.find('weighted_avg_R = ')
                    endidx = line.find(',')
                    waR = float(line[startidx:endidx].split('=')[1])
                    line = line[endidx+1:]
                    startidx = line.find('weighted_avg_F1 = ')
                    waF1 = float(line[startidx::].split('=')[1])
                    weighted_avg_PRFs.append((waP, waR, waF1))
                if 'macro_avg_' in line:
                    startidx = line.find('macro_avg_P = ')
                    endidx = line.find(',')
                    maP = float(line[startidx:endidx].split('=')[1])
                    line = line[endidx+1:]
                    startidx = line.find('macro_avg_R = ')
                    endidx = line.find(',')
                    maR = float(line[startidx:endidx].split('=')[1])
                    line = line[endidx+1:]
                    startidx = line.find('macro_avg_F1 = ')
                    maF1 = float(line[startidx::].split('=')[1])
                    macro_avg_PRFS.append((maP, maR, maF1))


    print('total %d records located' % len(train_process_info))
    if len(train_process_info) <= 0:
        return None
    train_process_info = np.array(train_process_info)
    AuROCs, AuPRs = np.array(AuROCs), np.array(AuPRs)
    weighted_avg_PRFs, macro_avg_PRFs = np.array(weighted_avg_PRFs), np.array(macro_avg_PRFS)

    #--- plot ER curve ---#
    figs = []
    bbox_inches = 'tight' if bbox_tight else None
    if save_fig or not noshow:
        if len(train_process_info) > 0:
            print('Average test time cost = %0.2f mins' % np.mean(train_process_info[:, 3]))
            print('Average test speed     = %0.2f samples/s' % np.mean(train_process_info[:, 4]))
            figs.append(plt.figure())
            _plot_ER(train_process_info, max_point=max_point)
            if save_fig:
                plt.savefig(os.path.join(folder, '%s_ER_curve.svg'% basename), bbox_inches=bbox_inches)
            # plt.text(0,0,'Average test time cost = %0.2f mins' % np.mean(train_process_info[:,3]))

        #--- plot AuROC curves ---#
        if AuROCs.size > 0:
            figs.append(plt.figure())
            _plot_AuROC(AuROCs, max_point=max_point)
            if save_fig:
                plt.savefig(os.path.join(folder, '%s_AuROC_curve.svg' % basename), bbox_inches=bbox_inches)

        #--- plot AuPR curves ---#
        if AuPRs.size > 0:
            figs.append(plt.figure())
            _plot_AuPR(AuPRs, max_point=max_point)
            if save_fig:
                plt.savefig(os.path.join(folder, '%s_AuPR_curve.svg' % basename), bbox_inches=bbox_inches)

        #--- plot P/R/F1 ---#
        for class_idx in PRFs_per_class:
            if len(PRFs_per_class[classidx]) > 0:
                figs.append(plt.figure())
                _plot_PRF1(PRFs_per_class, class_idx, max_point=max_point)
                if save_fig:
                    plt.savefig(os.path.join(folder, '%s_PRF_curve_class_%s.svg' % (basename, class_idx)), bbox_inches=bbox_inches)

        #--- plot relative enrichment factor curves ---#
        if len(alpha_enrichments) > 0:
            figs.append(plt.figure())
            _plot_enrichment_curves(alpha_enrichments, max_point=max_point, prefix='top-p')
            if save_fig:
                plt.savefig(os.path.join(folder, '%s_relative_enrichment_factor.svg' % basename), bbox_inches=bbox_inches)

        #--- plot topk relative enrichment factor curves ---#
        if len(topk_enrichments) > 0:
            figs.append(plt.figure())
            _plot_enrichment_curves(topk_enrichments, max_point=max_point, prefix='top-k')
            if save_fig:
                plt.savefig(os.path.join(folder, '%s_topk_relative_enrichment_factor.svg' % basename), bbox_inches=bbox_inches)

        #--- plot score level relative enrichment factor curves ---#
        if len(score_level_enrichments) > 0:
            figs.append(plt.figure())
            _plot_enrichment_curves(score_level_enrichments, max_point=max_point, prefix='score level')
            if save_fig:
                plt.savefig(os.path.join(folder, '%s_score_level_relative_enrichment_factor.svg' % basename), bbox_inches=bbox_inches)

        #--- plot Weighted Average Performance curves ---#
        if len(weighted_avg_PRFs) > 0:
            figs.append(plt.figure())
            _plot_weighted_avg_PRF1(weighted_avg_PRFs, max_point=max_point)
            if save_fig:
                plt.savefig(os.path.join(folder, '%s_weighted_avg_PRF_curve.svg' % basename), bbox_inches=bbox_inches)
        
        #--- plot Macro Average Performance curves ---#
        if len(macro_avg_PRFs) > 0:
            figs.append(plt.figure())
            _plot_macro_avg_PRF1(macro_avg_PRFs, max_point=max_point)
            if save_fig:
                plt.savefig(os.path.join(folder, '%s_macro_avg_PRF_curve.svg' % basename), bbox_inches=bbox_inches)

        #--- plot summary ---#
        if summary:
            fig = plt.figure(figsize=(18, 9))
            figs.append(fig)
            if len(alpha_enrichments) != 0: # binary classification
                plt.subplot(2,3,1)
                _plot_AuPR(AuPRs, max_point=max_point)
                plt.subplot(2,3,4)
                if len(score_level_enrichments) > 0:
                    _plot_enrichment_curves(score_level_enrichments, max_point=max_point, prefix='score level')
                else:
                    _plot_AuROC(AuROCs, max_point=max_point)
                plt.subplot(2,3,2)
                _plot_enrichment_curves(alpha_enrichments, max_point=max_point, prefix='top-p')
                plt.subplot(2,3,5)
                _plot_enrichment_curves(topk_enrichments, max_point=max_point, prefix='top-k')
                plt.subplot(2,3,3)
                _plot_PRF1(PRFs_per_class, '0', max_point=max_point)
                plt.subplot(2,3,6)
                _plot_PRF1(PRFs_per_class, '1', max_point=max_point)
            else:                           # multi-classification
                plt.subplot(2,2,1)
                _plot_AuPR(AuPRs, max_point=max_point)
                plt.subplot(2,2,3)
                _plot_AuROC(AuROCs, max_point=max_point)
                plt.subplot(2,2,2)
                _plot_weighted_avg_PRF1(weighted_avg_PRFs, max_point=max_point)
                plt.subplot(2,2,4)
                _plot_macro_avg_PRF1(macro_avg_PRFs, max_point=max_point)
            fig.text(0.5, 0.05, 'RUN = '+basename, fontsize=10, verticalalignment="center", horizontalalignment="center")
            if save_fig:
                plt.savefig(os.path.join(folder, '%s_summary.svg' % basename), bbox_inches=bbox_inches)
        if not noshow:
            plt.show()

    for fig in figs:
        plt.close(fig)
    r = (AuROCs, AuPRs, PRFs_per_class, alpha_enrichments, topk_enrichments, score_level_enrichments, weighted_avg_PRFs, macro_avg_PRFs)
    result_file = log_file + '_convergence_statistics.gpkl'
    gpickle.dump(r, result_file)
    return r

def _scan_file(folder, ext='.log'):
    result = []
    flist = os.listdir(folder)
    for f in flist:
        if f.endswith(ext):
            result.append(os.path.join(folder, f))
    return result

def _scan_subfolder(folder):
    result = []
    flist = os.listdir(folder)
    for f in flist:
        subfolder = os.path.join(folder, f)
        if os.path.isdir(subfolder):            
            result.append(subfolder)
    return result


if __name__ == '__main__':
    import os, argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-log', type=str, nargs='+', help='log file path, two or more paths are seperated by blank', default=None)
    argparser.add_argument('-d', type=int, default=None, help='folder scan depth, None = auto')
    argparser.add_argument('-max_point', type=int, default=None)
    argparser.add_argument('-tight', action='store_true', help='whether save figure with white margin trimmed')
    argparser.add_argument('-savefig', action='store_true', help='whether save figures into disk')
    argparser.add_argument('-noshow',  action='store_true', help='whether disable on screen plotting')
    arg = argparser.parse_args()
    log_files  = arg.log
    max_point  = arg.max_point
    save_fig   = arg.savefig
    noshow     = arg.noshow
    bbox_tight = arg.tight
    depth      = arg.d

    # log_file = r"C:\Users\lengdawei\Work\Project\DWGN\train_results\[MRUN 0-10, TB18_Se951_1um_w2]_VS_classification_model_4v4@2020-12-25_17-45-20.605720.log"
    # log_files = [log_file]
    # save_fig = True
    # bbox_tight = True

    if log_files is None or len(log_files) < 1:
        raise ValueError('At least 1 log file must be specified by -log input')

    if len(log_files) == 1:
        log_file = log_files[0]
        if os.path.isdir(log_file):
            folder = log_file
            log_files = _scan_file(folder, '.log')
            if depth is None:
                if len(log_files) == 0:  # check one more depth
                    flist = os.listdir(folder)
                    for f in flist:
                        subfolder = os.path.join(folder, f)
                        if os.path.isdir(subfolder):
                            log_files.extend(_scan_file(subfolder, '.log'))
            else:
                folders = [folder]
                while depth >= 0:
                    folder_next_depth = []
                    for folder in folders:
                        log_files.extend(_scan_file(folder, '.log'))
                        if depth > 0:
                            folder_next_depth.extend(_scan_subfolder(folder))
                    folders = folder_next_depth
                    depth -= 1
            print('total %d log files located' % (len(log_files)))

    if os.name == 'posix':  # linux
        save_fig = True
        noshow   = True

    for log_file in log_files:
        print('\nAnalyzing log file = %s' % log_file)
        r = plot_convergence_curve(log_file, max_point=max_point, bbox_tight=bbox_tight, save_fig=save_fig, noshow=noshow)
        # result_file = log_file+'_convergence_statistics.gpkl'
        # gpickle.dump(r, result_file)
        print('Convergence plot done for %s' % log_file)

    print('All done~')
