import torch
import random
import numpy as np
import pandas as pd
from sklearn import metrics
device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def print_metrics_binary(y_true, predictions, verbose=0):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    f1_score=2*prec1*rec1/(prec1+rec1)
    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))
        print("f1_score = {}".format(f1_score))

    return {"acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse,
            "f1_score":f1_score}


def data_process(input_data):
    input_data = input_data.cpu().detach().numpy()
    input_data = np.where(input_data == 0, np.nan, input_data)
    input_ffill = pd.DataFrame(input_data).fillna(method='ffill').values
    input_ffill = np.where(np.isnan(input_ffill), 0, input_ffill)
    input_bfill = pd.DataFrame(input_data).fillna(method='bfill').values
    input_bfill = np.where(np.isnan(input_bfill), 0, input_bfill)

    x_last = torch.tensor(input_ffill).to(device)
    x_next = torch.tensor(input_bfill).to(device)

    return x_last, x_next


def get_median(x):
    x_median = []
    for i in range(x.shape[1]):
        median = np.median(x[:,i,:].cpu().detach().numpy(),axis=0)
        x_median.append(torch.tensor(median))
    return x_median


def hdl_time(batch_ts, input_v_size):
    time_out = torch.cat((batch_ts, torch.zeros(batch_ts.shape[0], 1, input_v_size).to(device)), 1)
    time_out_diff = np.diff(time_out.cpu().detach().numpy(), axis=1)
    time_out_diff[np.where(time_out_diff < 0)] = 0

    return torch.tensor(time_out_diff).to(device)


def aug_f(x, time_input, mask, sorted_length, n_shift):
    max_length = x.shape[1]
    x_dim = x.shape[2]
    shift_step = random.sample(range(1, 30), n_shift)

    x_shift_all = []
    time_shift_all = []
    mask_shift_all = []
    x_reverse_list = []
    time_reverse_list = []
    mask_reverse_list = []
    for i in range(n_shift):
        x_shift = []
        time_shift = []
        mask_shift = []
        for j in range(x.shape[0]):
            if i == 0:
                idx = sorted_length[j]
                tmp_seq = torch.flip(x[j, 0:idx, :], dims=[0])
                x_reverse = torch.cat((tmp_seq, torch.zeros(max_length - idx, x_dim).to(device)), 0)
                tmp_seq_t = torch.flip(time_input[j, 0:idx], dims=[0])
                time_reverse = torch.cat((tmp_seq_t, torch.zeros(max_length - idx).to(device)), 0)
                tmp_seq_m = torch.flip(mask[j, 0:idx, :], dims=[0])
                mask_reverse = torch.cat((tmp_seq_m, torch.zeros(max_length - idx, x_dim).to(device)), 0)
                x_reverse_list.append(x_reverse)
                time_reverse_list.append(time_reverse)
                mask_reverse_list.append(mask_reverse)

            if sorted_length[j] <= shift_step[i]:
                x_shift.append(x[j])
                time_shift.append(time_input[j])
                mask_shift.append(mask[j])
            else:
                shift_len = shift_step[i]
                x_tmp = torch.cat((x[j, shift_step[i]:, :], torch.zeros(shift_len, x.shape[2]).to(device)), 0)
                time_tmp = torch.cat((time_input[j, shift_step[i]:], torch.zeros(shift_len).to(device)), 0)
                mask_tmp = torch.cat((mask[j, shift_step[i]:, :], torch.zeros(shift_len, mask.shape[2]).to(device)), 0)
                x_shift.append(x_tmp)
                time_shift.append(time_tmp)
                mask_shift.append(mask_tmp)

        x_shift_all.append(torch.stack(x_shift))
        time_shift_all.append(torch.stack(time_shift))
        mask_shift_all.append(torch.stack(mask_shift))

    return torch.stack(x_reverse_list), x_shift_all, torch.stack(time_reverse_list), time_shift_all, torch.stack(mask_reverse_list), mask_shift_all


def mre_f(y_pre, y):
    return torch.sum(torch.abs(y_pre - y)) / torch.sum(torch.abs(y))
