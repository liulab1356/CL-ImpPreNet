import os
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


# MIMIC3
class Reader(object):
    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)


class InHospitalMortalityReader(Reader):
    def __init__(self, dataset_dir, listfile=None, period_length=48.0):
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._period_length = period_length

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                if float(mas[0]) > self._period_length:
                    continue
                ret.append(np.array(mas))
        if ret == []:
            return (ret, header)
        final_ret = np.stack(ret)
        return (final_ret, header)

    def read_example(self, index):
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._period_length
        y = self._data[index][1]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if v == []:
                break
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data


def load_data(reader, normalizer, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]

    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    whole_data = (np.array(data), labels)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def str2float(data):
    new_data = []
    for i in range(data.shape[0]):
        new_data_tmp = []
        for j in range(data.shape[1]):
            if data[i][j] == '':
                new_data_tmp.append(-1.0)
            elif j == 3 or j == 4 or j == 6:
                num = data[i][j].split(' ', 1)[0]
                if num in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    num_data = float(num)
                elif data[i][j] == '1.0 ET/Trach' or data[i][j] == 'ET/Trach':
                    num_data = float(2.0)
                elif num == 'None':
                    num_data = -1.0
                elif num == 'Spontaneously':
                    num_data = 4.0
                elif num == 'To Speech':
                    num_data = 3.0
                elif num == 'To Pain':
                    num_data = 2.0
                elif num == 'No':
                    num_data = 1.0
                elif num == 'Obeys':
                    num_data = 6.0
                elif num == 'Localizes':
                    num_data = 5.0
                elif num == 'Oriented':
                    num_data = 5.0
                elif num == 'Inapprop':
                    num_data = 3.0
                elif num == 'Confused':
                    num_data = 4.0
                elif num == 'Abnorm':
                    if data[i][j] == 'Abnorm extensn':
                        num_data = 2.0
                    else:
                        num_data = 3.0
                elif num == 'Abnormal':
                    if data[i][j] == 'Abnormal extension':
                        num_data = 2.0
                    else:
                        num_data = 3.0
                elif num == 'Flex-withdraws':
                    num_data = 4.0
                elif num == 'To':
                    if data[i][j] == 'To Speech':
                        num_data = 3.0
                    else:
                        num_data = 2.0
                new_data_tmp.append(float(num_data))
            else:
                new_data_tmp.append(float(data[i][j]))

        new_data.append(np.array(new_data_tmp))

    return np.array(new_data)


class VisitSequenceWithLabelDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels, ts, names, num_features, reverse=False):
        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")

        self.labels = labels
        self.seqs = seqs
        self.ts = ts
        self.names = names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index], self.ts[index], self.names[index]


def visit_collate_fn(batch):
    batch_seq, batch_label, batch_ts, batch_name = zip(*batch)

    num_features = batch_seq[0].shape[1]
    seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
    max_length = 211

    sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
    sorted_padded_seqs = []
    sorted_labels = []
    sorted_ts = []
    sorted_names = []
    sorted_lengths_new = []

    k = 0
    for i in sorted_indices:
        length = batch_seq[i].shape[0]

        if length < max_length:
            padded = np.concatenate(
                (batch_seq[i], np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
            padded_ts = np.concatenate(
                (batch_ts[i], np.zeros((max_length - length,), dtype=np.float32)), axis=0)
        elif length == max_length:
            padded = batch_seq[i]
            padded_ts = batch_ts[i]
        else:
            k = k + 1
            continue

        sorted_padded_seqs.append(padded)
        sorted_labels.append(batch_label[i])
        sorted_ts.append(padded_ts)
        sorted_names.append(batch_name[i])
        sorted_lengths_new.append(sorted_lengths[k])
        k = k + 1

    seq_tensor = np.stack(sorted_padded_seqs, axis=0)
    label_tensor = torch.FloatTensor(sorted_labels)
    ts_tensor = np.stack(sorted_ts, axis=0)

    return torch.from_numpy(seq_tensor), label_tensor, sorted_lengths_new, torch.from_numpy(ts_tensor), list(
        sorted_names)


def data_process_x(data_x_path):
    small_part = False
    time_length = 48.0

    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_x_path, 'train'),
                                             listfile=os.path.join(data_x_path, 'train_listfile.csv'),
                                             period_length=time_length)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_x_path, 'train'),
                                           listfile=os.path.join(data_x_path, 'val_listfile.csv'),
                                           period_length=time_length)

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_x_path, 'test'),
                                            listfile=os.path.join(data_x_path, 'test_listfile.csv'),
                                            period_length=time_length)

    train_raw = load_data(train_reader, None, small_part, return_names=True)
    val_raw = load_data(val_reader, None, small_part, return_names=True)
    test_raw = load_data(test_reader, None, small_part, return_names=True)

    train_raw_new = np.array([str2float(train_raw['data'][0][i][:, 1:]) for i in range(len(train_raw['data'][0]))])
    val_raw_new = np.array([str2float(val_raw['data'][0][i][:, 1:]) for i in range(len(val_raw['data'][0]))])
    test_raw_new = np.array([str2float(test_raw['data'][0][i][:, 1:]) for i in range(len(test_raw['data'][0]))])
    train_raw_ts = np.array([train_raw['data'][0][i][:, 0].astype(float) for i in range(len(train_raw['data'][0]))])
    val_raw_ts = np.array([val_raw['data'][0][i][:, 0].astype(float) for i in range(len(val_raw['data'][0]))])
    test_raw_ts = np.array([test_raw['data'][0][i][:, 0].astype(float) for i in range(len(test_raw['data'][0]))])

    train_set = VisitSequenceWithLabelDataset(train_raw_new, train_raw['data'][1], train_raw_ts, train_raw['names'], 17)
    valid_set = VisitSequenceWithLabelDataset(val_raw_new, val_raw['data'][1], val_raw_ts, val_raw['names'], 17)
    test_set = VisitSequenceWithLabelDataset(test_raw_new, test_raw['data'][1], test_raw_ts, test_raw['names'], 17)

    train_loader = DataLoader(dataset=train_set, batch_size=256, shuffle=False, collate_fn=visit_collate_fn,
                              num_workers=0)
    valid_loader = DataLoader(dataset=valid_set, batch_size=256, shuffle=False, collate_fn=visit_collate_fn,
                              num_workers=0)
    test_loader = DataLoader(dataset=test_set, batch_size=256, shuffle=False, collate_fn=visit_collate_fn,
                             num_workers=0)

    return train_loader, valid_loader, test_loader


# eICU
def embedding(root_dir):
    all_df = prepare_categorical_variables(root_dir)
    return all_df


def prepare_categorical_variables(root_dir):
    columns_ord = [ 'patientunitstayid', 'itemoffset',
    'Eyes', 'Motor', 'GCS Total', 'Verbal',
    'ethnicity', 'gender','apacheadmissiondx',
    'FiO2','Heart Rate', 'Invasive BP Diastolic',
    'Invasive BP Systolic', 'MAP (mmHg)',  'O2 Saturation',
    'Respiratory Rate', 'Temperature (C)', 'admissionheight',
    'admissionweight', 'age', 'glucose', 'pH',
    'hospitaladmitoffset',
    'hospitaldischargestatus','unitdischargeoffset',
    'unitdischargestatus']
    all_df = pd.read_csv(os.path.join(root_dir, 'all_data.csv'))

    all_df = all_df[all_df.gender != 0]
    all_df = all_df[all_df.hospitaldischargestatus != 2]
    all_df = all_df[columns_ord]

    all_df.apacheadmissiondx = all_df.apacheadmissiondx.astype(int)
    all_df.ethnicity = all_df.ethnicity.astype(int)
    all_df.gender = all_df.gender.astype(int)
    all_df['GCS Total'] = all_df['GCS Total'].astype(int)
    all_df['Eyes'] = all_df['Eyes'].astype(int)
    all_df['Motor'] = all_df['Motor'].astype(int)
    all_df['Verbal'] = all_df['Verbal'].astype(int)

    return all_df


def filter_mortality_data(all_df):
    all_df = all_df[all_df.gender != 0]
    all_df = all_df[all_df.hospitaldischargestatus!=2]
    all_df['unitdischargeoffset'] = all_df['unitdischargeoffset']/(1440)
    all_df['itemoffsetday'] = (all_df['itemoffset']/24)
    all_df.drop(columns='itemoffsetday',inplace=True)
    mort_cols = ['patientunitstayid', 'itemoffset', 'apacheadmissiondx', 'ethnicity','gender',
                'GCS Total', 'Eyes', 'Motor', 'Verbal',
                'admissionheight','admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)',
                'Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation',
                'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH',
                'unitdischargeoffset','hospitaldischargestatus']

    all_mort = all_df[mort_cols]
    all_mort = all_mort[all_mort['unitdischargeoffset'] >=2]
    all_mort = all_mort[all_mort['itemoffset']> 0]
    return all_mort

def data_extraction_mortality(time_window,root_dir):
    all_df = embedding(root_dir)
    all_mort = filter_mortality_data(all_df)
    all_mort = all_mort[all_mort['itemoffset']<=time_window]
    return all_mort


def df_to_list(df):
    grp_df = df.groupby('patientunitstayid')
    df_arr = []
    for idx, frame in grp_df:
        df_arr.append(frame)

    return df_arr


def pad(data, max_len=200):
    padded_data = []
    nrows = []
    for item in data:
        if item.shape[0] > 200:
            continue

        tmp = np.zeros((max_len, item.shape[1]))
        tmp[:item.shape[0], :item.shape[1]] = item
        padded_data.append(tmp)
        nrows.append(item.shape[0])
    padded_data = np.array(padded_data)

    return padded_data, nrows


def normalize_data_mort(data, train_idx, test_idx):
    train = data[data['patientunitstayid'].isin(train_idx)]
    test = data[data['patientunitstayid'].isin(test_idx)]

    col_used = ['patientunitstayid', 'itemoffset']

    dec_cat = ['GCS Total', 'Eyes', 'Motor', 'Verbal', 'apacheadmissiondx', 'ethnicity', 'gender']
    dec_num = ['admissionheight', 'admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)', 'Invasive BP Diastolic',
               'Invasive BP Systolic',
               'O2 Saturation', 'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH']

    col_used += dec_cat
    col_used += dec_num
    col_used += ['hospitaldischargestatus']

    train = train[col_used]
    test = test[col_used]

    train = df_to_list(train)
    test = df_to_list(test)

    train, nrows_train = pad(train)
    test, nrows_test = pad(test)

    return (train, nrows_train), (test, nrows_test)


def get_task_specific_labels(data):
    return data[:, 0, :].astype(int)


def get_data_generator(data, label, nrows, batch_size, train=True):
    data_gen = batch_generator(data, label, nrows=nrows, batch_size=batch_size)
    steps = np.ceil(len(data) / batch_size)
    return data_gen, int(steps)


def batch_generator(data, labels, nrows=None, batch_size=256, rng=np.random.RandomState(0), shuffle=True, sample=False):
    while True:
        if shuffle:
            d = list(zip(data, labels, nrows))
            random.shuffle(d)
            data, labels, nrows = zip(*d)
        data = np.stack(data)
        labels = np.stack(labels)
        for i in range(0, len(data), batch_size):
            x_batch = data[i:i + batch_size]
            y_batch = labels[i:i + batch_size]
            if nrows:
                nrows_batch = np.array(nrows)[i:i + batch_size]

            x_cat = x_batch[:, :, 1:8].astype(int)
            x_num = x_batch[:, :, 8:]
            x_time = x_batch[:, :, 0]
            yield [x_cat, x_num], y_batch, nrows_batch, x_time


def get_data(train, test, batch_size):
    nrows_train = train[1]
    nrows_test = test[1]
    n_labels = 1

    X_train = train[0][:, :, 1:-n_labels]
    X_test = test[0][:, :, 1:-n_labels]

    Y_train = get_task_specific_labels(train[0][:, :, -n_labels:])
    Y_test = get_task_specific_labels(test[0][:, :, -n_labels:])

    train_gen, train_steps = get_data_generator(X_train, Y_train, nrows_train, batch_size)
    test_gen, test_steps = get_data_generator(X_test, Y_test, nrows_test, batch_size)

    return train_gen, train_steps, test_gen, test_steps
