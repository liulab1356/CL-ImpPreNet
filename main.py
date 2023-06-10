import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.utils import class_weight
from utils import setup_seed, print_metrics_binary, get_median, hdl_time, aug_f, mre_f, device
from model import Model, CLLoss
from data_process import data_process_x


def train(train_loader,
          valid_loader,
          demographic_data,
          diagnosis_data,
          idx_list,
          f_idx,
          emb_f_size,
          input_v_size,
          emb_v_size,
          proj1_e_size,
          proj2_e_size,
          base_size,
          base_emb_size,
          hid1_size,
          hid2_size,
          phi,
          drop_prob,
          lamda_p,
          lamda_i,
          lr,
          task,
          seed,
          epochs,
          file_name,
          device):

    model = Model(emb_f_size, input_v_size, emb_v_size, proj1_e_size, proj2_e_size, base_size, base_emb_size, hid1_size, hid2_size, phi, drop_prob, task).to(device)
    opt_model = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_model, milestones=[40, 60, 80, 90], gamma=0.5)

    setup_seed(seed)
    train_loss_ce = []
    train_loss_mae = []
    train_loss_mre = []
    train_loss_mse = []
    valid_loss_mae = []
    valid_loss_mre = []
    valid_loss_mse = []
    best_epoch = 0
    max_auroc = 0
    min_mse = 9999999
    n_shift = 1

    for each_epoch in range(epochs):
        batch_loss_ce = []
        batch_loss_mae = []
        batch_loss_mre = []
        batch_loss_mse = []
        model.train()

        for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.to(device)
            batch_ts = batch_ts.float().to(device)

            mask = torch.where(batch_x != -1, torch.ones(batch_x.shape).to(device),
                               torch.zeros(batch_x.shape).to(device))

            x_mean = torch.stack(get_median(batch_x)).to(device)
            x_mean = x_mean.unsqueeze(0).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2])
            batch_x = torch.where(batch_x == -1, x_mean, batch_x)

            batch_demo = []
            batch_diag = []
            for i in range(len(batch_name)):
                cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                cur_idx = cur_id + '_' + cur_ep
                idx = idx_list.index(cur_idx) if cur_idx in idx_list else None

                cur_demo = torch.tensor(demographic_data[idx], dtype=torch.float32)
                cur_diag = torch.tensor(diagnosis_data[idx], dtype=torch.float32)

                batch_demo.append(cur_demo)
                batch_diag.append(cur_diag)

            batch_demo = torch.stack(batch_demo).to(device)
            batch_diag = torch.stack(batch_diag).to(device)
            batch_base = torch.cat((batch_demo, batch_diag), 1)

            output, E_star = model(f_idx, batch_x, batch_ts, batch_base, mask)
            batch_y = batch_y.long()
            y_out = batch_y.cpu().numpy()
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_out),
                                                              y=y_out)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            ce_f = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
            mae_f = torch.nn.L1Loss(reduction='mean')
            mse_f = torch.nn.MSELoss(reduction='mean')

            sc_f = CLLoss()
            data_aug = []
            time_aug = []
            mask_aug = []
            data_reverse, data_shift, time_reverse, time_shift, mask_reverse, mask_shift = aug_f(batch_x, batch_ts, mask, sorted_length, n_shift)
            if n_shift == 1:
                data_aug.append(data_reverse)
                data_aug.append(data_shift[0])
                time_aug.append(time_reverse)
                time_aug.append(time_shift[0])
                mask_aug.append(mask_reverse)
                mask_aug.append(mask_shift[0])
                data_aug = torch.stack(data_aug)
                time_aug = torch.stack(time_aug)
                mask_aug = torch.stack(mask_aug)
            else:
                data_aug = torch.cat((data_reverse.unsqueeze(0), torch.stack(data_shift)), 0)
                time_aug = torch.cat((time_reverse.unsqueeze(0), torch.stack(time_shift)), 0)
                mask_aug = torch.cat((mask_reverse.unsqueeze(0), torch.stack(mask_shift)), 0)

            _, E_star_r = model(f_idx, data_aug[0, :, :, :], time_aug[0, :, :], batch_base, mask_aug[0, :, :, :])
            _, E_star_s = model(f_idx, data_aug[1, :, :, :], time_aug[1, :, :], batch_base, mask_aug[1, :, :, :])

            if task == 'Prediction':
                E_aug = torch.stack([E_star, E_star_r, E_star_s]).permute(1, 0, 2)
                sc_loss = sc_f(features=E_aug, labels=batch_y)
                loss_pre = ce_f(output, batch_y)
                loss = lamda_p * loss_pre + (1 - lamda_p) * sc_loss
                batch_loss_ce.append(loss_pre.cpu().detach().numpy())
            if task == 'Imputation':
                E_aug = torch.stack([E_star, E_star_r, E_star_s]).permute(1, 0, 2, 3)
                sc_loss = sc_f(features=E_aug, labels=None)
                x_hat = output[:, :, :input_v_size]
                loss_mae = mae_f(mask * x_hat, mask * batch_x)
                loss_mre = mre_f(mask * x_hat, mask * batch_x)
                loss_mse = mse_f(mask * x_hat, mask * batch_x)
                loss = lamda_i * loss_mse + (1 - lamda_i) * sc_loss
                batch_loss_mae.append(loss_mae.cpu().detach().numpy())
                batch_loss_mre.append(loss_mre.cpu().detach().numpy())
                batch_loss_mse.append(loss_mse.cpu().detach().numpy())

            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

        train_loss_ce.append(np.mean(np.array(batch_loss_ce)))
        train_loss_mae.append(np.mean(np.array(batch_loss_mae)))
        train_loss_mre.append(np.mean(np.array(batch_loss_mre)))
        train_loss_mse.append(np.mean(np.array(batch_loss_mse)))

        # scheduler.step()
        with torch.no_grad():
            batch_loss_mae = []
            batch_loss_mre = []
            batch_loss_mse = []
            y_true = []
            y_pred = []
            model.eval()

            for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.to(device)
                batch_ts = batch_ts.float().to(device)

                mask = torch.where(batch_x != -1, torch.ones(batch_x.shape).to(device),
                                   torch.zeros(batch_x.shape).to(device))

                x_mean = torch.stack(get_median(batch_x)).to(device)
                x_mean = x_mean.unsqueeze(0).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2])
                batch_x = torch.where(batch_x == -1, x_mean, batch_x)

                batch_demo = []
                batch_diag = []
                for i in range(len(batch_name)):
                    cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                    cur_idx = cur_id + '_' + cur_ep
                    idx = idx_list.index(cur_idx) if cur_idx in idx_list else None

                    cur_demo = torch.tensor(demographic_data[idx], dtype=torch.float32)
                    cur_diag = torch.tensor(diagnosis_data[idx], dtype=torch.float32)

                    batch_demo.append(cur_demo)
                    batch_diag.append(cur_diag)

                batch_demo = torch.stack(batch_demo).to(device)
                batch_diag = torch.stack(batch_diag).to(device)
                batch_base = torch.cat((batch_demo, batch_diag), 1)

                output, _ = model(f_idx, batch_x, batch_ts, batch_base, mask)

                if task == 'Prediction':
                    batch_y = batch_y.long()
                    y_pred.append(output)
                    y_true.append(batch_y)
                if task == 'Imputation':
                    x_hat = output[:, :, :input_v_size]
                    loss_mae = mae_f(mask * x_hat, mask * batch_x)
                    loss_mre = mre_f(mask * x_hat, mask * batch_x)
                    loss_mse = mse_f(mask * x_hat, mask * batch_x)
                    batch_loss_mae.append(loss_mae.cpu().detach().numpy())
                    batch_loss_mre.append(loss_mre.cpu().detach().numpy())
                    batch_loss_mse.append(loss_mse.cpu().detach().numpy())

        if task == 'Prediction':
            y_pred = torch.cat(y_pred, 0)
            y_true = torch.cat(y_true, 0)
            valid_y_pred = y_pred.cpu().detach().numpy()
            valid_y_true = y_true.cpu().detach().numpy()
            ret = print_metrics_binary(valid_y_true, valid_y_pred)
            cur_auroc = ret['auroc']
            if cur_auroc > max_auroc:
                best_epoch = each_epoch
                max_auroc = cur_auroc
                state = {
                    'net': model.state_dict(),
                    'optimizer': opt_model.state_dict(),
                    'epoch': each_epoch
                }
                torch.save(state, file_name)

        if task == 'Imputation':
            valid_loss_mae.append(np.mean(np.array(batch_loss_mae)))
            valid_loss_mre.append(np.mean(np.array(batch_loss_mre)))
            valid_loss_mse.append(np.mean(np.array(batch_loss_mse)))
            cur_mse = valid_loss_mse[-1]
            if cur_mse < min_mse:
                best_epoch = each_epoch
                min_mse = cur_mse
                state = {
                    'net': model.state_dict(),
                    'optimizer': opt_model.state_dict(),
                    'epoch': each_epoch
                }
                torch.save(state, file_name)

    return best_epoch


def test(test_loader,
          demographic_data,
          diagnosis_data,
          idx_list,
          f_idx,
          emb_f_size,
          input_v_size,
          emb_v_size,
          proj1_e_size,
          proj2_e_size,
          base_size,
          base_emb_size,
          hid1_size,
          hid2_size,
          phi,
          drop_prob,
          task,
          seed,
          file_name,
          device):

    setup_seed(seed)
    model = Model(emb_f_size, input_v_size, emb_v_size, proj1_e_size, proj2_e_size, base_size, base_emb_size, hid1_size, hid2_size, phi, drop_prob, task).to(device)
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    batch_loss_mae = []
    batch_loss_mre = []
    test_loss_mae = []
    test_loss_mre = []
    y_true = []
    y_pred = []

    for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.to(device)
        batch_ts = batch_ts.float().to(device)

        mask = torch.where(batch_x != -1, torch.ones(batch_x.shape).to(device),
                           torch.zeros(batch_x.shape).to(device))

        x_mean = torch.stack(get_median(batch_x)).to(device)
        x_mean = x_mean.unsqueeze(0).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2])
        batch_x = torch.where(batch_x == -1, x_mean, batch_x)

        batch_demo = []
        batch_diag = []
        for i in range(len(batch_name)):
            cur_id, cur_ep, _ = batch_name[i].split('_', 2)
            cur_idx = cur_id + '_' + cur_ep
            idx = idx_list.index(cur_idx) if cur_idx in idx_list else None

            cur_demo = torch.tensor(demographic_data[idx], dtype=torch.float32)
            cur_diag = torch.tensor(diagnosis_data[idx], dtype=torch.float32)

            batch_demo.append(cur_demo)
            batch_diag.append(cur_diag)

        batch_demo = torch.stack(batch_demo).to(device)
        batch_diag = torch.stack(batch_diag).to(device)
        batch_base = torch.cat((batch_demo, batch_diag), 1)

        output, _ = model(f_idx, batch_x, batch_ts, batch_base, mask)

        if task == 'Prediction':
            batch_y = batch_y.long()
            y_pred.append(output)
            y_true.append(batch_y)
        if task == 'Imputation':
            mae_f = torch.nn.L1Loss(reduction='mean')
            x_hat = output[:, :, :input_v_size]
            loss_mae = mae_f(mask * x_hat, mask * batch_x)
            loss_mre = mre_f(mask * x_hat, mask * batch_x)
            batch_loss_mae.append(loss_mae.cpu().detach().numpy())
            batch_loss_mre.append(loss_mre.cpu().detach().numpy())


    if task == 'Prediction':
        y_pred = torch.cat(y_pred, 0)
        y_true = torch.cat(y_true, 0)
        test_y_pred = y_pred.cpu().detach().numpy()
        test_y_true = y_true.cpu().detach().numpy()
        ret = print_metrics_binary(test_y_true, test_y_pred)
        cur_auroc = ret['auroc']
        cur_auprc = ret['auprc']
        results = {'auroc':cur_auroc, 'auprc':cur_auprc}
    if task == 'Imputation':
        test_loss_mae.append(np.mean(np.array(batch_loss_mae)))
        test_loss_mre.append(np.mean(np.array(batch_loss_mre)))
        cur_mae = test_loss_mae[-1]
        cur_mre = test_loss_mre[-1]
        results = {'mae': cur_mae, 'mre': cur_mre}

    return results


if __name__ == '__main__':
    # Define Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_f_size", type=int, default=5)
    parser.add_argument("--input_v_size", type=int, default=17)
    parser.add_argument("--emb_v_size", type=int, default=3)
    parser.add_argument("--proj1_e_size", type=int, default=51)
    parser.add_argument("--proj2_e_size", type=int, default=28)
    parser.add_argument("--base_emb_size", type=int, default=1)
    parser.add_argument("--hid1_size", type=int, default=34)
    parser.add_argument("--hid2_size", type=int, default=55)
    parser.add_argument("--phi", type=float, default=0.56)
    parser.add_argument("--drop_prob", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0023)
    parser.add_argument("--task", type=str, default='Prediction')
    parser.add_argument("--lamda_p", type=float, default=0.923)
    parser.add_argument("--lamda_i", type=float, default=0.908)
    parser.add_argument("--base_size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_x_path", type=str)
    parser.add_argument("--data_s_path1", type=str)
    parser.add_argument("--data_s_path2", type=str)
    parser.add_argument("--data_s_path3", type=str)
    parser.add_argument("--file_name", type=str)
    args = parser.parse_args()

    emb_f_size = args.emb_f_size
    input_v_size = args.input_v_size
    emb_v_size = args.emb_v_size
    proj1_e_size = args.proj1_e_size
    proj2_e_size = args.proj2_e_size
    base_size = args.base_size
    base_emb_size = args.base_emb_size
    hid1_size = args.hid1_size
    hid2_size = args.hid2_size
    phi = args.phi
    drop_prob = args.drop_prob
    lamda_p = args.lamda_p
    lamda_i = args.lamda_i
    lr = args.lr
    task = args.task
    data_x_path = args.data_x_path
    data_s_path1 = args.data_s_path1
    data_s_path2 = args.data_s_path2
    data_s_path3 = args.data_s_path3
    file_name = args.file_name
    seed = args.seed
    epochs = args.epochs

    f_list = ['Capillary refill rate', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glascow coma scale eye opening',
              'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response', 'Glucose',
              'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure',
              'Temperature', 'Weight', 'pH']
    def str_lower(str_list):
        return [str_list[i].lower() for i in range(len(str_list))]

    f_list = [str_lower(f_list[i].split()) for i in range(len(f_list))]
    lens = [len(f_list[i]) for i in range(len(f_list))]
    name_dic_feature = list(set((np.concatenate(f_list).flat)))
    f_idx_list = []
    for i in range(len(f_list)):
        tmp_idx_list = []
        for j in range(len(f_list[i])):
            idx = name_dic_feature.index(f_list[i][j]) + 1
            tmp_idx_list.append(idx)
        if len(tmp_idx_list) < max(lens):
            tmp_idx_list = tmp_idx_list + [0] * (max(lens) - len(tmp_idx_list))
        f_idx_list.append(tmp_idx_list)
    f_idx = torch.LongTensor(f_idx_list).to(device)

    train_loader, valid_loader, test_loader = data_process_x(data_x_path)
    demographic_data = np.load(data_s_path1).tolist()
    diagnosis_data = np.load(data_s_path2).tolist()
    idx_list = np.load(data_s_path3).tolist()

    best_epoch = train(train_loader, valid_loader, demographic_data, diagnosis_data, idx_list, f_idx, emb_f_size, input_v_size,
          emb_v_size, proj1_e_size, proj2_e_size, base_size, base_emb_size, hid1_size, hid2_size, phi, drop_prob, lamda_p, lamda_i, lr, task, seed, epochs, file_name, device)
    results = test(test_loader, demographic_data, diagnosis_data, idx_list, f_idx, emb_f_size, input_v_size,
          emb_v_size, proj1_e_size, proj2_e_size, base_size, base_emb_size, hid1_size, hid2_size, phi, drop_prob, task, seed, file_name, device)
    print(results)