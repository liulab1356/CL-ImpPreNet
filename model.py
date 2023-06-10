import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class FeatureEmbedding(nn.Module):
    def __init__(self, emb_f_size):
        super(FeatureEmbedding, self).__init__()
        self.emb_f = nn.Embedding(34, emb_f_size)

    def forward(self, f_idx):
        emb_out = self.emb_f(f_idx)
        return emb_out


class ValueTimeEmbedding(nn.Module):
    def __init__(self, input_v_size, emb_v_size, proj_e_size):
        super(ValueTimeEmbedding, self).__init__()
        self.input_v_size = input_v_size
        for i in range(input_v_size):
            setattr(self, "emb_v1_" + str(i + 1), nn.Linear(1, int(math.sqrt(emb_v_size))))
            setattr(self, "emb_v2_" + str(i + 1), nn.Linear(int(math.sqrt(emb_v_size)), emb_v_size, bias=False))
            setattr(self, "emb_t1_" + str(i + 1), nn.Linear(1, int(math.sqrt(emb_v_size))))
            setattr(self, "emb_t2_" + str(i + 1), nn.Linear(int(math.sqrt(emb_v_size)), emb_v_size, bias=False))
        self.emb_v1 = [getattr(self, "emb_v1_" + str(i + 1)) for i in range(input_v_size)]
        self.emb_v2 = [getattr(self, "emb_v2_" + str(i + 1)) for i in range(input_v_size)]
        self.emb_t1 = [getattr(self, "emb_t1_" + str(i + 1)) for i in range(input_v_size)]
        self.emb_t2 = [getattr(self, "emb_t2_" + str(i + 1)) for i in range(input_v_size)]

        self.Wx = nn.Linear(input_v_size*emb_v_size, proj_e_size)
        self.Wt = nn.Linear(input_v_size*emb_v_size, proj_e_size)
        self.tanh = nn.Tanh()

    def forward(self, x, time):
        out = [self.emb_v2[i](self.tanh(self.emb_v1[i](x[:, :, i].unsqueeze(2)))) for i in range(self.input_v_size)]
        x_out = torch.stack(out).permute(1, 2, 0, 3)
        out = [self.emb_t2[i](self.tanh(self.emb_t1[i](time[:, :, i].unsqueeze(2)))) for i in range(self.input_v_size)]
        time_out = torch.stack(out).permute(1, 2, 0, 3)

        x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], -1)
        time_out = time_out.reshape(time_out.shape[0], time_out.shape[1], -1)
        final_out = self.Wx(x_out) + self.Wt(time_out)

        return final_out


class CrossModule(nn.Module):
    def __init__(self, emb_f_size, input_v_size, input_v_len, emb_v_size, proj_e_size):
        super(CrossModule, self).__init__()
        self.input_v_size = input_v_size
        self.vte = ValueTimeEmbedding(input_v_size, emb_v_size, proj_e_size)
        self.fe = FeatureEmbedding(emb_f_size)
        self.fe_att = nn.Linear(emb_f_size * 5, emb_v_size)
        self.vte_seq = nn.Linear(input_v_len, 1)

        self.conv = nn.Conv1d(proj_e_size, proj_e_size, kernel_size=2 * 1 + 1, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, f_idx, x, time):
        Ex = self.vte(x, time)
        Ex_ = self.vte_seq(Ex.permute(0, 2, 1))
        Ex_fea = Ex.permute(0, 2, 1)
        if f_idx == None:
            return Ex_fea.permute(0, 2, 1)

        Ef = self.fe(f_idx)
        Ef = Ef.view(self.input_v_size, -1)
        Ef = Ef.unsqueeze(0).expand(x.shape[0], Ef.shape[0], Ef.shape[1])
        Ef_ = self.fe_att(Ef)
        Ef_ = Ef_.reshape(Ef_.shape[0],-1).unsqueeze(2)

        E = torch.bmm(Ef_, Ex_.permute(0, 2, 1)) / torch.sqrt(torch.tensor(self.input_v_size).to(utils.device))
        att = self.softmax(self.conv(E))
        out = torch.bmm(att, Ex_fea).permute(0, 2, 1)

        return out


class Similarity(nn.Module):
    def __init__(self, input_v_size, base_size, base_emb_size, phi):
        super(Similarity, self).__init__()
        self.input_v_size = input_v_size
        self.phi = nn.Parameter(torch.tensor(phi), requires_grad=True)
        self.visit = nn.Linear(input_v_size, 1)
        self.proj = nn.Linear(base_size, base_emb_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, Ex, base):
        base = self.proj(base)
        visit_att = self.softmax(self.visit(Ex))
        Ex_ = torch.bmm(visit_att.permute(0, 2, 1), Ex).squeeze(1)
        Ex_all = torch.cat((Ex_, base), 1)
        adj_matrix = torch.mm(Ex_all, Ex_all.T) / torch.pow(torch.tensor(self.input_v_size).to(utils.device), 2)

        adj_matrix_out = torch.where(adj_matrix > self.phi, adj_matrix, torch.zeros(adj_matrix.shape).to(utils.device))

        return adj_matrix_out, Ex_all, base


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, task, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.task = task
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if self.task == 'Imputation':
            support = torch.matmul(input, self.weight)
            output = torch.matmul(support.permute(1, 2, 0), adj.T).permute(2, 0, 1)
        if self.task == 'Prediction':
            support = torch.mm(input, self.weight)
            output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, input_size, hid1_size, hid2_size, task):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_size, hid1_size, task)
        self.gc2 = GraphConvolution(hid1_size, hid2_size, task)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class InfoAgg(nn.Module):
    def __init__(self, input_size, hid_size):
        super(InfoAgg, self).__init__()
        self.proj = nn.Linear(hid_size, input_size)
        self.proj1 = nn.Linear(input_size, 1)
        self.proj2 = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_self, x_other):
        x_other = self.proj(x_other)
        gamma = self.sigmoid(self.proj1(x_self))
        eta = self.sigmoid(self.proj2(x_other))
        gamma = gamma / (gamma + eta)
        eta = 1 - gamma
        x_final = gamma * x_self + eta * x_other

        return x_final


class Model(nn.Module):
    def __init__(self, emb_f_size, input_v_size, emb_v_size, proj1_e_size, proj2_e_size, base_size, base_emb_size, hid1_size, hid2_size, phi, drop_p, task):
        super(Model, self).__init__()
        self.input_v_size = input_v_size
        self.base_emb_size = base_emb_size
        self.task = task
        input_v_len = 211
        self.Emb = CrossModule(emb_f_size, input_v_size, input_v_len, emb_v_size, proj1_e_size)
        if self.task == 'Imputation':
            self.Emb_last = CrossModule(emb_f_size, input_v_size, input_v_len, emb_v_size, proj2_e_size)
            self.Emb_next = CrossModule(emb_f_size, input_v_size, input_v_len, emb_v_size, proj2_e_size)
        self.Sim = Similarity(proj1_e_size, base_size, base_emb_size, phi)
        self.Gcn = GCN(proj1_e_size + base_emb_size, hid1_size, hid2_size, task)
        self.Agg = InfoAgg(proj1_e_size + base_emb_size, hid2_size)
        self.W1 = nn.Linear(proj1_e_size + base_emb_size, input_v_size)
        self.W2 = nn.Linear(proj2_e_size * 2, input_v_size)
        self.Wy = nn.Linear(proj1_e_size + base_emb_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, f_idx, x, time, xbase, mask):
        time = time.unsqueeze(2).expand(time.shape[0], time.shape[1], self.input_v_size)

        if self.task == 'Imputation':
            time = time * mask

            time_last_list = []
            time_next_list = []
            x_last_list = []
            x_next_list = []
            for i in range(x.shape[0]):
                time_last, time_next = utils.data_process(time[i])
                time_last_list.append(time_last)
                time_next_list.append(time_next)
                x_last, x_next = utils.data_process(x[i])
                x_last_list.append(x_last)
                x_next_list.append(x_next)

            time_last = torch.stack(time_last_list)
            time_next = torch.stack(time_next_list)
            x_last = torch.stack(x_last_list)
            x_next = torch.stack(x_next_list)
            time = utils.hdl_time(time, self.input_v_size)
            time_last = utils.hdl_time(time_last, self.input_v_size)
            time_next = utils.hdl_time(time_next, self.input_v_size)
        if self.task == 'Prediction':
            time = utils.hdl_time(time, self.input_v_size)

        # 1.Embedding
        Ex = self.Emb(f_idx, x, time)  # b x seq_len x input_v_size

        # 2.Similar Patients Discovery
        adj_matrix_out, Ex_all, ebase = self.Sim(Ex, xbase)
        if self.task == 'Imputation':
            ebase = ebase.unsqueeze(1).expand(ebase.shape[0], Ex.shape[1], ebase.shape[1])
            Ex = torch.cat((Ex, ebase), 2)
            x_self = Ex
            x_other = self.Gcn(Ex, adj_matrix_out)
            x_final = self.Agg(x_self, x_other)
            Ex_last = self.Emb_last(None, x_last, time_last)
            Ex_next = self.Emb_next(None, x_next, time_next)
            x_cmb = torch.cat((Ex_last, Ex_next), 2)
            # 3.Prediction
            output = self.W1(x_final) + self.W2(x_cmb)
        if self.task == 'Prediction':
            x_self = Ex_all
            x_other = self.Gcn(Ex_all, adj_matrix_out)
            x_final = self.Agg(x_self, x_other)
            x_final = self.dropout(x_final)
            # 3.Prediction
            output = self.softmax(self.Wy(x_final))

        return output, x_final


class CLLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(CLLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(utils.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(utils.device)
        else:
            mask = mask.float().to(utils.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(utils.device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        tmp = exp_logits.sum(1, keepdim=True)
        tmp = torch.where(tmp==0,torch.zeros(tmp.shape).to(utils.device).float()+1e-3,tmp)
        log_prob = logits - torch.log(tmp)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
