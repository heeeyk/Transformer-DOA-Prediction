import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.tranlstm import tranlstm as transformer
from model.tranlstm import criterions, params_save

import imp
imp.reload(transformer)


class Trainer:
    def __init__(self, config):
        self.model_name = config.model_name
        self.device = config.device
        self.epoch = config.train_epoch
        self.pre_train = config.pre_train
        self.pre_tr_times = config.pre_tr_times
        self.save_pth = f"/home/user02/HYK/bis_transformer/output/{config.model_name}"
        self.t_dim = 5

        args = params_save.Params.tft()

        self.loss_quan = criterions.QuantileLoss(args).to(self.device)
        self.loss_mse = nn.MSELoss()
        self.model = transformer.LstmVsn(config=args).to(self.device)

        # 参数初始化
        self.model.apply(weights_init)

    @staticmethod
    def weighted_mse_loss(inputs, targets, weights=None):
        loss = (inputs - targets) ** 2
        loss1 = 0
        if weights is not None:
            loss1 = loss * weights.expand_as(loss)
        loss += loss1/10
        return torch.mean(loss1)

    def train(self, X, X2, model_file, best_loss, config, p):
        print("train begin")

        model = self.model.train()

        if config.pre_train:
            model.load_state_dict(torch.load(model_file))
            best_loss = np.loadtxt(f"{self.save_pth}/loss.txt")[0]
            print(best_loss)
            print(self.pre_tr_times)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        for i in range(1, self.epoch + 1):
            loss = 0

            for seq, labels in tqdm(X):
                optimizer.zero_grad()

                labels = labels.to(self.device)
                seq = seq.to(self.device)
                x1 = seq[:, :, :self.t_dim]
                # x1:(batchsize, 180, 2)
                x2 = seq[:, 0, self.t_dim:]

                rnn_out = model.forward(x1, x2)

                mloss = 0.8 * self.loss_mse(rnn_out, labels.unsqueeze(-1)) + 0.2/2048 * sum(self.loss_quan(rnn_out, labels.unsqueeze(-1)))

                w = 1 / p[np.trunc(labels.cpu().detach().numpy()).astype(np.int8)-1]
                # mloss = self.weighted_mse_loss(rnn_out, labels.unsqueeze(-1), weights=torch.tensor(w).unsqueeze(-1).to(self.device))
                mloss.backward()
                optimizer.step()

                loss += mloss.detach().item()

            vaild_loss, vaild_mse = self.vaild_full(X=X2, model=model, p=p)
            model = model.train()

            if vaild_mse < best_loss:
                print("new")
                best_loss = vaild_mse
                np.savetxt(f"{self.save_pth}/loss.txt", np.asarray([vaild_mse, vaild_mse]))
                torch.save(model.state_dict(), f'{self.save_pth}/model/best_epoch.pth')
            if i % 5 == 0:
                torch.save(model.state_dict(), f'{self.save_pth}/model/epoch{i + self.pre_tr_times}.pth')
            print(f"{i} train loss: {loss}")
            print(f"eval weighted loss: {vaild_loss}")
            print(f"eval mse loss: {vaild_mse}")

        return

    def vaild_full(self, X, model, p):
        model = model.eval()

        wloss = 0
        loss = 0
        for seq, labels in tqdm(X):
            seq = seq.to(self.device)
            labels = labels.to(self.device)
            x1 = seq[:, :, :self.t_dim]
            x2 = seq[:, 0, self.t_dim:]

            with torch.no_grad():

                rnn_out = model.forward(x1, x2)
                # mloss = 0.8 * self.loss_mse(rnn_out, labels.unsqueeze(-1)) + 0.2/2048 * sum(self.loss_quan(rnn_out, labels.unsqueeze(-1)))
                # mloss = self.loss_mse(rnn_out, labels.unsqueeze(-1))
                w = 1 / p[np.trunc(labels.cpu().detach().numpy()).astype(np.int8) - 1]
                mloss = self.weighted_mse_loss(rnn_out, labels.unsqueeze(-1), weights=torch.tensor(w).unsqueeze(-1).to(self.device))
                mse = self.loss_mse(rnn_out, labels.unsqueeze(-1))

                # batchloss = sum(self.loss_function(rnn_out, labels.unsqueeze(-1)))
                wloss += mloss
                loss += mse

        return wloss, loss

    def test(self, X, epoch_pth, test_batch):
        print("test begin")
        test_output = []
        for _ in range(len(X)):
            test_output.append([])

        model2 = self.model.eval()
        model2.load_state_dict(torch.load(f'{epoch_pth}'))

        for j in tqdm(range(test_batch)):
            for seq, labels in X[j]:
                seq = seq.to(self.device)
                x1 = seq[:, :, :self.t_dim]
                x2 = seq[:, 0, self.t_dim:]
                with torch.no_grad():
                    y_pred = model2(x1, x2)
                    # y_pred = y_pred.view(y_pred.shape[0])
                    test_output[j].extend(y_pred.squeeze(-1).tolist())

        return test_output


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

