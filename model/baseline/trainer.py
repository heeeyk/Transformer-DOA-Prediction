import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.baseline import baseline
from model.baseline import params
import imp



class Trainer:
    def __init__(self, config):
        self.model_name = config.model_name
        self.device = config.device
        self.epoch = config.train_epoch
        self.pre_train = config.pre_train
        self.pre_tr_times = config.pre_tr_times
        self.save_pth = f"/data/HYK/DATASET/bis/output/{config.model_name}"


        args = params.Params.lstm_params()

        self.loss_function = nn.MSELoss()
        self.model = baseline.LstmModel(config=args).cuda()

        # 参数初始化
        # self.model.apply(weights_init)

    def train(self, X, X2, lr, model_file, best_loss):
        print("train begin")

        model = self.model.train()

        if self.pre_train:
            model.load_state_dict(torch.load(model_file))
            best_loss = np.loadtxt(f"{self.save_pth}/loss.txt")[0]
            print(best_loss)
            print(self.pre_tr_times)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for i in range(1, self.epoch + 1):
            loss = 0

            for seq, labels in tqdm(X):
                optimizer.zero_grad()

                labels = labels.cuda()
                seq = seq.cuda()
                x1 = seq[:, :, :2]
                # x1:(batchsize, 180, 2)
                x2 = seq[:, 0, 5:]
                rnn_out = model.forward(x1, x2)
                batchloss = self.loss_function(rnn_out, labels.unsqueeze(-1))
                # batchloss = sum(self.loss_function(rnn_out, labels.unsqueeze(-1)))
                batchloss.backward()

                optimizer.step()

                loss += batchloss.detach().item()

            vaild_loss = self.vaild_full(X=X2, model=model)
            model = model.train()

            if vaild_loss < best_loss:
                print("new")
                best_loss = vaild_loss
                np.savetxt(f"{self.save_pth}/loss.txt", np.asarray([vaild_loss, vaild_loss]))
                torch.save(model.state_dict(), f'{self.save_pth}/model/best_epoch.pth')
            torch.save(model.state_dict(), f'{self.save_pth}/model/epoch{i + self.pre_tr_times}.pth')
            print(f"{i} train loss: {loss}")
            print(f"eval loss: {vaild_loss}")

        return

    def vaild_full(self, X, model):
        model = model.eval()

        loss = 0
        for seq, labels in tqdm(X):
            seq = seq.cuda()
            labels = labels.cuda()
            x1 = seq[:, :, :2]
            x2 = seq[:, 0, 5:]

            with torch.no_grad():

                rnn_out = model.forward(x1, x2)
                batchloss = self.loss_function(rnn_out, labels.unsqueeze(-1))
                # batchloss = sum(self.loss_function(rnn_out, labels.unsqueeze(-1)))
                loss += batchloss.detach().item()

        return loss

    def test(self, X, epoch_pth, test_batch):
        print("test begin")
        test_output = []
        for _ in range(len(X)):
            test_output.append([])

        model2 = self.model.eval()
        model2.load_state_dict(torch.load(f'{epoch_pth}', map_location='cuda:0'))

        for j in tqdm(range(test_batch)):
            for seq, labels in X[j]:
                seq = seq.cuda()
                x1 = seq[:, :, :2]
                x2 = seq[:, 0, 5:]
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
