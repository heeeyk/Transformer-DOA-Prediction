import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from mainer import evaluate
from tensorboardX import SummaryWriter
from loader import data_loader1
import imp
from model.base_transformer import transformer_basic as transformer
imp.reload(transformer)


class Trainer:
    def __init__(self, model_name, device, epoch, epoch_pth, pre_train, pre_tr_times, vaild_label, args):
        self.model_name = model_name
        self.device = device
        self.epoch = epoch
        self.epoch_pth = epoch_pth
        self.pre_train = pre_train
        self.pre_tr_times = pre_tr_times
        self.vaild_label = vaild_label
        self.save_pth = f"{args.root}/output/{args.model_name}"
        self.loss_function = nn.MSELoss()
        if model_name == "tst":
            from model.tst import transformer
            self.model = transformer.Transformer(
                d_input=2,
                d_model=32,
                d_output=1,
                q=8, v=8, h=8, N=1,
                attention_size=90,
                pe="regular",
                dropout=0.1,
                chunk_mode=None
                ).cuda()

        if model_name == "base":
            from model.base_transformer import transformer_basic as transformer
            self.model = transformer.MyTransformer(
                device=self.device,
                number_time_series=2,
                seq_length=180,
                output_seq_len=1,
                d_model=32,
                n_heads=4,
                ).cuda()
            # 参数初始化
            self.model.apply(weights_init)

    def train(self, X, X2, lr, pre_file, best_loss, args, test_loader, ist, isp, test_label):
        print("train begin")
        writer = SummaryWriter('runs/train')

        model2 = self.model.train()

        if self.pre_train:
            model2.load_state_dict(torch.load(pre_file))
            best_loss = 999999999
            # best_loss = np.loadtxt(f"{self.save_pth}/loss.txt")[0]
            # print(self.pre_tr_times)

        optimizer = torch.optim.Adam(model2.parameters(), lr=lr)
        j = 0
        for i in range(1, self.epoch + 1):
            loss = 0

            for seq, labels in tqdm(X):
                optimizer.zero_grad()

                labels = labels.cuda()
                seq = seq.cuda()
                x = seq[:, :, :2]
                # t = seq[:, :, 6:]

                y_pred = model2(x, x)

                y_pred = y_pred[-1]
                single_loss = self.loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

                loss += single_loss.detach().item()
                j += 1

            vaild_loss = self.vaild_full(X=X2, model=model2, epotimes=i)
            model2 = model2.train()

            writer.add_scalar('train_loss', loss, global_step=i)
            writer.add_scalar('vaild_loss', vaild_loss, global_step=i)
            torch.save(model2.state_dict(), f'{self.save_pth}/exp{args.exp}/epoch_{i}.pth')
            # torch.save(model2.state_dict(), f'{self.save_pth}/epoch{i + self.pre_tr_times}.pth')
            if vaild_loss < best_loss:
                print("new")
                best_loss = vaild_loss
                np.savetxt(f"{self.save_pth}/loss.txt", np.asarray([vaild_loss, vaild_loss]))
                torch.save(model2.state_dict(), f'{self.save_pth}/exp{args.exp}/best_epoch.pth')

                self.real_test(ist, isp, test_loader, f'{self.save_pth}/exp{args.exp}/best_epoch.pth', test_label)
                model2 = model2.train()

            print(f"{i} train loss: {loss}")
            print(f"eval loss: {vaild_loss}")
        writer.close()
        return

    def vaild(self, X, model, device, ist, isp, epotimes):
        vaild_output = []
        for _ in range(len(X)):
            vaild_output.append([])
        model_vaild = model.eval()

        for j in tqdm(range(len(X))):
            for seq, labels in X[j]:
                seq = seq.cuda()
                x = seq[:, :, :2]
                t = seq[:, :, 2:]

                with torch.no_grad():
                    y_pred = model_vaild(x, x)
                    y_pred = y_pred.view(y_pred.shape[0])
                    vaild_output[j].extend(y_pred.tolist())

        vaild_access = evaluate.Evalulate(self.vaild_label, vaild_output, ist, isp, case_num=len(vaild_output))
        mdpe, mdape, rmse = vaild_access.loss()
        print(f"第{epotimes + self.pre_tr_times}：MDPE={mdpe}\n", f"MDAPE={mdape}\n", f"RMSE={rmse}\n")
        return

    def vaild_full(self, X, model, epotimes):
        model_vaild = model.eval()

        loss = 0
        for seq, labels in tqdm(X):
            seq = seq.cuda()
            labels = labels.cuda()
            x = seq[:, :, :2]

            with torch.no_grad():
                y_pred = model_vaild(x, x)
                y_pred = y_pred[-1]
                single_loss = self.loss_function(y_pred, labels)
                loss += single_loss

        return loss.detach().item()

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
                t = seq[:, :, 6:]
                with torch.no_grad():
                    y_pred = model2(seq[:, :, :2], seq[:, :, :2])
                    y_pred = y_pred[-1]
                    test_output[j].extend(y_pred.tolist())

        return test_output


    def real_test(self, ist, isp, test_loader, file, test_label):
        test_out = self.test(
            X=test_loader,
            epoch_pth=file,
            test_batch=len(test_loader))
        access = evaluate.Evalulate(test_out, test_label, ist, isp, case_num=len(test_loader))
        print("MDPE    MDAPE    RMSE\r")
        for i in range(4):
            x = access.loss(i)
            print("%.2f  " % x["meanMDPE"],
                  "%.2f  " % x["meanMDAPE"],
                  "%.2f  " % x["meanRMSE"],
                  "%.2f  " % x["meanMAE"])
        self.model.train()


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

