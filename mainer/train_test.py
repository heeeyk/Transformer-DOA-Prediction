import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import evaluate
from loader import data_loader1
from model.tst import transformer

import imp

imp.reload(transformer)

def train(X, X2, model_name, device, epoch, lr, pre_train=False, epoch_pth=None, pre_tr_times=0, vaild_label=0):
    print("train begin")
    model2 = 0
    t = torch.tensor(list(range(180))).float().to(device)
    ist, isp = data_loader1.time_devide(case_nums=50, traindata="vaild")

    model2 = transformer.Transformer(
        d_input=2,
        d_model=32,
        d_output=1,
        q=8, v=8, h=8, N=1,
        attention_size=90,
        pe="regular",
        dropout=0.1,
        chunk_mode=None
    ).to(device).train()

    if pre_train:
        model2.load_state_dict(torch.load(f'{epoch_pth}'))
        print(pre_tr_times)

        pre_tr_times = 0

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model2.parameters(), lr=lr)

    for i in range(1, epoch + 1):
        for seq, labels in tqdm(X):
            optimizer.zero_grad()
            labels = labels.to(device)
            seq = seq.to(device)
            y_pred, _ = model2(seq[:, :, :2])
            y_pred = y_pred.view(y_pred.shape[0])
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        model2.batch_size = 512
        vaild_out = vaild(X=X2, model=model2, device=device)
        vaild_access = evaluate.Evalulate(vaild_label, vaild_out, ist, isp, case_num=len(vaild_out))
        mdpe, mdape, rmse = vaild_access.loss()
        print(f"第{i + pre_tr_times}：MDPE={mdpe}\n", f"MDAPE={mdape}\n", f"RMSE={rmse}\n")

        model2 = model2.train()
        torch.save(model2.state_dict(), f'/home/user02/HYK/bis_transformer/output/epoch{i + pre_tr_times}.pth')

    return


def vaild(X, model, device):
    vaild_output = []
    for _ in range(len(X)):
        vaild_output.append([])
    model_vaild = model.eval()

    for j in tqdm(range(len(X))):
        for seq, labels in X[j]:
            seq = seq.to(device)
            with torch.no_grad():
                y_pred, _ = model_vaild(seq[:, :, :2])
                y_pred = y_pred.view(y_pred.shape[0])
                vaild_output[j].extend(y_pred.tolist())

    return vaild_output


def test(X, epoch_pth, device, test_batch):
    print("test begin")
    test_output = []
    for _ in range(len(X)):
        test_output.append([])

    model2 = transformer.Transformer(
        d_input=2,
        d_model=32,
        d_output=1,
        q=8, v=8, h=8, N=1,
        attention_size=90,
        pe="regular",
        dropout=0.1,
        chunk_mode=None
        ).to(device).eval()
    model2.load_state_dict(torch.load(f'{epoch_pth}'))


    for j in tqdm(range(test_batch)):
        for seq, labels in X[j]:
            seq = seq.to(device)
            with torch.no_grad():
                y_pred, _ = model2(seq[:, :, :2])
                y_pred = y_pred.view(y_pred.shape[0])
                test_output[j].extend(y_pred.tolist())

    return test_output

