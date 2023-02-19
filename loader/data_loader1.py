import time

import torch
import numpy as np
from pandas import read_csv
import os
from tqdm import tqdm
import torch.utils.data as dat
import matplotlib.pyplot as plt
import pandas as pd



def normalizition(x, mu, sigma):
    # mu 均值 sigms 标准差
    x = (x - mu) / sigma
    return x


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files  # 当前路径下所有非目录子文件,列表


def data_resave(case_nums=1, traindata="train", time_step=10):
    case_id = file_name(f'/home/user02/HYK/bis/database/new_{traindata}_clean')

    for i in range(len(case_id)):
        case_id[i] = case_id[i].split('.')[0]  # 字符串转数字

    case_id = list(map(int, case_id))
    case_id.sort()
    print("file_name:", case_id)

    data_seq = [0]*case_nums
    data_label = [0]*case_nums
    for i in tqdm(range(case_nums)):
        df = read_csv(f'/home/user02/HYK/bis/database/new_{traindata}_clean/{case_id[i]}.csv')

        x_len = int(len(df.BIS) / time_step)

        # 清除异常值
        modify_RFTN = df.RFTN20_VOL.values
        modify_PPF = df.PPF20_VOL.values
        diff_RFTN = np.diff(modify_RFTN)
        diff_PPF = np.diff(modify_PPF)
        for j in range(len(diff_RFTN)):
            if diff_RFTN[j] < 0:
                temp = (modify_RFTN[j] + modify_RFTN[j + 2]) / 2
                df.loc[j + 1, "RFTN20_VOL"] = temp
            if diff_PPF[j] < 0:
                temp = (modify_PPF[j] + modify_PPF[j + 2]) / 2
                df.loc[j + 1, "PPF20_VOL"] = temp

        # 为0时刻补上-1800s的零数据
        PPF = list(np.zeros(1800))
        PPF.extend(df.PPF20_VOL.values)
        RFTN = list(np.zeros(1800))
        RFTN.extend(df.RFTN20_VOL.values)

        # 特征制作
        X1 = torch.zeros((x_len, 180))
        X2 = torch.zeros((x_len, 180))

        for x in range(1800, len(PPF)-10, time_step):
            # 从补完数据1800s（实际0s）时刻开始取数据段
            PPF_10s, RFTN_10s = [], []
            for k in range(179, -1, -1):
                # 第k个10s片段, 共180个
                PPF_10s.append((PPF[x-k*10]-PPF[x-(k+1)*10])*10)
                RFTN_10s.append((RFTN[x-k*10]-RFTN[x-(k+1)*10])*10)
            X1[int((x-1800)/time_step)] = torch.tensor(PPF_10s)
            X2[int((x-1800)/time_step)] = torch.tensor(RFTN_10s)

        pre_bis = {}
        for j in range(len(X1)):
            pre_bis[f"t{j}"] = X1[j, :]
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pre_bis.items()]))

        df.to_csv(f'/home/user02/HYK/bis/database/{traindata}/{i}.csv')

    print(f"{traindata}data resave finish!", 'case_nums = ', case_nums)
    return


def dataload(case_nums=1, traindata="train", time_step=10, tw=180):
    """
    :param case_nums:一次性加载的样本数
    :return: 带有bis,生理特征,用药量信息的字典列表,列表长度为 case_nums
    case_id:样本id列表
    case_information:样本的生理信息表
    case_in_information:样本在信息表中的位置，如 case3:->[0], case30:->[13]
    """
    case_information = read_csv(f'/home/user02/HYK/bis/database/new_{traindata}_clean.csv')
    case_id = file_name(f'/home/user02/HYK/bis/database/new_{traindata}_clean')
    for i in range(len(case_id)):
        case_id[i] = case_id[i].split('.')[0]  # 字符串转数字

    case_id = list(map(int, case_id))
    case_id.sort()
    print("file_name:", case_id)
    case_in_information = information_deal(case_id, traindata)

    data_seq = [0]*case_nums
    data_label = [0]*case_nums

    for i in tqdm(range(case_nums)):
        df = read_csv(f'/home/user02/HYK/bis/database/new_{traindata}_clean/{case_id[i]}.csv')
        x_len = int(len(df.BIS) / time_step)
        # body信息读取
        age = case_information.age[case_in_information[i]]
        sex = case_information.sex[case_in_information[i]]
        height = case_information.height[case_in_information[i]]
        weight = case_information.weight[case_in_information[i]]
        body = torch.tensor([age, sex, height, weight]).float().reshape(1, 1, 4).repeat(x_len, tw, 1)


        # 清除异常值
        modify_RFTN = df.RFTN20_VOL.values
        modify_PPF = df.PPF20_VOL.values
        diff_RFTN = np.diff(modify_RFTN)
        diff_PPF = np.diff(modify_PPF)
        for j in range(len(diff_RFTN)):
            if diff_RFTN[j] < 0:
                temp = (modify_RFTN[j] + modify_RFTN[j + 2]) / 2
                df.loc[j + 1, "RFTN20_VOL"] = temp
            if diff_PPF[j] < 0:
                temp = (modify_PPF[j] + modify_PPF[j + 2]) / 2
                df.loc[j + 1, "PPF20_VOL"] = temp

        # 为0时刻补上-1800s的零数据
        PPF = list(np.zeros(tw*10))
        PPF.extend(df.PPF20_VOL.values)
        RFTN = list(np.zeros(tw*10))
        RFTN.extend(df.RFTN20_VOL.values)
        # 特征制作
        X1 = torch.zeros((x_len, tw))
        X2 = torch.zeros((x_len, tw))
        X3 = torch.zeros((x_len, tw))
        for x in range(tw*10, len(PPF)-10, time_step):
            # 从补完数据1800s（实际0s）时刻开始取数据段
            PPF_10s, RFTN_10s = [], []
            for k in range(179, -1, -1):
                # 第k个10s片段, 共180个
                PPF_10s.append((PPF[x-k*10]-PPF[x-(k+1)*10])*0.1)
                RFTN_10s.append((RFTN[x-k*10]-RFTN[x-(k+1)*10])*0.1)

            X1[int((x-tw*10)/time_step)] = torch.tensor(PPF_10s)
            X2[int((x-tw*10)/time_step)] = torch.tensor(RFTN_10s)

        bis = torch.tensor(df.BIS.values)
        for k in range(x_len):
            if k*time_step < tw:
                # print(torch.ones(180-k*time_step).shape, bis[:k*time_step].shape, "\r\n")
                X3[k, :] = torch.cat((torch.ones(tw-k*time_step)*92, bis[:k*time_step]), dim=0)
            else:
                X3[k, :] = bis[k*time_step-tw:k*time_step]

        seq = torch.zeros((x_len, tw, 3)).float()
        seq[:, :, 0] = X1
        seq[:, :, 1] = X2
        seq[:, :, 2] = X3

        # 归一化
        mean = torch.mean(seq, dim=1).reshape((seq.shape[0], 1, seq.shape[2])).repeat(1, tw, 1)
        std = torch.std(seq, dim=1).reshape((seq.shape[0], 1, seq.shape[2])).repeat(1, tw, 1)+1e-3
        seq = normalizition(x=seq, mu=mean, sigma=std)

        out = torch.cat((seq[:, :, :2], body), dim=2)
        out = torch.cat((out, seq[:, :, 2].reshape(seq.shape[0], tw, 1)), dim=2)

        data_seq[i] = out.float()
        label = np.zeros(x_len)
        for j in range(0, x_len, 1):
            label[int(j)] = df.BIS.values[j*time_step]

        data_label[i] = torch.tensor(label).float()

    print(f"{traindata}data load finish!", 'case_nums = ', case_nums)
    return data_seq, data_label


def information_deal(people_list, data="train"):
    """
    :param people_list: 样本的id列表，如[3, 30, 67 ...]
    :return: 样本在information表中的位置
    """
    case_information = list(read_csv(f'/home/user02/HYK/bis/database/new_{data}_clean.csv').caseid)
    case_location = list(np.zeros(len(people_list)))
    for i in range(len(people_list)):
        case_location[i] = case_information.index(people_list[i])
    print(case_location, people_list)
    print(case_location[48], people_list[48])
    return case_location  # clear3，30，36......等csv信息在information文件中的位置


def train_data_loader(tw=180, batch=1, batch_size=1, data="train", time_step=10):
    train_seq, train_label = dataload(case_nums=batch, traindata=data, time_step=time_step, tw=tw)
    A = train_seq[0]
    B = train_label[0]
    for i in range(1, batch):
        A = torch.cat((A, train_seq[i]), 0)
        B = torch.cat((B, train_label[i]), 0)
    train_data = dat.TensorDataset(A, B)
    train_loader = dat.DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=True)
    return train_loader


def test_data_loader(tw=180, batch=1, batch_size=1, data="test", timestep=1):
    test_seq, test_label = dataload(case_nums=batch, traindata=data, time_step=timestep, tw=tw)
    test_data = list(np.zeros(batch))
    test_loader = list(np.zeros(batch))
    for i in range(batch):
        test_data[i] = dat.TensorDataset(test_seq[i], test_label[i])
        test_loader[i] = dat.DataLoader(dataset=test_data[i],
                                        batch_size=batch_size,
                                        drop_last=True,
                                        pin_memory=True,
                                        num_workers=8)
    return test_loader, test_label


def time_devide(case_nums=1, traindata="test"):
    """
    :param traindata: 测试集或验证集
    :param case_nums:加载的样本数
    :return: istart:开始注射时间 istop: 停止注射时间
    """
    case_id = file_name(f'/home/user02/HYK/bis/database/new_{traindata}_clean')

    for i in range(len(case_id)):
        case_id[i] = case_id[i].split('.')[0]  # 字符串转数字

    case_id = list(map(int, case_id))
    case_id.sort()
    print("file_name:", case_id)
    infusion_start, infusion_stop = [0] * case_nums, [0] * case_nums
    for i in tqdm(range(case_nums)):
        df = read_csv(f'/home/user02/HYK/bis/database/new_{traindata}_clean/{case_id[i]}.csv')

        x_len = int(len(df.BIS))
        ppf = df.PPF20_VOL.values
        start_flag = True
        stop_flag = True
        for j in range(x_len):
            if ppf[j] > 0 and start_flag:
                infusion_start[i] = j
                start_flag = False
            if ppf[-j-1] != ppf[-j-2] and stop_flag:
                infusion_stop[i] = x_len-j+1
                stop_flag = False
            if not start_flag and not stop_flag:
                break

    print(f"{traindata}data load finish!", 'case_nums = ', case_nums)
    return infusion_start, infusion_stop





if __name__ == "__main__":
    data_resave(case_nums=1, traindata="test", time_step=1)
    # with torch.cuda.device(0):
    #     test_label = test_data_loader(tw=180, batch=300, batch_size=1, timestep=10)
    #     for i in range(300):
    #         plt.figure(i)
    #         plt.plot(test_label[i])
    #         plt.savefig(f'/home/user02/HYK/bis/output_picture/case{i}.png')







