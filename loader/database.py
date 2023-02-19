import torch
import numpy as np
from pandas import read_csv
import os
from tqdm import tqdm
import torch.utils.data as dat
import matplotlib.pyplot as plt


class Dataloader:
    def __init__(self, database_wdir, nums, time_step, tw):
        self.database_wdir = database_wdir
        self.nums = nums
        self.time_step = time_step
        self.tw = tw

    def label_stat(self, case_nums=1, traindata="train"):
        """
        :param case_nums:一次性加载的样本数
        :return: 带有bis,生理特征,用药量信息的字典列表,列表长度为 case_nums
        case_id:样本id列表
        case_information:样本的生理信息表
        case_in_information:样本在信息表中的位置，如 case3:->[0], case30:->[13]
        """
        case_information = read_csv(f'/HDD_data/HYK/bis/database/new_{traindata}_clean.csv')
        case_id = self.file_name(data=traindata)
        for i in range(len(case_id)):
            case_id[i] = case_id[i].split('.')[0]  # 字符串转数字

        case_id = list(map(int, case_id))
        case_id.sort()

        data_label = []

        for i in tqdm(range(case_nums)):
            df = read_csv(f'{self.database_wdir}/{traindata}/{case_id[i]}.csv')
            x_len = int(len(df.BIS) / self.time_step)

            label = np.zeros(x_len)
            for j in range(0, x_len, 1):
                label[int(j)] = df.BIS.values[j * self.time_step]

            data_label.extend(label)

        data_label.sort()
        for i in range(len(data_label)):
            data_label[i] = int(data_label[i])
        j = 0
        label_num = list(np.zeros(100))
        for i in range(100):
            while data_label[j] == i:
                label_num[i] += 1
                j += 1
                if j == len(data_label)-10:
                    break

        import matplotlib.pyplot as plt

        plt.grid(True)
        plt.autoscale(axis='x', tight=True)
        plt.bar(list(range(100)), label_num)
        plt.xlabel("bis index")
        plt.ylabel("label nums")
        plt.show()

        return data_label, label_num

    def dataload(self, case_nums=1, traindata="train"):
        """
        :param case_nums:一次性加载的样本数
        :return: 带有bis,生理特征,用药量信息的字典列表,列表长度为 case_nums
        case_id:样本id列表
        case_information:样本的生理信息表
        case_in_information:样本在信息表中的位置，如 case3:->[0], case30:->[13]
        x1:ppf_vol
        x2:rftn_vol
        x3:pkpd_bis
        X4:bis_history
        X5:RFTN_CP
        x6-x9:body information(age, sex, height, weight)
        """
        # case_information = read_csv(f'/HDD_data/HYK/bis/database/new_{traindata}_clean.csv')
        case_information = read_csv(f'/HDD_data/HYK/bis/database/ni_dataset/info.csv')
        case_id = self.file_name(data=traindata)
        # for i in range(len(case_id)):
        #     case_id[i] = case_id[i].split('.')[0]  # 字符串转数字

        # case_id = list(map(int, case_id))


        case_id.sort()
        print("file_name:", case_id)
        case_in_information = self.information_deal(case_id, traindata)

        data_seq = [0] * case_nums
        data_label = [0] * case_nums

        for i in tqdm(range(case_nums)):
            df = read_csv(f'{self.database_wdir}/{traindata}/{case_id[i]}.csv')
            x_len = int(len(df.BIS) / self.time_step)
            # body信息读取
            age = case_information.age[case_in_information[i]]
            sex = case_information.sex[case_in_information[i]]
            height = case_information.height[case_in_information[i]]
            weight = case_information.weight[case_in_information[i]]
            body = torch.tensor([age, sex, height, weight]).float().reshape(1, 1, 4).repeat(x_len, self.tw, 1)

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
            PPF = list(np.zeros(self.tw * 10))
            PPF.extend(df.PPF20_VOL.values)
            RFTN = list(np.zeros(self.tw * 10))
            RFTN.extend(df.RFTN20_VOL.values)

            ppf_cp = list(np.zeros(self.tw * 10))
            ppf_cp.extend(df.PPF_CP.values)
            rftn_cp = list(np.zeros(self.tw * 10))
            rftn_cp.extend(df.RFTN20_CP.values)

            ppf_ce = df.PPF_CE.values
            rftn_ce = df.RFTN20_CE.values

            pkpd_bis = self.pkpd(ppf_ce, rftn_ce)
            PKPD_bis = list(np.ones(self.tw * 10)*98)
            PKPD_bis.extend(pkpd_bis)

            history_bis = df.BIS.values
            bis = list(np.zeros(self.tw * 10))
            bis.extend(history_bis)

            # 特征制作
            X1 = torch.zeros((x_len, self.tw))
            X2 = torch.zeros((x_len, self.tw))
            X3 = torch.zeros((x_len, self.tw))
            X4 = torch.zeros((x_len, self.tw))
            X5 = torch.zeros((x_len, self.tw))

            for x in range(self.tw*10, len(PPF) - self.time_step, self.time_step):
                # 从补完数据1800s（实际0s）时刻开始取数据段
                PPF_10s, RFTN_10s, BIS_10s, history_10s, RFTN_CP_10s = [], [], [], [], []
                for k in range(self.tw-1, -1, -1):
                    # 第k个10s片段, 共180个
                    PPF_10s.append((PPF[x - k * 10] - PPF[x - (k + 1) * 10]) * 0.1)
                    RFTN_10s.append((RFTN[x - k * 10] - RFTN[x - (k + 1) * 10]) * 0.1)
                    BIS_10s.append((PKPD_bis[x - k * 10]))
                    history_10s.append((bis[x - k * 10]))
                    RFTN_CP_10s.append((rftn_cp[x - k * 10]))


                X1[int((x - self.tw * 10) / self.time_step)] = torch.tensor(PPF_10s)
                X2[int((x - self.tw * 10) / self.time_step)] = torch.tensor(RFTN_10s)
                X3[int((x - self.tw * 10) / self.time_step)] = torch.tensor(BIS_10s)
                X4[int((x - self.tw * 10) / self.time_step)] = torch.tensor(history_10s)
                X5[int((x - self.tw * 10) / self.time_step)] = torch.tensor(RFTN_CP_10s)

            # bis = torch.tensor(df.BIS.values)
            # for k in range(x_len):
            #     if k * self.time_step < self.tw:
            #         X4[k, :] = torch.cat((torch.ones(self.tw - k * self.time_step) * 98, bis[:k * self.time_step]), dim=0)
            #         # X3[k, :] = torch.cat((torch.zeros(self.tw - k * self.time_step), pkpd_bis[:k * self.time_step]), dim=0)
            #         # X5[k, :] = torch.cat((torch.zeros(180 - k * self.time_step), rftn_ce[:k * self.time_step]), dim=0)
            #
            #     else:
            #         X4[k, :] = bis[k * self.time_step - self.tw:k * self.time_step]
            #         # X3[k, :] = pkpd_bis[k * self.time_step - self.tw:k * self.time_step]
            #         # X5[k, :] = rftn_ce[k * self.time_step - 180:k * self.time_step]

            seq = torch.zeros((x_len, self.tw, 5)).float()
            seq[:, :, 0] = X1  # ppf vol
            seq[:, :, 1] = X2  # rftn vol
            # 归一化
            mean = torch.mean(seq, dim=1).reshape((seq.shape[0], 1, seq.shape[2])).repeat(1, self.tw, 1)
            std = torch.std(seq, dim=1).reshape((seq.shape[0], 1, seq.shape[2])).repeat(1, self.tw, 1) + 1e-3
            seq = self.normalizition(x=seq, mu=mean, sigma=std)

            seq[:, :, 2] = X3  # pk-pd bis
            seq[:, :, 3] = X4  # ppf cp
            seq[:, :, 4] = X5  # rftn cp

            out = torch.cat((seq, body), dim=2)
            # out = torch.cat((out, seq[:, :, 2].reshape(seq.shape[0], 180, 1)), dim=2)

            data_seq[i] = out.float()
            label = np.zeros(x_len)
            for j in range(0, x_len, 1):
                label[int(j)] = df.BIS.values[j * self.time_step]

            data_label[i] = torch.tensor(label).float()

        print(f"{traindata}data load finish!", 'case_nums = ', case_nums)
        return data_seq, data_label

    def train_data_loader(self, batch=1, batch_size=1, data="train", shuffle=True):
        train_seq, train_label = self.dataload(case_nums=batch, traindata=data)
        A = train_seq[0]
        B = train_label[0]
        for i in range(1, batch):
            A = torch.cat((A, train_seq[i]), 0)
            B = torch.cat((B, train_label[i]), 0)

        torch.save(A, f"/HDD_data/HYK/bis/database/validdata.pt")
        torch.save(B, f"/HDD_data/HYK/bis/database/validlabel.pt")

        # np.save(A.data.numpy(), "/HDD_data/HYK/bis/database/traindata.npy")
        # np.save(B.data.numpy(), "/HDD_data/HYK/bis/database/trainlabel.npy")

        # train_data = dat.TensorDataset(A, B)
        # train_loader = dat.DataLoader(dataset=train_data,
        #                               batch_size=batch_size,
        #                               drop_last=True,
        #                               num_workers=4,
        #                               pin_memory=True,
        #                               shuffle=shuffle)
        # return train_loader
        return 0

    def test_data_loader(self, batch=1, batch_size=1, data="test"):
        test_seq, test_label = self.dataload(case_nums=batch, traindata=data)
        test_data = list(np.zeros(batch))
        test_loader = list(np.zeros(batch))
        for i in range(batch):
            torch.save(test_seq[i], f"/HDD_data/HYK/bis/database/test_box/testndata{i}.pt")
            torch.save(test_label[i], f"/HDD_data/HYK/bis/database/test_box/testlabel{i}.pt")
            # test_data[i] = dat.TensorDataset(test_seq[i], test_label[i])
            # test_loader[i] = dat.DataLoader(dataset=test_data[i],
            #                                 batch_size=batch_size,
            #                                 drop_last=True,
            #                                 pin_memory=True,
            #                                 num_workers=8)
        return test_loader, test_label

    def information_deal(self, people_list, data="train"):
        """
        :param people_list: 样本的id列表，如[3, 30, 67 ...]
        :return: 样本在information表中的位置
        """
        case_information = list(read_csv(f'/HDD_data/HYK/bis/database/new_{data}_clean.csv').caseid)
        case_location = list(np.zeros(len(people_list)))
        for i in range(len(people_list)):
            case_location[i] = case_information.index(people_list[i])
        return case_location  # clear3，30，36......等csv信息在information文件中的位置

    def time_devide(self, case_nums=1, traindata="test"):
        """
        :param traindata: 测试集或验证集
        :param case_nums:加载的样本数
        :return: istart:开始注射时间 istop: 停止注射时间
        """
        case_id = self.file_name(traindata)

        for i in range(len(case_id)):
            case_id[i] = case_id[i].split('.')[0]  # 字符串转数字

        case_id = list(map(int, case_id))
        case_id.sort()
        print("file_name:", case_id)
        infusion_start, infusion_stop = [0] * case_nums, [0] * case_nums
        for i in tqdm(range(case_nums)):
            df = read_csv(f'/HDD_data/HYK/bis/database/{traindata}/{case_id[i]}.csv')

            x_len = int(len(df.BIS))
            ppf = df.PPF20_VOL.values
            start_flag = True
            stop_flag = True
            for j in range(x_len):
                if ppf[j] > 0 and start_flag:
                    infusion_start[i] = j
                    start_flag = False
                if ppf[-j - 1] != ppf[-j - 2] and stop_flag:
                    infusion_stop[i] = x_len - j + 1
                    stop_flag = False
                if not start_flag and not stop_flag:
                    break

        print(f"{traindata}data load finish!", 'case_nums = ', case_nums)
        return infusion_start, infusion_stop

    def file_name(self, data):
        for root, dirs, files in os.walk(f'{self.database_wdir}/{data}'):
            return files  # 当前路径下所有非目录子文件,列表

    @staticmethod
    def pkpd(Ec1, Ec2):
        ppf_ec50 = 4.47
        rftn_ec50 = 19.3
        gamma = 1.43
        p_gamma = (Ec1/ppf_ec50 + Ec2/rftn_ec50)**gamma
        bis = 98. - 98. * p_gamma / (1 + p_gamma)
        return bis

    @staticmethod
    def normalizition(x, mu, sigma):
        # mu 均值 sigms 标准差
        x = (x - mu) / sigma
        return x

    def ceload(self, case_nums=1, traindata="test"):
        """
        :param case_nums:一次性加载的样本数
        :return: 带有bis,生理特征,用药量信息的字典列表,列表长度为 case_nums
        case_id:样本id列表
        case_information:样本的生理信息表
        case_in_information:样本在信息表中的位置，如 case3:->[0], case30:->[13]
        x1:ppf_vol
        x2:rftn_vol
        x3:pkpd_bis
        X4:RFTN_CP
        x5-x8:body information(age, sex, height, weight)
        """
        case_id = self.file_name(data=traindata)
        for i in range(len(case_id)):
            case_id[i] = case_id[i].split('.')[0]  # 字符串转数字
        case_id = list(map(int, case_id))
        case_id.sort()

        PKPD_bis = []
        for i in tqdm(range(case_nums)):
            df = read_csv(f'{self.database_wdir}/{traindata}/{case_id[i]}.csv')
            x_len = int(len(df.BIS) / self.time_step)

            ppf_ce = df.PPF_CE.values
            rftn_ce = df.RFTN20_CE.values

            pkpd_bis = self.pkpd(ppf_ce, rftn_ce)
            PKPD_bis.append(pkpd_bis)

        return PKPD_bis

    def data_save(self, case_nums=1, traindata="test"):
        case_information = read_csv(f'/HDD_data/HYK/bis/database/before_bodyinformation.csv')
        case_id = self.file_name(data=traindata)
        for i in range(len(case_id)):
            case_id[i] = case_id[i].split('.')[0]  # 字符串转数字
        case_id = list(map(int, case_id))
        case_in_information = self.information_deal(case_id, traindata)
        case_id.sort()
        X = list(range(case_nums))
        for i in tqdm(range(case_nums)):
            df = read_csv(f'{self.database_wdir}/{traindata}/{case_id[i]}.csv')
            age = case_information.age[case_in_information[i]]
            sex = case_information.sex[case_in_information[i]]
            height = case_information.height[case_in_information[i]]
            weight = case_information.weight[case_in_information[i]]

            X[i] = [
                np.median(df.BIS.values),
                df.PPF20_VOL.values[-1]*20/1000,
                df.RFTN20_VOL.values[-1]*20/1000,
                np.median(df.PPF_CE.values),
                np.median(df.RFTN20_CE.values),
                age, sex, height, weight]
        file = {}
        X = np.asarray(X)
        name = ["bis", "ppf_dose", "rftn_dose", "ppf_ce", "rftn_ce", "age", "sex", "height", "weight"]
        for j in range(len(name)):
            file[f"{name[j]}"] = X[:, j]

        import pandas as pd
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in file.items()]))

        df.to_csv(f'/HDD_data/HYK/bis/database/{traindata}.csv')
        return X

    def pt_load(self, dataset, batch_size):
        import gc
        if dataset == "train":
            A = torch.load("/HDD_data/HYK/bis/database/traindata.pt")
            B = torch.load("/HDD_data/HYK/bis/database/trainlabel.pt")
            train_data = dat.TensorDataset(A, B)
            train_loader = dat.DataLoader(dataset=train_data,
                                          batch_size=batch_size,
                                          drop_last=True,
                                          num_workers=4,
                                          pin_memory=True,
                                          shuffle=True)
            print("Training set loading completed")
            del A, B
            gc.collect()
            return train_loader
        elif dataset == "test":
            test_loader = list(np.zeros(76))
            B = list(np.zeros(76))
            for i in tqdm(range(76)):
                A = torch.load(f"/HDD_data/HYK/bis/database/test_box/testndata{i}.pt")
                B[i] = torch.load(f"/HDD_data/HYK/bis/database/test_box/testlabel{i}.pt")
                C = dat.TensorDataset(A, B[i])
                test_loader[i] = dat.DataLoader(
                    dataset=C,
                    batch_size=batch_size,
                    drop_last=True, )

                del A, C
                gc.collect()

            print("Testing set loading completed")
            return test_loader, B
        elif dataset == "valid":
            A = torch.load("/HDD_data/HYK/bis/database/validdata.pt")
            B = torch.load("/HDD_data/HYK/bis/database/validlabel.pt")
            train_data = dat.TensorDataset(A, B)
            valid_loader = dat.DataLoader(dataset=train_data,
                                          batch_size=batch_size,
                                          drop_last=True,
                                          num_workers=4,
                                          pin_memory=True,
                                          shuffle=True)
            print("Validation set loading completed")
            return valid_loader

    def load_all(self, vb, trb, teb):
        vaild_loader = self.pt_load(
            dataset="valid",
            batch_size=vb
        )

        train_loader = self.pt_load(
            dataset="train",
            batch_size=trb,
        )

        test_loader, test_label = self.pt_load(
            dataset="test",
            batch_size=teb,
        )
        return vaild_loader, train_loader, test_loader, test_label


def data_distribution_bar(data, label_error=None):
    """
    :param data: data will be plot in bar
    :return:
    """
    fig = plt.figure(figsize=(24, 16))

    da = plt.Rectangle((24, 0), 38, 50, color="cornsilk")
    ga = plt.Rectangle((32, 0), 14.5, 50, color="paleturquoise")
    # s = plt.Rectangle((60, 0), 30, 50, color="cornsilk")
    w = plt.Rectangle((0, 0), 100, 50, color="pink")

    if not label_error:
        ax = fig.add_subplot(111)
        plt.xlabel('BIS', fontsize=30)
    else:
        ax_error = fig.add_subplot(212)
        ax_error.add_patch(w)
        ax_error.add_patch(da)
        ax_error.add_patch(ga)

        ax_error.bar(list(range(100)), label_error[1], color='forestgreen')
        ax_error.bar(list(range(100)), label_error[0], color='salmon')
        ax_error.legend(['Few-shot region', 'Medium-shot region', 'Many-shot region', 'Baseline', 'Ours'],
                        fontsize=25, loc=1)
        plt.xlim(0, 100)
        plt.ylim(0, 50)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlabel('BIS', fontsize=30)
        plt.ylabel('Test error', fontsize=30)

        ax = fig.add_subplot(211)

    plt.xlim(0, 100)
    plt.ylim(0, 6.5)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylabel('Percentage(%)', fontsize=30)

    da = plt.Rectangle((24, 0), 38, 50, color="cornsilk")
    ga = plt.Rectangle((32.5, 0), 14, 50, color="paleturquoise")
    # s = plt.Rectangle((60, 0), 30, 50, color="cornsilk")
    w = plt.Rectangle((0, 0), 100, 50, color="pink")

    ax.add_patch(w)
    ax.add_patch(da)
    ax.add_patch(ga)

    # for i in range(3):
    #     ax.text(15+i*30, 6.65, '%.2f%%' % sum(data[i*30:i*30+30]),
    #             fontsize=30, ha='center', va='bottom')
    ax.text(12, 6.65, '%.2f%%' % sum(data[:24]),
            fontsize=30, ha='center', va='bottom')
    ax.text(28, 6.65, '%.2f%%' % sum(data[24:32]),
            fontsize=30, ha='center', va='bottom')
    ax.text(39.75, 6.65, '%.2f%%' % sum(data[32:46]),
            fontsize=30, ha='center', va='bottom')
    ax.text(54.25, 6.65, '%.2f%%' % sum(data[46:62]),
            fontsize=30, ha='center', va='bottom')
    ax.text(81, 6.65, '%.2f%%' % sum(data[62:]),
            fontsize=30, ha='center', va='bottom')

    ax.bar(list(range(100)), data, color='darkslateblue')
    ax.legend(['Few-shot region', 'Medium-shot region', 'Many-shot region', 'Label Percentage'],
              fontsize=25, loc=1)

    plt.savefig('/HDD_data/HYK/bis/output/test error.jpg')
    plt.show()


def error_down(e):
    e1 = np.asarray(e[0][:98])
    e2 = np.asarray(e[1][:98])
    return (e1-e2)/e2


if __name__ == "__main__":
    data = 1
    # data_distribution_bar(data, label_error=None)






