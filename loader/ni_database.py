import torch
import numpy as np
from pandas import read_csv
import os
from tqdm import tqdm
import torch.utils.data as dat


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
        case_id = self.file_name(data=traindata)

        case_id.sort()
        cid = list(range(case_nums))

        data_label = []

        for i in tqdm(range(case_nums)):
            df = read_csv(f'{self.database_wdir}/{traindata}/{case_id[i]}')
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
        case_information = read_csv(f'{self.database_wdir}/info.csv')
        case_id = self.file_name(data=traindata)

        case_id.sort()
        print("file_name:", case_id)
        cid = list(range(case_nums))
        case_in_information = self.information_deal(cid, traindata)

        data_seq = [0] * case_nums
        data_label = [0] * case_nums

        for i in tqdm(range(case_nums)):
            df = read_csv(f'{self.database_wdir}/{traindata}/{case_id[i]}')
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

            # 插值
            # ppf = list(range(len(df.PPF20_VOL.values)))
            # source_data_x = list(range(0, len(df.PPF20_VOL.values), 5))
            # source_data_y = df.PPF20_VOL.values
            # ppf = np.interp(ppf, source_data_x, source_data_y)
            #
            # rftn = list(range(len(df.RFTN20_VOL.values)))
            # source_data_x = list(range(0, len(df.RFTN20_VOL.values), 5))
            # source_data_y = df.RFTN20_VOL.values
            # rftn = np.interp(rftn, source_data_x, source_data_y)
            #
            # ppf_ce = list(range(len(df.PPF20_CE.values)))
            # ppf_ce = np.interp(ppf_ce, source_data_x, source_data_y)
            # rftn_ce = df.RFTN20_CE.value

            # 为0时刻补上-1800s的零数据
            PPF = list(np.zeros(self.tw * 10))
            PPF.extend(df.PPF20_VOL.values)
            RFTN = list(np.zeros(self.tw * 10))
            RFTN.extend(df.RFTN20_VOL.values)

            ppf_ce = df.PPF20_CE.values
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


                X1[int((x - self.tw * 10) / self.time_step)] = torch.tensor(PPF_10s)
                X2[int((x - self.tw * 10) / self.time_step)] = torch.tensor(RFTN_10s)
                X3[int((x - self.tw * 10) / self.time_step)] = torch.tensor(BIS_10s)
                X4[int((x - self.tw * 10) / self.time_step)] = torch.tensor(history_10s)


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

        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess
        train_label_new = list(range(batch))
        for i in tqdm(range(batch)):
            axis = list(range(len(train_label[i])))
            train_label_new[i] = lowess(train_label[i], axis, frac=0.03)[:, 1]


        A = train_seq[0]
        B = torch.tensor(train_label_new[0]).float()
        for i in range(1, batch):
            A = torch.cat((A, train_seq[i]), 0)
            B = torch.cat((B, torch.tensor(train_label_new[i]).float()), 0)

        torch.save(A, f"{self.database_wdir}/{data}data.pt")
        torch.save(B, f"{self.database_wdir}/{data}label.pt")
        return 0

    def test_data_loader(self, batch=1, batch_size=1, data="test", box=0):
        test_seq, test_label = self.dataload(case_nums=batch, traindata=data)
        test_data = list(np.zeros(batch))
        test_loader = list(np.zeros(batch))
        for i in range(batch):
            if box == 0:
                torch.save(test_seq[i], f"{self.database_wdir}/box/testndata{i}.pt")
                torch.save(test_label[i], f"{self.database_wdir}/box/testlabel{i}.pt")
            else:
                torch.save(test_seq[i], f"{self.database_wdir}/ours_box/testndata{i}.pt")
                torch.save(test_label[i], f"{self.database_wdir}/ours_box/testlabel{i}.pt")

        return test_loader, test_label

    def information_deal(self, people_list, data="train"):
        """
        :param people_list: 样本的id列表，如[3, 30, 67 ...]
        :return: 样本在information表中的位置
        """
        case_information = list(read_csv(f'{self.database_wdir}/info.csv').caseid)
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
        case_id.sort()
        print("file_name:", case_id)
        infusion_start, infusion_stop = [0] * case_nums, [0] * case_nums
        for i in tqdm(range(case_nums)):
            df = read_csv(f'{self.database_wdir}/{traindata}/{case_id[i]}')

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
        case_id = list(case_id)
        case_id.sort()

        PKPD_bis = []
        for i in tqdm(range(case_nums)):
            df = read_csv(f'{self.database_wdir}/{traindata}/{case_id[i]}.csv')
            x_len = int(len(df.BIS) / self.time_step)

            ppf_ce = df.PPF20_CE.values
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

    def pt_load(self, dataset, batch_size, box=0):
        import gc
        import statsmodels.api as sm
        if box == 0:
            box = 'box'
        else:
            box = 'ours_box'
        if dataset == "train":
            A = torch.load(f"{self.database_wdir}/traindata.pt")
            B = torch.load(f"{self.database_wdir}/trainlabel.pt")




            train_data = dat.TensorDataset(A, B)
            train_loader = dat.DataLoader(dataset=train_data,
                                          batch_size=batch_size,
                                          drop_last=True,
                                          num_workers=8,
                                          pin_memory=True,
                                          shuffle=True)
            print("Training set loading completed")
            del A, B
            gc.collect()
            return train_loader
        elif dataset == "test":
            test_list = os.listdir(f'{self.database_wdir}/{box}')
            file_nums = int(len(test_list)*0.5)
            test_loader = list(np.zeros(file_nums))
            B = list(np.zeros(file_nums))
            for i in tqdm(range(file_nums)):
                A = torch.load(f"{self.database_wdir}/{box}/testndata{i}.pt")
                B[i] = torch.load(f"{self.database_wdir}/{box}/testlabel{i}.pt")
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
            A = torch.load(f"{self.database_wdir}/validdata.pt")
            B = torch.load(f"{self.database_wdir}/validlabel.pt")
            train_data = dat.TensorDataset(A, B)
            valid_loader = dat.DataLoader(dataset=train_data,
                                          batch_size=batch_size,
                                          drop_last=True,
                                          num_workers=4,
                                          pin_memory=True,
                                          shuffle=True)
            print("Validation set loading completed")
            return valid_loader

    def load_all(self, vb, trb, teb, test_box=0):
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
            box=test_box
        )
        return vaild_loader, train_loader, test_loader, test_label


if __name__ == "__main__":
    # a = Dataloader(database_wdir="/HDD_data/HYK/bis/database", time_step=1, nums=1, tw=180)
    # A = a.data_save(87, "valid")
    d_box = Dataloader(
        database_wdir='/data/HYK/DATASET/bis/database',
        time_step=1,
        nums=15,
        tw=180
    )
    d_box.time_step = 1
    # train_loader = d_box.train_data_loader(
    #     data="train",
    #     batch=24
    # )
    #
    # valid_loader = d_box.train_data_loader(
    #     data="valid",
    #     batch=20
    # )

    test_loader, test_label = d_box.test_data_loader(
        data="test",
        batch=20,
        box=0
    )


    # test_loader, test_label = d_box.pt_load(
    #     dataset="test",
    #     batch_size=256,
    # )




    # X = d_box.data_save(180, "train")




