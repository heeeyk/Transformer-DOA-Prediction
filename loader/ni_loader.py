# import database
# import torch
# import numpy as np
# from pandas import read_csv
# import os
# from tqdm import tqdm
# import torch.utils.data as dat
# import xlrd
# import csv
#
#
# class NiLoader:
#     """
#     step 1: 整理好数据，一定要先让三个文件夹的记录对应起来
#     step 2: run body_information
#     step 3: run clean()， 我把数据加载写好了，你把之前清理数据那部分放过来就行
#     """
#     def __init__(self, root_dir="/HDD_data/HYK/bis/database/ni_dataset", nums=48, time_step=1, tw=180):
#         self.root_dir = root_dir
#         self.nums = nums
#         self.time_step = time_step
#         self.tw = tw
#
#     def file_name(self, wir):
#         # wir: record/ppf, record/rmft, or label
#         for root, dirs, files in os.walk(f"{self.root_dir}/{wir}"):
#             files.sort()
#             return files  # 当前路径下所有非目录子文件,列表
#
#     def body_information(self, files):
#         """
#         在root_wir下生成生理信息表
#         :param files:文件名列表
#         :return: 生理信息
#         """
#         a = []
#         for n in range(self.nums):
#             b = [n]
#             file_dir = f"{self.root_dir}/record/ppf/{files[n]}"
#             data = xlrd.open_workbook(file_dir)
#             table = data.sheets()[0]
#             info = table.row_values(1)
#             for i in range(1, 8, 2):
#                 b.append(info[i])
#
#             if b[2] == '女':
#                 b[2] = 0
#             else:
#                 b[2] = 1
#             b[3] = int(b[3].split(' cm')[0])
#             b[4] = int(b[4].split(' Kg')[0])
#             a.append(b)
#
#
#         with open(f"{self.root_dir}/info.csv", 'w', newline='') as t:
#             writer = csv.writer(t)
#             writer.writerow(["caseid", "age", "sex", "height", "weight"])
#             writer.writerows(a)
#         return a
#
#     def load(self, files, m="ppf"):
#         """
#         :param files: list, file path list
#         :param m: str, "ppf" or "rftn"
#         :return: 药物记录 和 时间记录 和 效应室浓度
#         """
#         a, b, c = [], [], []  # a:药的记录 b:时间记录
#         for n in tqdm(range(self.nums)):
#             file_dir = f"{self.root_dir}/record/{m}/{files[n]}"
#             data = xlrd.open_workbook(file_dir)
#             table = data.sheets()[0]
#             vol = table.col_values(6)
#             vol = vol[4:-1]
#             times = table.col_values(8)
#             times = times[4:-1]
#             ce = table.col_values(4)
#             ce = ce[4:-1]
#
#             a.append(vol)
#             b.append(times)
#             c.append(ce)
#
#         return a, b, c
#
#     def clean(self):
#         files_ppf = self.file_name("record/ppf")
#         files_rftn = self.file_name("record/rftn")
#         ppf, ppf_ce, tp = self.load(files_ppf, "ppf")
#         rftn, rftn_ce, tr = self.load(files_rftn, "rftn")
#
#
#
#
#         for i in range(48):
#             # step 1：裁剪
#             ppf[i], rftn[i] = self.crop(ppf[i], rftn[i], tp[i], tr[i])
#
#             # step 2：插值, 交给你了
#
#             # step 3：平滑， 交给你了
#
#         return ppf, rftn
#
#     @staticmethod
#     def crop(x1, x2, x3, x4, t1, t2):
#         """
#         将两种药进行时间对齐, 对不齐的丢掉
#         :param x1: ppf record
#         :param x2: rftn record
#         :param x3: ppf ce
#         :param x4: rftn ce
#         :param t1: ppf infusion times record
#         :param t2: rftn infusion times record
#         :return:
#         """
#         t, i, j = 0, 0, 0
#         if len(t1) < len(t2):
#             t = len(t1)
#         else:
#             t = len(t2)
#         while t1[i] - t2[j] >= 0.0005 or t1[i] - t2[j] <= -0.0005:
#             if t1[i] < t2[j]:
#                 i += 1
#             else:
#                 j += 1
#             if i == t or j == t:
#                 return 0, 0
#
#         x1 = x1[i:-1]
#         x2 = x2[j:-1]
#
#         x3 = x3[i:-1]
#         x4 = x4[j:-1]
#
#         if len(x1) < len(x2):
#             x2 = x2[:len(x1)]
#             x4 = x4[:len(x1)]
#         else:
#             x1 = x1[:len(x2)]
#             x3 = x3[:len(x1)]
#
#         return x1, x2, x3, x4
#
#     def dataload(self, case_nums=1, traindata="train"):
#         """
#         :param case_nums:一次性加载的样本数
#         :return: 带有bis,生理特征,用药量信息的字典列表,列表长度为 case_nums
#         case_id:样本id列表
#         case_information:样本的生理信息表
#         case_in_information:样本在信息表中的位置，如 case3:->[0], case30:->[13]
#         x1:ppf_vol
#         x2:rftn_vol
#         x3:pkpd_bis
#         X4:bis_history
#         X5:RFTN_CP
#         x6-x9:body information(age, sex, height, weight)
#         """
#         case_information = read_csv(f'/HDD_data/HYK/bis/database/new_{traindata}_clean.csv')
#         case_id = self.file_name(data=traindata)
#         for i in range(len(case_id)):
#             case_id[i] = case_id[i].split('.')[0]  # 字符串转数字
#
#         case_id = list(map(int, case_id))
#         case_id.sort()
#         print("file_name:", case_id)
#         case_in_information = self.information_deal(case_id, traindata)
#
#         data_seq = [0] * case_nums
#         data_label = [0] * case_nums
#
#         for i in tqdm(range(case_nums)):
#             df = read_csv(f'{self.database_wdir}/{traindata}/{case_id[i]}.csv')
#             x_len = int(len(df.BIS) / self.time_step)
#             # body信息读取
#             age = case_information.age[case_in_information[i]]
#             sex = case_information.sex[case_in_information[i]]
#             height = case_information.height[case_in_information[i]]
#             weight = case_information.weight[case_in_information[i]]
#             body = torch.tensor([age, sex, height, weight]).float().reshape(1, 1, 4).repeat(x_len, self.tw, 1)
#
#             # 清除异常值
#             modify_RFTN = df.RFTN20_VOL.values
#             modify_PPF = df.PPF20_VOL.values
#             diff_RFTN = np.diff(modify_RFTN)
#             diff_PPF = np.diff(modify_PPF)
#             for j in range(len(diff_RFTN)):
#                 if diff_RFTN[j] < 0:
#                     temp = (modify_RFTN[j] + modify_RFTN[j + 2]) / 2
#                     df.loc[j + 1, "RFTN20_VOL"] = temp
#                 if diff_PPF[j] < 0:
#                     temp = (modify_PPF[j] + modify_PPF[j + 2]) / 2
#                     df.loc[j + 1, "PPF20_VOL"] = temp
#
#             # 为0时刻补上-1800s的零数据
#             PPF = list(np.zeros(self.tw * 10))
#             PPF.extend(df.PPF20_VOL.values)
#             RFTN = list(np.zeros(self.tw * 10))
#             RFTN.extend(df.RFTN20_VOL.values)
#
#             ppf_cp = list(np.zeros(self.tw * 10))
#             ppf_cp.extend(df.PPF_CP.values)
#             rftn_cp = list(np.zeros(self.tw * 10))
#             rftn_cp.extend(df.RFTN20_CP.values)
#
#             ppf_ce = df.PPF_CE.values
#             rftn_ce = df.RFTN20_CE.values
#
#             pkpd_bis = self.pkpd(ppf_ce, rftn_ce)
#             PKPD_bis = list(np.ones(self.tw * 10)*98)
#             PKPD_bis.extend(pkpd_bis)
#
#             history_bis = df.BIS.values
#             bis = list(np.zeros(self.tw * 10))
#             bis.extend(history_bis)
#
#             # 特征制作
#             X1 = torch.zeros((x_len, self.tw))
#             X2 = torch.zeros((x_len, self.tw))
#             X3 = torch.zeros((x_len, self.tw))
#             X4 = torch.zeros((x_len, self.tw))
#             X5 = torch.zeros((x_len, self.tw))
#
#             for x in range(self.tw*10, len(PPF) - self.time_step, self.time_step):
#                 # 从补完数据1800s（实际0s）时刻开始取数据段
#                 PPF_10s, RFTN_10s, BIS_10s, history_10s, RFTN_CP_10s = [], [], [], [], []
#                 for k in range(self.tw-1, -1, -1):
#                     # 第k个10s片段, 共180个
#                     PPF_10s.append((PPF[x - k * 10] - PPF[x - (k + 1) * 10]) * 0.1)
#                     RFTN_10s.append((RFTN[x - k * 10] - RFTN[x - (k + 1) * 10]) * 0.1)
#                     BIS_10s.append((PKPD_bis[x - k * 10]))
#                     history_10s.append((bis[x - k * 10]))
#                     RFTN_CP_10s.append((rftn_cp[x - k * 10]))
#
#
#                 X1[int((x - self.tw * 10) / self.time_step)] = torch.tensor(PPF_10s)
#                 X2[int((x - self.tw * 10) / self.time_step)] = torch.tensor(RFTN_10s)
#                 X3[int((x - self.tw * 10) / self.time_step)] = torch.tensor(BIS_10s)
#                 X4[int((x - self.tw * 10) / self.time_step)] = torch.tensor(history_10s)
#                 X5[int((x - self.tw * 10) / self.time_step)] = torch.tensor(RFTN_CP_10s)
#
#             # bis = torch.tensor(df.BIS.values)
#             # for k in range(x_len):
#             #     if k * self.time_step < self.tw:
#             #         X4[k, :] = torch.cat((torch.ones(self.tw - k * self.time_step) * 98, bis[:k * self.time_step]), dim=0)
#             #         # X3[k, :] = torch.cat((torch.zeros(self.tw - k * self.time_step), pkpd_bis[:k * self.time_step]), dim=0)
#             #         # X5[k, :] = torch.cat((torch.zeros(180 - k * self.time_step), rftn_ce[:k * self.time_step]), dim=0)
#             #
#             #     else:
#             #         X4[k, :] = bis[k * self.time_step - self.tw:k * self.time_step]
#             #         # X3[k, :] = pkpd_bis[k * self.time_step - self.tw:k * self.time_step]
#             #         # X5[k, :] = rftn_ce[k * self.time_step - 180:k * self.time_step]
#
#             seq = torch.zeros((x_len, self.tw, 5)).float()
#             seq[:, :, 0] = X1  # ppf vol
#             seq[:, :, 1] = X2  # rftn vol
#             # 归一化
#             mean = torch.mean(seq, dim=1).reshape((seq.shape[0], 1, seq.shape[2])).repeat(1, self.tw, 1)
#             std = torch.std(seq, dim=1).reshape((seq.shape[0], 1, seq.shape[2])).repeat(1, self.tw, 1) + 1e-3
#             seq = self.normalizition(x=seq, mu=mean, sigma=std)
#
#             seq[:, :, 2] = X3  # pk-pd bis
#             seq[:, :, 3] = X4  # ppf cp
#             seq[:, :, 4] = X5  # rftn cp
#
#             out = torch.cat((seq, body), dim=2)
#             # out = torch.cat((out, seq[:, :, 2].reshape(seq.shape[0], 180, 1)), dim=2)
#
#             data_seq[i] = out.float()
#             label = np.zeros(x_len)
#             for j in range(0, x_len, 1):
#                 label[int(j)] = df.BIS.values[j * self.time_step]
#
#             data_label[i] = torch.tensor(label).float()
#
#         print(f"{traindata}data load finish!", 'case_nums = ', case_nums)
#         return data_seq, data_label
#
#
#
# if __name__ == "__main__":
#     a = NiLoader()
#
#     x1, x2 = a.clean()
#

import pandas as pd
import torch
import numpy as np
from pandas import read_csv
import os
from tqdm import tqdm
import torch.utils.data as dat
import xlrd
import csv
import statsmodels.api as sm


class NiLoader:
    """
    step 1: 整理好数据，一定要先让三个文件夹的记录对应起来
    step 2: run body_information
    step 3: run clean()， 我把数据加载写好了，你把之前清理数据那部分放过来就行
    """
    def __init__(self, root_dir="/data/HYK/DATASET/bis/database", nums=44, time_step=1, tw=180):
        self.root_dir = root_dir
        self.nums = nums
        self.time_step = time_step
        self.tw = tw

    def file_name(self, wir):
        # wir: record/ppf, record/rmft, or label
        for root, dirs, files in os.walk(f"{self.root_dir}/{wir}"):
            files.sort()
            return files  # 当前路径下所有非目录子文件,列表

    def body_information(self, files):
        """
        在root_wir下生成生理信息表
        :param files:文件名列表
        :return: 生理信息
        """
        a = []
        for n in range(self.nums):
            b = [n]
            file_dir = f"{self.root_dir}/record/ppf/{files[n]}"
            medicine_data = xlrd.open_workbook(file_dir)
            table = medicine_data.sheets()[0]
            info = table.row_values(1)
            for i in range(1, 8, 2):
                b.append(info[i])

            if b[2] == '女':
                b[2] = 0
            else:
                b[2] = 1
            b[3] = int(b[3].split(' cm')[0])
            b[4] = int(b[4].split(' Kg')[0])
            a.append(b)


        with open(f"{self.root_dir}/info.csv", 'w', newline='') as t:
            writer = csv.writer(t)
            writer.writerow(["caseid", "age", "sex", "height", "weight"])
            writer.writerows(a)
        return a

    def load(self, files, m="ppf"):
        """
        :param files: list, file path list
        :param m: str, "ppf" or "rftn"
        :return: 药物记录 和 时间记录
        """
        a, b, c = [], [], []  # a:药的记录 b:时间记录

        for n in tqdm(range(self.nums)):
            file_dir = f"{self.root_dir}/record/{m}/{files[n]}"
            medicine_data = xlrd.open_workbook(file_dir)
            table = medicine_data.sheets()[0]
            vol = table.col_values(6)
            vol = vol[4:-1]
            times = table.col_values(8)
            times = times[4:-1]
            ce = table.col_values(4)
            ce = ce[4:-1]
            c.append(ce)
            a.append(vol)
            b.append(times)

        return a, b, c

    def load_ni(self, files, m="label"):
        """
        :param files: list, file path list
        :param m: str, "ppf" or "rftn"
        :return: 药物记录 和 时间记录
        """
        a, b = [], []  # a:药的记录 b:时间记录
        for n in tqdm(range(self.nums)):
            file_dir = f"{self.root_dir}/{m}/{files[n]}"
            medicine_data = xlrd.open_workbook(file_dir)
            table = medicine_data.sheets()[0]
            ni = table.col_values(2)
            ni = ni[4:-1]
            times = table.col_values(0)
            times = times[4:-1]
            a.append(ni)
            b.append(times)


        return a, b

    def clean(self):
        files_ppf = self.file_name("record/ppf")
        files_rftn = self.file_name("record/rftn")
        # files_ni = self.file_name("label/")
        files_ni = ['2022-4-21.xls', '2022-4-22.xls', '2022-4-28.xls', '2022-5-27-1.xls', '2022-5-27-2.xls', '2022-5-30-1.xls', '2022-5-30-2.xls', '2022-5-30-3.xls', '2022-6-1-2.xls', '2022-6-1-3.xls', '2022-6-2-1.xls', '2022-6-2-2.xls', '2022-6-2-3.xls', '2022-6-7-1.xls', '2022-6-7-2.xls', '2022-6-7-3.xls', '2022-6-9-1.xls', '2022-6-9-2.xls', '2022-6-10-1.xls', '2022-6-10-2.xls', '2022-6-13-1.xls', '2022-6-13-2.xls', '2022-6-14-1.xls', '2022-6-14-2.xls', '2022-6-15-1.xls', '2022-6-15-2.xls', '2022-6-15-3.xls', '2022-6-15-4.xls', '2022-6-15-5.xls', '2022-6-21-1.xls', '2022-6-21-2.xls', '2022-6-21-3.xls', '2022-6-22-1.xls', '2022-6-23-1.xls', '2022-6-23-2.xls', '2022-6-24-1.xls', '2022-6-24-2.xls', '2022-6-24-3.xls', '2022-6-27-1.xls', '2022-6-27-2.xls', '2022-6-28-1.xls', '2022-6-28-2.xls', '2022-7-1-1.xls', '2022-7-1-2.xls']
        # for k in range(self.nums):
        #     if "NI" in files_ni[k]:
        #         files_ni[k] = files_ni[k].split("NI")[0] + files_ni[k].split("NI")[1]
        # files_ni.sort()
        ppf, tp, ppf_ce = self.load(files_ppf, "ppf")
        rftn, tr, rftn_ce = self.load(files_rftn, "rftn")
        ni, tn = self.load_ni(files_ni)
        for i in range(self.nums):
            data1 = pd.DataFrame()
            filename = files_ppf[i].split(".")[0]
            # step 1：裁剪
            ppf[i], rftn[i], ni[i], ppf_ce[i], rftn_ce[i] = self.crop(ppf[i], rftn[i], ni[i], tp[i], tr[i], tn[i], ppf_ce[i], rftn_ce[i])
            if ppf[i] == 0 and rftn[i] == 0 and ni[i] == 0:
                print("files_ppf {}, files_rftn{}, files_ni{}".format(files_ppf[i], files_rftn[i], files_ni[i]))
                print("i  is {} {}somethinbg wrong".format(i, filename))
                print("ppf  rftn ni {} {} {}".format(ppf[i], rftn[i], ni[i]))
                continue
            # step 2：插值, 交给你了
            # print("ppf shape {}".format(len(ppf[i])))
            print("{} crop finish".format(i))
            for z in range(len(ppf[i])):
                if ppf[i][z] == "":
                    ppf[i][z] = np.nan
                if rftn[i][z] == "":
                    rftn[i][z] = np.nan
                if ni[i][z] == "":
                    ni[i][z] = np.nan
                if ppf_ce[i][z] == "":
                    ppf_ce[i][z] = np.nan
                if rftn_ce[i][z] == "":
                    rftn_ce[i][z] = np.nan
            if np.isnan(ppf[i][0]):
                ppf[i][0] = 0
            if np.isnan(rftn[i][0]):
                rftn[i][0] = 0
            if np.isnan(ppf_ce[i][0]):
                ppf_ce[i][0] = 0
            if np.isnan(rftn_ce[i][0]):
                rftn_ce[i][0] = 0
            if np.isnan(ni[i][0]):
                for l in range(len(ni[i])):
                    flag = np.isnan(ni[i][l])
                    if not flag:
                        ni[i][0] = ni[i][l]
                        break

            data1["PPF20_VOL"] = ppf[i]
            data1["RFTN20_VOL"] = rftn[i]
            data1["BIS"] = ni[i]
            data1["PPF20_CE"] = ppf_ce[i]
            data1["RFTN20_CE"] = rftn_ce[i]
            data1 = data1.astype(float)
            data1.RFTN20_VOL = data1.RFTN20_VOL.interpolate(method='linear', limit_direction='forward', axis=0)
            data1.PPF20_VOL = data1.PPF20_VOL.interpolate(method='linear', limit_direction='forward', axis=0)
            data1.PPF20_CE = data1.PPF20_CE.interpolate(method='linear', limit_direction='forward', axis=0)
            data1.RFTN20_CE = data1.RFTN20_CE.interpolate(method='linear', limit_direction='forward', axis=0)
            data1.BIS = data1.BIS.interpolate(method='linear', limit_direction='forward', axis=0)

            ppf1 = list(range(5 * len(data1.PPF20_VOL.values)))
            source_data_x = list(range(0, 5 * len(data1.PPF20_VOL.values), 5))
            source_data_y = data1.PPF20_VOL.values
            # data1.PPF20_VOL = np.interp(ppf1, source_data_x, source_data_y) * 2  # 浓度标准化处理， vitalDB: 20, NI:10
            data_ppf_vol = np.interp(ppf1, source_data_x, source_data_y) * 2


            rftn1 = list(range(5 * len(data1.RFTN20_VOL.values)))
            source_data_x = list(range(0, 5 * len(data1.RFTN20_VOL.values), 5))
            source_data_y = data1.RFTN20_VOL.values
            data_rftn_vol = np.interp(rftn1, source_data_x, source_data_y) / 2  # 浓度标准化处理， vitalDB: 20, NI:40

            ppf_ce1 = list(range(5 * len(data1.PPF20_CE.values)))
            data_ppf_ce = np.interp(ppf_ce1, source_data_x, data1.PPF20_CE.values)
            rftn_ce1 = list(range(5 * len(data1.RFTN20_CE.values)))
            data_rftn_ce = np.interp(rftn_ce1, source_data_x, data1.RFTN20_CE.values)

            bis = list(range(5 * len(data1.BIS.values)))
            data_BIS = np.interp(bis, source_data_x, data1.BIS.values)
            # data1.to_csv("/HDD_data/HYK/bis/database/ni_dataset/test/{}.csv".format(filename))
            print("{} is ok ".format(i))

            lowess = sm.nonparametric.lowess
            axis = list(range(len(data_BIS)))
            data_BIS = lowess(data_BIS, axis, frac=0.03)[:, 1]

            data = {
                'PPF20_VOL': data_ppf_vol,
                'RFTN20_VOL': data_rftn_vol,
                'PPF20_CE': data_ppf_ce,
                'RFTN20_CE': data_rftn_ce,
                'BIS': data_BIS
            }
            df = pd.DataFrame(data, index=range(5*len(data1.RFTN20_CE.values)))
            df.to_csv(f"{self.root_dir}/proposed_ni/{filename}.csv")
        return ppf, rftn

    @staticmethod
    def crop(x1, x2, x3, t1, t2, t3, ppf_ce, rftn_ce):
        """
        将两种药进行时间对齐, 对不齐的丢掉
        :param x1: ppf record
        :param x2: rftn record
        :param t1: ppf infusion times record
        :param t2: rftn infusion times record
        :return:
        """
        t, i, j, k = 0, 0, 0, 0

        if len(t1) < len(t2):
            t = len(t1)
        else:
            t = len(t2)
        tk = len(t3)
        # t1 t2 t3 , t1<
        while t1[i] - t2[j] != 0.0 or t3[k] - t1[i] != 0.0 or t3[k] - t2[j] != 0.0:

            if t2[j] < t3[k] < t1[i]:
                j += 1
                k += 1
            elif t2[j] < t1[i] < t3[k]:
                j += 1
            elif t1[i] < t2[j] < t3[k]:
                i += 1
            elif t1[i] < t3[k] < t2[j]:
                i += 1
                k += 1
            elif t3[k] < t1[i] < t2[j]:
                k += 1
                i += 1
            elif t3[k] < t2[j] < t1[i]:
                k += 1
                j += 1
            elif t3[k] < t2[j] == t1[i]:
                k += 1
            elif t2[j] < t3[k] == t1[i]:
                j += 1
            elif t1[i] < t2[j] == t3[k]:
                i += 1
            elif t1[i] == t2[j] < t3[k]:
                i += 1
                j += 1
            elif t1[i] == t3[k] < t2[j]:
                i += 1
                k += 1
            elif t2[j] == t3[k] < t1[i]:
                j += 1
                k += 1
            else:
                print("i , j ,k is {} {} {}".format(i,j,k))
                print("t1  is {}".format(t1[i]))
                print("t2 is {}".format(t2[j]))
                print("t3 is {}".format(t3[k]))
            if i == t or j == t or k==tk:
                return 0, 0, 0

        # print("t1  is {}".format(t1[i]))
        # print("t2 is {}".format(t2[j]))
        # print("t3 is {}".format(t3[k]))
        # print("before")
        # print("x1 len is {}".format(len(x1)))
        # print("x2 len is {}".format(len(x2)))
        # print("ppfce len is {}".format(len(ppf_ce)))
        # print("rftnce len is {}".format(len(rftn_ce)))
        x1 = x1[i:-1]
        x2 = x2[j:-1]
        x3 = x3[k:-1]
        ppf_ce = ppf_ce[i:-1]
        rftn_ce = rftn_ce[j:-1]
        # print("after i j ")
        # print("x1 len is {}".format(len(x1)))
        # print("x2 len is {}".format(len(x2)))
        # print("ppfce len is {}".format(len(ppf_ce)))
        # print("rftnce len is {}".format(len(rftn_ce)))
        x1 = x1[::5]
        x2 = x2[::5]
        ppf_ce = ppf_ce[::5]
        rftn_ce = rftn_ce[::5]
        # print("after ni crop")
        # print("x1 len is {}".format(len(x1)))
        # print("x2 len is {}".format(len(x2)))
        # print("ppfce len is {}".format(len(ppf_ce)))
        # print("rftnce len is {}".format(len(rftn_ce)))

        flag = True
        while flag:
            print("x1 len is {}".format(len(x1)))
            print("x2 len is {}".format(len(x2)))
            print("x3 len is {}".format(len(x3)))

            if len(x1) < len(x2):
                x2 = x2[:len(x1)]
            else:
                x1 = x1[:len(x2)]

            if len(x1) < len(x3):
                x3 = x3[:len(x1)]
            else:
                x1 = x1[:len(x3)]
            if len(x1) == len(x2) and len(x3) == len(x2):
                flag = False
            else:
                continue
        ppf_ce = ppf_ce[:len(x1)]
        rftn_ce = rftn_ce[:len(x2)]
        if len(ppf_ce) != len(rftn_ce) or len(ppf_ce) != len(x1) or len(rftn_ce)!= len(x2):
            print("ce length error")
        return x1, x2, x3, ppf_ce, rftn_ce

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
        case_information = read_csv(f'/HDD_data/HYK/bis/database/new_{traindata}_clean.csv')
        case_id = self.file_name(data=traindata)
        for i in range(len(case_id)):
            case_id[i] = case_id[i].split('.')[0]  # 字符串转数字

        case_id = list(map(int, case_id))
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


if __name__ == "__main__":
    a = NiLoader()

    x1, x2 = a.clean()











