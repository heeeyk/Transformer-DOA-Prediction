import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import tqdm


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        return files  # 当前路径下所有非目录子文件,列表


def get_test_clinical(fileload="new_train_clean"):

    file_list = file_name(f'/home/user02/HYK/bis/database/ce_clean/{fileload}')
    file_num = len(file_list)

    clinical = pd.read_csv('/home/user02/HYK/bis/information.csv',
                           usecols=['caseid', 'age', 'sex', 'height', 'weight', "bmi"])
    for i in range(file_num):
        if clinical['sex'][i] == 'F':
            clinical['sex'][i] = 0
        else:
            clinical['sex'][i] = 1
    data = pd.DataFrame()

    for i in range(len(file_list)):
        fileid = int(file_list[i].split('.csv')[0])
        clinical_data = clinical.loc[clinical.caseid == fileid, "age":"bmi"]
        if clinical_data['sex'].item() == 'F':
            clinical_data['sex'] = 0
        else:
            clinical_data['sex'] = 1
        data = data.append([{"caseid": fileid,
                             'age': float(clinical_data['age']),
                             "sex":float(clinical_data['sex']),
                             "weight":float(clinical_data['weight']),
                             "height":float(clinical_data['height']),
                             "bmi": float(clinical_data['bmi'])}])
    data.to_csv(f'/home/user02/HYK/bis/database/{fileload}.csv',
                encoding='utf-8')


def data_clean(file, train=True):
    #  读取数据
    print(file)
    people = pd.read_csv(f'/home/user02/HYK/bis/vital_1s/{file}')
    people = people.rename(columns=lambda x: x.replace("Solar8000/", "").replace("Orchestra/", "").replace("BIS/", ""))

    #  删除错误的心率数据
    people.HR = people.HR.interpolate(method='linear', limit_direction='forward', axis=0)
    people = people.dropna(subset=['HR'])
    people.index = range(0, len(people))
    people = people.drop(people[people.HR == 0].index, axis=0)
    people.index = range(0, len(people))

    #  删除错误的BIS数据
    people.BIS = people.BIS.interpolate(method='linear', limit_direction='forward', axis=0)
    people = people.dropna(subset=['BIS'])
    people.index = range(0, len(people))
    people = people.drop(people[people.BIS == 0].index, axis=0)
    people.index = range(0, len(people))

    #  BIS平滑
    if train:
        lowess = sm.nonparametric.lowess
        people.BIS = lowess(people.BIS, people.index, frac=0.03)[:, 1]

    #  错误数据补齐
    # people.PPF20_RATE = people.PPF20_RATE.interpolate(method='linear', limit_direction='forward', axis=0)
    # people.RFTN20_RATE = people.RFTN20_RATE.interpolate(method='linear', limit_direction='forward', axis=0)
    people.RFTN20_VOL = people.RFTN20_VOL.interpolate(method='linear', limit_direction='forward', axis=0)
    people.PPF20_VOL = people.PPF20_VOL.interpolate(method='linear', limit_direction='forward', axis=0)
    people = people.fillna(0)
    for i in range(len(people)-1):
        if np.abs(people.RFTN20_VOL[i+1] - people.RFTN20_VOL[i]) >= 10:
            people.RFTN20_VOL[i+1] = people.RFTN20_VOL[i]
        if np.abs(people.PPF20_VOL[i+1] - people.PPF20_VOL[i]) >= 10:
            people.PPF20_VOL[i+1] = people.PPF20_VOL[i]

    # 丢掉只有前半场手术数据的样本
    if people.BIS[len(people)-1] <= 60:
        return 0

    #  丢弃前100s内数据缺失超过30s的样本
    for i in range(100):
        if people.time[i+1] - people.time[i] >= 30:
            return 0

    #  保存数据
    if people.RFTN20_VOL[0] == 0 and people.PPF20_VOL[0] == 0 and people.BIS[0] >= 80:
        if train:
            people.to_csv(f'/home/user02/HYK/bis/new_train/{file}', encoding='utf-8')
        if not train:
            people.to_csv(f'/home/user02/HYK/bis/new_test/{file}', encoding='utf-8')
        print(file, "loading finish")
        return 1
    else:
        return 0


def ce_data_clean(file, train=True):
    #  读取数据
    print(file)
    people = pd.read_csv(f'/home/user02/HYK/bis/database/ce/{file}')
    people = people.rename(columns=lambda x: x.replace("Solar8000/", "").replace("Orchestra/", "").replace("BIS/", ""))

    #  删除错误的心率数据
    people.HR = people.HR.interpolate(method='linear', limit_direction='forward', axis=0)
    people = people.dropna(subset=['HR'])
    people.index = range(0, len(people))
    people = people.drop(people[people.HR == 0].index, axis=0)
    people.index = range(0, len(people))

    #  删除错误的BIS数据
    people.BIS = people.BIS.interpolate(method='linear', limit_direction='forward', axis=0)
    people = people.dropna(subset=['BIS'])
    people.index = range(0, len(people))
    people = people.drop(people[people.BIS == 0].index, axis=0)
    people.index = range(0, len(people))

    #  BIS平滑
    if train:
        lowess = sm.nonparametric.lowess
        people.BIS = lowess(people.BIS, people.index, frac=0.03)[:, 1]

    #  错误数据补齐
    # people.PPF20_RATE = people.PPF20_RATE.interpolate(method='linear', limit_direction='forward', axis=0)
    # people.RFTN20_RATE = people.RFTN20_RATE.interpolate(method='linear', limit_direction='forward', axis=0)
    people.PPF20_VOL = people.PPF20_VOL.interpolate(method='linear', limit_direction='forward', axis=0)
    people.PPF20_CP = people.PPF20_CP.interpolate(method='linear', limit_direction='forward', axis=0)
    people.PPF20_CE = people.PPF20_CE.interpolate(method='linear', limit_direction='forward', axis=0)

    people.RFTN20_VOL = people.RFTN20_VOL.interpolate(method='linear', limit_direction='forward', axis=0)
    people.RFTN20_CP = people.RFTN20_CP.interpolate(method='linear', limit_direction='forward', axis=0)
    people.RFTN20_CE = people.RFTN20_CE.interpolate(method='linear', limit_direction='forward', axis=0)


    people = people.fillna(0)
    for i in range(len(people)-1):
        if np.abs(people.RFTN20_VOL[i+1] - people.RFTN20_VOL[i]) >= 10:
            people.RFTN20_VOL[i+1] = people.RFTN20_VOL[i]
        if np.abs(people.PPF20_VOL[i+1] - people.PPF20_VOL[i]) >= 10:
            people.PPF20_VOL[i+1] = people.PPF20_VOL[i]

    # 丢掉只有前半场手术数据的样本
    if people.BIS[len(people)-1] <= 60:
        return 0

    #  丢弃前100s内数据缺失超过30s的样本
    for i in range(100):
        if people.time[i+1] - people.time[i] >= 30:
            return 0

    #  保存数据
    if people.RFTN20_VOL[0] == 0 and people.PPF20_VOL[0] == 0 and people.BIS[0] >= 80:
        if train:
            people.to_csv(f'/home/user02/HYK/bis/database/ce_clean/train/{file}', encoding='utf-8')
            return 1
        if not train:
            people.to_csv(f'/home/user02/HYK/bis/database/ce_clean/test/{file}', encoding='utf-8')
        print(file, "loading finish")
        return 1
    else:
        return 0

def casefile_clean():
    file_list = file_name('/HDD_data/HYK/bis/ce_clean/train')
    print(len(file_list), "files was found")
    x = 0   # 加载的第x个case

    y = 0   # 符合要求的case，加载训练集
    train = True
    while y < 300:
        if ce_data_clean(file_list[x], train) == 1:
            y += 1
        x += 1

    y = 0   # 符合要求的case清零，加载测试集
    train = False
    while y < 300:
        if ce_data_clean(file_list[x], train) == 1:
            y += 1
        x += 1


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def informationfile_clean():
    people = pd.read_csv('/home/user02/HYK/bis/before_information.csv')
    people.age = normalization(people.age)
    people.height = normalization(people.height)
    people.weight = normalization(people.weight)
    people.to_csv('/home/user02/HYK/bis/clean_data1/information.csv', encoding='utf-8')


# casefile_clean()
# informationfile_clean()
# get_test_clinical("vaild")






