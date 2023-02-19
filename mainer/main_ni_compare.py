import torch
import matplotlib.pyplot as plt
import tqdm
from loader import ni_database as database
import evaluate
import numpy as np
import random
from model.baseline import trainer as basemodel
from model.featurefusion import trainer as mymodel
from model.baseline import params as baseparams
from model.featurefusion import params_save as myparams
from model.traditional import model as oldmodel
import statsmodels.api as sm
import os
import imp


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(2)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# 训练参数读取
base_args = baseparams.Params.trainparam()
base_args.best_file = '/data/HYK/DATASET/bis/output/baseline/best_epoch.pth'
args = myparams.Params.trainparam()
args.best_file = '/data/HYK/DATASET/bis/output/featurefusion/model/best_epoch(9.52).pth'

if __name__ == "__main__":
    # with torch.cuda.device(args.device):
    #     args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    #     if torch.cuda.is_available():
    #         print(f"GPU{args.device} open")
    #     else:
    #         print("cpu open")
    #     print(1)
    box1 = basemodel.Trainer(base_args)
    box2 = mymodel.Trainer(args)
    d_box = database.Dataloader(
        database_wdir="/data/HYK/DATASET/bis/database",
        time_step=1,
        nums=1,
        tw=args.tw
    )

    vaild_loader, train_loader, test_loader, test_label = d_box.load_all(
        512, 3000, 256, test_box=0
    )

    # from loader import ni_database
    # ni_box = ni_database.Dataloader(
    #     database_wdir="/HDD_data/HYK/bis/database/ni_dataset",
    #     time_step=1,
    #     nums=1,
    #     tw=args.tw
    # )
    # from model.ni_ada import trainer, params_save
    # args = params_save.Params.trainparam()
    # box2 = trainer.Trainer(args)
    # vaild_loader, train_loader, test_loader, test_label = ni_box.load_all(
    #     512, 3000, 32
    # )

    test_patch = 20
    """
        baseline
    """
    print('lstm')
    base_out = box1.test(
        X=test_loader,
        epoch_pth=base_args.best_file,
        test_batch=test_patch)
    """
        my model
    """
    print('my model')
    my_out = box2.test(
        X=test_loader,
        epoch_pth=args.best_file,
        test_batch=test_patch)
    """
        pkpd model
    """
    print('pk-pd')
    pkpd_out = d_box.ceload(case_nums=test_patch, traindata='test')

    """
        pred smoothing
    """
    print('smoothing...')
    lowess = sm.nonparametric.lowess
    new = [0, 0]
    for index, x in enumerate((my_out, base_out)):
        new[index] = list(range(test_patch))
        for i in tqdm.tqdm(range(test_patch)):
            axis = list(range(len(x[i])))
            new[index][i] = lowess(x[i], axis, frac=0.008)[:, 1]

            # access = evaluate.Evalulate(test_label, new[index], ist, isp, case_num=test_patch)
            # print("MDPE    MDAPE    RMSE\r")
            # for i in range(4):
            #     print("%.2f     %.2f     %.2f" % access.loss(i))

        # for i in range(1):
        #     plt.figure()
        #     plt.plot(test_label[i])
        #     plt.plot(new[0][i])
        #     plt.plot(new[1][i])
        #     plt.plot(pkpd_out[i])
        #     plt.show()

"""
    Metrics compare
"""
ist, isp = d_box.time_devide(case_nums=test_patch, traindata='test')
access0 = evaluate.Evalulate(new[0], test_label, ist, isp, case_num=test_patch)  # my model
access1 = evaluate.Evalulate(new[1], test_label, ist, isp, case_num=test_patch)  # baseline
access2 = evaluate.Evalulate(pkpd_out, test_label, ist, isp, case_num=test_patch)
label_error = [0, 0]
data_label, label_num = d_box.label_stat(24, "train")
_, label_error[0] = access0.test_error(label_num)
_, label_error[1] = access1.test_error(label_num)

database.data_distribution_bar(label_num/sum(label_num)*100, label_error)


x0 = 0
x1 = 0
print("MDPE    MDAPE    RMSE\r")
for i in range(4):
    x0 = access0.loss(i)
    x1 = access1.loss(i)
    x2 = access2.loss(i)
    print("%.2f  " % x0["meanMDPE"],
          "%.2f  " % x0["meanMDAPE"],
          "%.2f  " % x0["meanRMSE"],
          "%.2f  " % x0["meanMAE"])

print('base compare picture plotting...')
save_path = '/HDD_data/HYK/bis/output/compare_pkpd_lstm/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
for i in tqdm.tqdm(range(test_patch)):
    plt.figure()
    plt.rcParams['figure.figsize'] = (6.0, 8.0)
    plt.subplot(2, 1, 1)
    plt.plot(test_label[i], 'silver', label='Label')
    plt.plot(pkpd_out[i], label='PK-PD')
    plt.legend(loc='upper right')
    plt.ylabel("Bispectral index")
    plt.subplot(2, 1, 2)
    plt.plot(test_label[i], 'silver', label='Label')
    plt.plot(new[1][i], label='LSTM', color='darkorange')
    plt.ylabel("Bispectral index")
    plt.xlabel("Time(sec)")
    plt.legend(loc='upper right')
    plt.savefig(f'{save_path}/case{i}.png')
    plt.close()

print('ours compare picture plotting...')
save_path = '/HDD_data/HYK/bis/output/compare_ours_lstm/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
for i in tqdm.tqdm(range(test_patch)):
    plt.figure()
    plt.rcParams['figure.figsize'] = (6.0, 8.0)
    plt.subplot(2, 1, 1)
    plt.plot(test_label[i], 'silver', label='Label')
    plt.plot(my_out[i], label='PK-PD')
    plt.legend(loc='upper right')
    plt.ylabel("Bispectral index")
    plt.subplot(2, 1, 2)
    plt.plot(test_label[i], 'silver', label='Label')
    plt.plot(new[1][i], label='LSTM', color='darkorange')
    plt.ylabel("Bispectral index")
    plt.xlabel("Time(sec)")
    plt.legend(loc='upper right')
    plt.savefig(f'{save_path}/case{i}.png')
    plt.close()



for i in tqdm.tqdm(range(44)):
    # plt.figure()
    plt.rcParams['figure.figsize'] = (6.0, 4.0)
    # plt.subplot(2, 1, 1)
    plt.plot(test_label[i], 'silver', label='Ground true')
    plt.plot(new[0][i], label='Ours')
    plt.legend(loc='upper right')
    plt.ylabel("Bispectral index")
    # plt.subplot(2, 1, 2)
    # plt.plot(new[1][i]/2.5, label='Baseline', color='darkorange')
    plt.plot(pkpd_out[i], label='PK-PD', color='green')
    # plt.ylabel("Bispectral index")
    plt.xlabel("Time(sec)")
    # plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(f'/data/HYK/DATASET/bis/database/ours_box_output/case{i}.png')
    plt.close()


best_case = [5, 21, 15, 25, 27, 29, 38, 45, 46, 47, 56, 72]
loc = [1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20]
a = [47, 54, 72]

ni_best = [0, 1, 2, 4, 5, 6, 11, 12, 13]
# 4行3列 ours跟baseline一起画
for i, case in enumerate(best_case):
    plt.rcParams['figure.figsize'] = (24.0, 12.0)
    plt.subplot(3, 4, i+1)
    plt.plot(test_label[case], 'silver', label='Ground True')
    plt.plot(pkpd_out[case], label='PK-PD', color='green')
    plt.plot(new[1][case], label='Baseline', color='lightcoral')
    plt.plot(new[0][case], label='Ours')
    plt.legend(loc='upper right')
    plt.ylabel("Bispectral index")
    plt.xlabel("Time(sec)")
# plt.show()
plt.savefig(f'/HDD_data/HYK/bis/output/3_method_compare.png',bbox_inches='tight',pad_inches=0.0)
plt.close()

