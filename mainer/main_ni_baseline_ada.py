import torch
import matplotlib.pyplot as plt
import tqdm
from loader import database
from loader import ni_database
import argparse
import evaluate
import evaluate_output
import numpy as np
import random
from model.baseline import trainer, params_ni
import statsmodels.api as sm
import imp


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
# setup_seed(2407)
# 训练或测试模式
mode = 'test'
# 训练参数读取
args = params_ni.Params.trainparam()

if __name__ == "__main__":
    with torch.cuda.device(args.device):
        args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"GPU{args.device} open")
        else:
            print("cpu open")

        box = trainer.Trainer(args)
        d_box = ni_database.Dataloader(
            database_wdir="/data/HYK/DATASET/bis/database",
            time_step=1,
            nums=1,
            tw=args.tw
        )
        # 开始训练或测试
        if mode == "train":

            vaild_loader, train_loader, test_loader, test_label = d_box.load_all(
                512, 1024, 64
            )

            box.train(
                X=train_loader,
                X2=vaild_loader,
                lr=args.lr,
                model_file=args.best_file,
                best_loss=args.best_loss,
            )

        elif mode == "test":
            test_batch = 20
            test_loader, test_label = d_box.pt_load(
                dataset="test",
                batch_size=64,
            )

            # vaild_loader, train_loader, test_loader, test_label = d_box.load_all(
            #     512, 3000, 256
            # )
            # pre_tr_times = 25
            # pre_file = f'/home/user02/HYK/bis_transformer/output/tranlstm/model/epoch{pre_tr_times}.pth'
            test_out = box.test(
                X=test_loader,
                epoch_pth='/data/HYK/DATASET/bis/output/baseline/model/best_epoch6626.pth',
                test_batch=test_batch)

            test_out2 = box.test(
                X=test_loader,
                epoch_pth=args.best_file,
                test_batch=test_batch)

            for i in range(len(test_label)):
                test_label[i] = test_label[i][:len(test_out[i])]
            ist, isp = d_box.time_devide(case_nums=test_batch, traindata="test")
            access = evaluate.Evalulate(test_out2, test_label, ist, isp, case_num=test_batch)
            print("MDPE    MDAPE    RMSE\r")
            for i in range(4):
                x = access.loss(i)
                print("%.2f  " % x["meanMDPE"],
                      "%.2f  " % x["meanMDAPE"],
                      "%.2f  " % x["meanRMSE"],
                      "%.2f  " % x["meanMAE"])

            """
                平滑后的结果
            """
            lowess = sm.nonparametric.lowess
            test_new = list(range(test_batch))
            for i in tqdm.tqdm(range(test_batch)):
                axis = list(range(len(test_out2[i])))
                test_new[i] = lowess(test_out2[i], axis, frac=0.005)[:, 1]
            access = evaluate.Evalulate(test_new, test_label, ist, isp, case_num=test_batch)
            print("MDPE    MDAPE    RMSE    MAE\r")
            for i in range(4):
                x = access.loss(i)
                print("%.2f  " % x["meanMDPE"],
                      "%.2f  " % x["meanMDAPE"],
                      "%.2f  " % x["meanRMSE"],
                      "%.2f  " % x["meanMAE"])


            # plt.grid(True)
            # plt.autoscale(axis='x', tight=True)
            for i in range(test_batch):
                plt.figure()
                plt.plot(test_label[i], 'silver', label="ground truth")
                plt.plot(test_out[i], label="ours")
                plt.plot(test_new[i], label="fine tuned")
                plt.ylabel("Bispectral index")
                plt.xlabel("Time(sec)")
                plt.legend(loc='upper right')
                plt.savefig(f'/data/HYK/DATASET/bis/output/NI/case{i}.png')
                # plt.show()




