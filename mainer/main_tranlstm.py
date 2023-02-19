import torch
import matplotlib.pyplot as plt
import tqdm
from loader import database
import argparse
import evaluate
import evaluate_output
import numpy as np
import random
from model.tranlstm import trainer, params_save
import statsmodels.api as sm
import imp


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(2)
# 训练或测试模式
mode = 'train'
# 训练参数读取
args = params_save.Params.trainparam()

if __name__ == "__main__":
    with torch.cuda.device(args.device):
        args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"GPU{args.device} open")
        else:
            print("cpu open")

        box = trainer.Trainer(args)
        d_box = database.Dataloader(
            database_wdir="/HDD_data/HYK/bis/database",
            time_step=1,
            nums=1,
            tw=args.tw
        )
        # 开始训练或测试
        if mode == "train":

            d_box.time_step = 10
            vaild_loader = d_box.train_data_loader(
                data="test",
                batch=args.vaild_batch,
                batch_size=512
            )
            d_box.time_step = 10
            train_loader = d_box.train_data_loader(
                batch=args.train_batch,
                batch_size=3000,
                shuffle=True,
            )

            data_label, label_num = d_box.label_stat(180, "train")
            p = np.asarray(label_num)/len(data_label)+1e-3
            for i in range(10):
                p = np.append(p, 1e-3)

            plt.scatter(p)
            lds_kernel_window = evaluate.Evalulate.get_lds_kernel_window(kernel='gaussian', ks=10, sigma=8)
            from scipy.ndimage import convolve1d
            eff_label_dist = convolve1d(p, weights=lds_kernel_window, mode='constant')
            box.train(
                X=train_loader,
                X2=vaild_loader,
                model_file=args.best_file,
                best_loss=args.best_loss,
                config=args,
                p=eff_label_dist
            )

            d_box.time_step = 1
            test_loader, test_label = d_box.test_data_loader(
                data="test",
                batch=76,
                batch_size=256
            )

            test_out = box.test(
                X=test_loader,
                epoch_pth=args.best_file,
                test_batch=76)

            ist, isp = d_box.time_devide(case_nums=76, traindata="test")
            access = evaluate.Evalulate(test_out, test_label, ist, isp, case_num=76)
            print("MDPE    MDAPE    RMSE\r")
            for i in range(4):
                x = access.loss(i)
                print("%.2f  " % x["meanMDPE"],
                      "%.2f  " % x["meanMDAPE"],
                      "%.2f  " % x["meanRMSE"],
                      "%.2f  " % x["meanMAE"])

            lowess = sm.nonparametric.lowess
            test_new = list(range(76))
            for i in tqdm.tqdm(range(76)):
                axis = list(range(len(test_out[i])))
                test_new[i] = lowess(test_out[i], axis, frac=0.005)[:, 1]
            access = evaluate.Evalulate(test_new, test_label, ist, isp, case_num=76)
            print("MDPE    MDAPE    RMSE    MAE\r")
            for i in range(4):
                x = access.loss(i)
                print("%.2f  " % x["meanMDPE"],
                      "%.2f  " % x["meanMDAPE"],
                      "%.2f  " % x["meanRMSE"],
                      "%.2f  " % x["meanMAE"])

            access = evaluate_output.Evalulate(test_new, test_label, ist, isp, case_num=76)
            file = {}
            for i in range(4):
                X = np.asarray(access.loss(i))
                name = ["mdpe", "mdape", "rmse",
                        "induction_mdpe", "induction_mdape", "induction_rmse",
                        "maintence_mdpe", "maintence_mdape", "maintence_rmse",
                        "recovery_mdpe", "recovery_mdape", "recovery_rmse"]
                for j in range(3):
                    file[f"{name[3*i+j]}"] = X[j, :]

            import pandas as pd
            df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in file.items()]))

            df.to_csv('/HDD_data/HYK/bis/database/result.csv')



        elif mode == "test":
            test_loader, test_label = d_box.test_data_loader(
                data="test",
                batch=args.test_batch,
                batch_size=128
            )
            # pre_tr_times = 25
            # pre_file = f'/home/user02/HYK/bis_transformer/output/tranlstm/model/epoch{pre_tr_times}.pth'
            test_out = box.test(
                X=test_loader,
                epoch_pth=args.best_file,
                test_batch=1)

            # plt.grid(True)
            # plt.autoscale(axis='x', tight=True)
            for i in range(10):
                plt.figure()
                plt.plot(test_label[i], label="ground truth")
                plt.plot(test_new[i], label="ours")
                plt.legend()
                plt.ylabel("bis index")
                plt.xlabel("t/s")
                # plt.savefig(f'/home/user02/HYK/bis/attention/output_picture/case{i}.png')
                plt.show()




