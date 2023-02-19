import torch
import matplotlib.pyplot as plt
import tqdm
from loader import database
import argparse
import evaluate
import numpy as np
import random
from model.model520 import trainer, params_save
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

            test_loader, test_label = d_box.test_data_loader(
                data="test",
                batch=args.test_batch,
                batch_size=128
            )

            d_box.time_step = 10
            vaild_loader = d_box.train_data_loader(
                data="test",
                batch=args.vaild_batch,
                batch_size=512
            )
            d_box.time_step = 11
            train_loader = d_box.train_data_loader(
                batch=args.train_batch,
                batch_size=2048,
                shuffle=True,
            )

            box.train(
                X=train_loader,
                X2=vaild_loader,
                model_file=args.best_file,
                best_loss=args.best_loss,
                config=args
            )

            test_out = box.test(
                X=test_loader,
                epoch_pth=args.best_file,
                test_batch=76)

            ist, isp = d_box.time_devide(case_nums=76, traindata="test")
            access = evaluate.Evalulate(test_label, test_out, ist, isp, case_num=76)
            print("MDPE    MDAPE    RMSE\r")
            for i in range(4):
                print("%.2f     %.2f     %.2f" % access.loss(i))



            lowess = sm.nonparametric.lowess
            test_new = list(range(76))
            for i in tqdm.tqdm(range(76)):
                axis = list(range(len(test_out[i])))
                test_new[i] = lowess(test_out[i], axis, frac=0.03)[:, 1]
            access = evaluate.Evalulate(test_label, test_new, ist, isp, case_num=76)
            print("MDPE    MDAPE    RMSE\r")
            for i in range(4):
                print("%.2f     %.2f     %.2f" % access.loss(i))

            access = evaluate.Evalulate(test_label, test_new, ist, isp, case_num=76)
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
                tw=args.tw,
                batch=args.test_batch,
                batch_size=100,
                timestep=1)
            pre_tr_times = 25
            pre_file = f'/home/user02/HYK/bis_transformer/output/tranlstm/model/epoch{pre_tr_times}.pth'
            test_out = box.test(
                X=test_loader,
                epoch_pth=pre_file,
                test_batch=1)

            plt.grid(True)
            plt.autoscale(axis='x', tight=True)
            for i in range(3):
                plt.figure()
                plt.plot(test_label[i])
                plt.plot(test_out[i])
                # plt.savefig(f'/home/user02/HYK/bis/attention/output_picture/case{i}.png')
                plt.show()




