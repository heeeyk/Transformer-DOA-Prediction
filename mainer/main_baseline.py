import torch
import matplotlib.pyplot as plt
import tqdm
from loader import database
import evaluate
import numpy as np
import random
from model.baseline import trainer, params
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
mode = 'test'
# 训练参数读取
args = params.Params.trainparam()

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
            tw=180
        )
        # 开始训练或测试
        if mode == "train":

            test_loader, test_label = d_box.test_data_loader(
                data="test",
                batch=76,
                batch_size=128
            )

            d_box.time_step = 10
            vaild_loader = d_box.train_data_loader(
                data="test",
                batch=30,
                batch_size=512
            )

            train_loader = d_box.train_data_loader(
                batch=100,
                batch_size=1024,
            )

            box.train(
                X=train_loader,
                X2=vaild_loader,
                lr=args.lr,
                model_file=args.best_file,
                best_loss=args.best_loss
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

        elif mode == "test":
            test_loader, test_label = d_box.test_data_loader(
                batch=args.test_batch,
                batch_size=76
            )

            pre_tr_times = 9
            pre_file = f'/home/user02/HYK/bis_transformer/output/baseline/model/epoch{pre_tr_times}.pth'
            test_out = box.test(
                X=test_loader,
                epoch_pth=args.best_file,
                test_batch=76)

            import statsmodels.api as sm
            lowess = sm.nonparametric.lowess
            new = list(range(76))
            for i in tqdm.tqdm(range(76)):
                axis = list(range(len(test_out[i])))
                new[i] = lowess(test_out[i], axis, frac=0.03)[:, 1]
            ist, isp = d_box.time_devide(case_nums=76, traindata="test")
            access = evaluate.Evalulate(test_label, new, ist, isp, case_num=76)
            print("MDPE    MDAPE    RMSE\r")
            for i in range(4):
                print("%.2f     %.2f     %.2f" % access.loss(i))


            plt.grid(True)
            plt.autoscale(axis='x', tight=True)
            for i in range(4,9):
                plt.figure()
                plt.plot(test_label[i])
                plt.plot(test_out[i])
                plt.show()

