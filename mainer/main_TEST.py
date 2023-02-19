import torch
import matplotlib.pyplot as plt
import tqdm
from loader import database
import argparse
import evaluate
import numpy as np
import random
from model.TEST import trainer, params
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
            database_wdir="/home/user02/HYK/bis/database",
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


            vaild_loader = d_box.train_data_loader(
                data="test",
                batch=args.vaild_batch,
                batch_size=512
            )

            d_box.time_step = 10
            train_loader = d_box.train_data_loader(
                batch=args.train_batch,
                batch_size=1024,
                shuffle=args.shuffle
            )

            box.decodertrain(
                X=train_loader,
                X2=vaild_loader,
                lr=args.lr,
                model_file=args.best_file,
                model2_file=args.best_file2,
                best_loss=args.best_loss
            )

            test_out = box.test(
                X=test_loader,
                epoch_pth=args.best_file,
                test_batch=1)

            ist, isp = d_box.time_devide(case_nums=76, traindata="test")
            access = evaluate.Evalulate(test_label, test_out, ist, isp, case_num=76)
            print("MDPE    MDAPE    RMSE\r")
            for i in range(4):
                print("%.2f     %.2f     %.2f" % access.loss(i))

        elif mode == "test":
            box.decodertrain(
                X=train_loader,
                X2=vaild_loader,
                lr=args.lr,
                model_file=args.best_file2,
                best_loss=args.best_loss
            )


        # pre_tr_times = 46
        # pre_file = f'/home/user02/HYK/bis_transformer/output/TEST/model/epoch{pre_tr_times}.pth'
        # test_out = box.test(
        #     X=test_loader,
        #     epoch_pth=pre_file,
        #     test_batch=1)

        plt.grid(True)
        plt.autoscale(axis='x', tight=True)
        for i in range(1):
            plt.figure()
            plt.plot(test_label[i])
            plt.plot(test_out[i])
            plt.savefig(f'/home/user02/HYK/bis/attention/output_picture/case{i}.png')
            plt.show()

