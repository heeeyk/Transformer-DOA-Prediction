import time

import torch
import matplotlib.pyplot as plt
from loader import data_loader1
import pandas as pd
import argparse
import train_test
import evaluate
import imp
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)

# 训练或测试模式
mode = 'train'

parser = argparse.ArgumentParser()
parser.add_argument('--model_name',  default="transformer", type=str)
parser.add_argument('--tw',          default=180,     type=int)
parser.add_argument('--train_batch', default=100,     type=int)
parser.add_argument('--vaild_batch', default=30,      type=int)
parser.add_argument('--test_batch',  default=76,      type=int)
parser.add_argument('--batch_size',  default=64,      type=int)
parser.add_argument('--train_epoch', default=100,      type=int)
parser.add_argument('--lr',          default=3e-1,    type=float)
parser.add_argument('--pre_train',   default=True,    type=bool)
parser.add_argument('--pre_tr_times', default=6,      type=int)
parser.add_argument('--device',      default=0,       type=int)

args = parser.parse_args()

# 预训练文件路径
pre_file = f'/home/user02/HYK/bis_transformer/output/epoch{args.pre_tr_times}.pth'
# 保存文件路径
save_file = f'/home/user02/HYK/bis_transformer/output/epoch{args.pre_tr_times}.pth'


if __name__ == "__main__":
    with torch.cuda.device(args.device):
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"GPU{args.device} open")
        else:
            print("cpu open")

        # 开始训练或测试
        if mode == "train":
            test_loader, test_label = data_loader1.test_data_loader(
                tw=args.tw,
                data="test",
                batch=args.test_batch,
                batch_size=256,
                timestep=1)
            train_loader = data_loader1.train_data_loader(
                tw=args.tw,
                batch=args.train_batch,
                batch_size=256,
                time_step=10)
            vaild_loader, vaild_label = data_loader1.test_data_loader(
                tw=args.tw,
                data="vaild",
                batch=args.vaild_batch,
                batch_size=512,
                timestep=1)

            train_test.train(
                X=train_loader,
                X2=vaild_loader,
                model_name=args.model_name,
                device=args.device,
                epoch=args.train_epoch,
                lr=args.lr,
                pre_train=args.pre_train,
                epoch_pth=pre_file,
                vaild_label=vaild_label)

            test_out = train_test.test(
                X=test_loader,
                device=args.device,
                epoch_pth=save_file,
                test_batch=1)

            ist, isp = data_loader1.time_devide(case_nums=76, traindata="test")
            access = evaluate.Evalulate(test_label, test_out, ist, isp, case_num=76)
            print("MDPE    MDAPE    RMSE\r")
            for i in range(4):
                print("%.2f     %.2f     %.2f" % access.loss(i))

        elif mode == "test":
            test_loader, test_label = data_loader1.test_data_loader(
                tw=args.tw,
                batch=args.test_batch,
                batch_size=100,
                timestep=1)
            pre_tr_times = 11
            pre_file = f'/home/user02/HYK/bis_transformer/output/epoch{pre_tr_times}.pth'
            test_out = train_test.test(
                X=test_loader,
                device=args.device,
                epoch_pth=pre_file,
                test_batch=8)

            plt.grid(True)
            plt.autoscale(axis='x', tight=True)
            for i in range(8):
                plt.figure()
                plt.plot(test_label[i])
                plt.plot(test_out[i])
                plt.savefig(f'/home/user02/HYK/bis/attention/output_picture/case{i}.png')
                plt.show()


        # elif mode == "save":
        #     pre_bis = {}
        #     for i in range(len(test_out)):
        #         pre_bis[f"case{i}"] = test_out[i]
        #     df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pre_bis.items()]))
        #     df.to_csv('/home/user02/HYK/bis/attention/train_bis/pre_bis.csv')








