import torch
import matplotlib.pyplot as plt
from loader import data_loader1
import argparse
import evaluate
import numpy as np
import random
from model.base_transformer import trainer
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

parser = argparse.ArgumentParser()
parser.add_argument('--model_name',  default="base", type=str)
parser.add_argument('--tw',          default=180,     type=int)
parser.add_argument('--train_batch', default=100,     type=int)
parser.add_argument('--vaild_batch', default=30,      type=int)
parser.add_argument('--test_batch',  default=76,      type=int)
parser.add_argument('--batch_size',  default=64,      type=int)
parser.add_argument('--train_epoch', default=30,      type=int)
parser.add_argument('--lr',          default=3e-3,    type=float)
parser.add_argument('--pre_train',   default=True,    type=bool)
parser.add_argument('--pre_tr_times', default=0,      type=int)
parser.add_argument('--device',      default=1,       type=int)
parser.add_argument('--best_loss',   default=80000,       type=int)

args = parser.parse_args()

# 预训练文件路径
pre_file = f'/home/user02/HYK/bis_transformer/output/{args.model_name}/epoch{args.pre_tr_times}.pth'
best_file = f'/home/user02/HYK/bis_transformer/output/{args.model_name}/best_epoch.pth'
# 保存文件路径
save_file = f'/home/user02/HYK/bis_transformer/output/{args.model_name}/epoch{args.pre_tr_times}.pth'


if __name__ == "__main__":
    with torch.cuda.device(args.device):
        args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"GPU{args.device} open")
        else:
            print("cpu open")

        # 开始训练或测试
        if mode == "train":
            test_loader, test_label = data_loader1.test_data_loader(
                tw=args.tw,
                data="test",
                batch=7,
                batch_size=128,
                timestep=1)
            train_loader = data_loader1.train_data_loader(
                tw=args.tw,
                batch=150,
                batch_size=1024,
                time_step=1)
            vaild_loader = data_loader1.train_data_loader(
                tw=args.tw,
                data="vaild",
                batch=5,
                batch_size=1024,
                time_step=1)

            box = trainer.Trainer(
                model_name=args.model_name,
                device=args.device,
                epoch=args.train_epoch,
                epoch_pth=best_file,
                pre_train=args.pre_train,
                pre_tr_times=args.pre_tr_times,
                vaild_label=0,
            )

            box.train(
                X=train_loader,
                X2=vaild_loader,
                lr=args.lr,
                pre_file=best_file,
                best_loss=args.best_loss
            )

            test_out = box.test(
                X=test_loader,
                epoch_pth=best_file,
                test_batch=7)

            ist, isp = data_loader1.time_devide(case_nums=7, traindata="test")
            access = evaluate.Evalulate(test_label, test_out, ist, isp, case_num=7)
            print("MDPE    MDAPE    RMSE\r")
            for i in range(4):
                print("%.2f     %.2f     %.2f" % access.loss(i))

        elif mode == "test":
            test_loader, test_label = data_loader1.test_data_loader(
                tw=args.tw,
                batch=args.test_batch,
                batch_size=100,
                timestep=1)
            pre_tr_times = 5
            pre_file = f'/home/user02/HYK/bis_transformer/output/base/epoch{pre_tr_times}.pth'
            test_out = box.test(
                X=test_loader,
                epoch_pth=best_file,
                test_batch=7)

            plt.grid(True)
            plt.autoscale(axis='x', tight=True)
            for i in range(3):
                plt.figure()
                plt.plot(test_label[i])
                plt.plot(test_out[i])
                plt.savefig(f'/home/user02/HYK/bis/attention/output_picture/case{i}.png')
                plt.show()

