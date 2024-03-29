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
from model.featurefusion import trainer, params_save
import statsmodels.api as sm
import imp


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def lds():
    data_label, label_num = d_box.label_stat(180, "train")
    p = np.asarray(label_num) / len(data_label) + 1e-3
    for i in range(10):
        p = np.append(p, 1e-3)

    lds_kernel_window = evaluate.Evalulate.get_lds_kernel_window(kernel='gaussian', ks=10, sigma=8)
    from scipy.ndimage import convolve1d
    eff_label_dist = convolve1d(p, weights=lds_kernel_window, mode='constant')
    return eff_label_dist

setup_seed(2)
mode = 'test'
args = params_save.Params.trainparam()

if __name__ == "__main__":
    with torch.cuda.device(args.device):
        args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"GPU{args.device} open")
        else:
            print("cpu open")

        box = trainer.Trainer(args)
        d_box = ni_database.Dataloader(
            database_wdir="/HDD_data/HYK/bis/database/ni_dataset",
            time_step=1,
            nums=1,
            tw=args.tw
        )
        
        if mode == "train":

            vaild_loader, train_loader, test_loader, test_label = d_box.load_all(
                512, 3000, 256
            )

            eff_label_dist = lds()

            box.train(
                X=train_loader,
                X2=vaild_loader,
                model_file=args.best_file,
                best_loss=args.best_loss,
                config=args,
                p=eff_label_dist
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
                
            access = evaluate.Evalulate(test_out, test_label, ist, isp, case_num=76)
            print("MDPE    MDAPE    RMSE    MAE\r")
            for i in range(4):
                x = access.loss(i)
                print("%.2f  " % x["meanMDPE"],
                      "%.2f  " % x["meanMDAPE"],
                      "%.2f  " % x["meanRMSE"],
                      "%.2f  " % x["meanMAE"])


        elif mode == "test":

            test_loader, test_label = d_box.pt_load(
                dataset="test",
                batch_size=4,
            )

            test_out = box.test(
                X=test_loader,
                epoch_pth=args.best_file,
                test_batch=30)

            for i in range(10,22):
                plt.figure()
                plt.plot(test_label[i], 'silver', label="ground truth")
                plt.plot(test_out[i], label="ours")
                plt.ylabel("Bispectral index")
                plt.xlabel("Time(sec)")
                plt.legend(loc='upper right')
                # plt.savefig(f'/home/user02/HYK/bis/attention/output_picture/case{i}.png')
                plt.show()




