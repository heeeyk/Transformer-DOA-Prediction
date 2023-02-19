import argparse


class Params:
    def __init__(self):
        self.x = 1

    @staticmethod
    def lstm_params():
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_dim',     default=2,              type=int)
        parser.add_argument('--memory_cell',     default=8,              type=int)
        parser.add_argument('--body_dim',   default=4,              type=int)
        parser.add_argument('--n',         default=16,              type=int)

        args = parser.parse_args()
        return args

    @staticmethod
    def trainparam():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', default="baseline", type=str)
        parser.add_argument('--tw',           default=180,   type=int)
        parser.add_argument('--train_batch',  default=100,   type=int)
        parser.add_argument('--vaild_batch',  default=30,    type=int)
        parser.add_argument('--test_batch',   default=76,    type=int)
        parser.add_argument('--batch_size',   default=64,    type=int)
        parser.add_argument('--train_epoch',  default=50,    type=int)
        parser.add_argument('--lr',           default=3e-4,  type=float)
        parser.add_argument('--pre_train',    default=False,  type=bool)
        parser.add_argument('--pre_tr_times', default=0,     type=int)
        parser.add_argument('--device',       default=3,     type=int)
        parser.add_argument('--best_loss',    default=80000, type=int)

        args = parser.parse_args()

        # 预训练文件路径
        pre_file = f'/home/user02/HYK/bis_transformer/output/{args.model_name}/epoch{args.pre_tr_times}.pth'
        model_file = f'/home/user02/HYK/bis_transformer/output/{args.model_name}/model/epoch{args.pre_tr_times}.pth'
        best_file = f'/home/user02/HYK/bis_transformer/output/{args.model_name}/model/best_epoch.pth'
        # 保存文件路径
        save_file = f'/home/user02/HYK/bis_transformer/output/{args.model_name}/epoch{args.pre_tr_times}.pth'

        parser.add_argument('--pre_file', default=pre_file, type=str)
        parser.add_argument('--model_file', default=model_file, type=str)
        parser.add_argument('--best_file', default=best_file, type=str)
        parser.add_argument('--save_file', default=save_file, type=str)
        args = parser.parse_args()

        return args
