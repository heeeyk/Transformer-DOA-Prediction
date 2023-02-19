import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class Evalulate:
    def __init__(self, x, y, istart, istop, case_num):
        # x:预测结果 y:label len:样本长度
        self.len = case_num
        self.x = x[:self.len]
        self.y = y[:self.len]
        self.MSE = nn.MSELoss()
        self.istrat = istart
        self.istop = istop

        for i in range(case_num):
            self.x[i] = torch.tensor(self.x[i])
            self.y[i] = torch.tensor(self.y[i])
            if len(x[i]) < len(y[i]):
                self.y[i] = self.y[i][:len(x[i])]
            else:
                self.x[i] = self.x[i][:len(y[i])]

    def rateplot(self, i):
        r = self.x[i] / self.y[i]
        plt.plot(r)

    def ratelist(self):
        r = [0] * self.len
        for i in range(self.len):
            r[i] = self.x[i] / self.y[i]
            plt.plot(r[i])
        plt.show()

    def loss(self, period=0):
        # 0, 1, 2, 3 全阶段,引导期，维持期，复苏期
        t1, t2 = 0, 0

        PE = [0] * self.len
        MSE = [0] * self.len

        MDPE, MDAPE, RMSE = [0] * self.len, [0] * self.len, [0] * self.len
        for i in range(self.len):
            if period == 0:
                t1 = self.istrat[i]
                t2 = -1
            elif period == 1:
                t1 = self.istrat[i]
                t2 = self.istrat[i] + 600
            elif period == 2:
                t1 = self.istrat[i] + 600
                t2 = self.istop[i]
            elif period == 3:
                t1 = self.istop[i]
                t2 = -1
            PE[i] = ((self.x[i][t1:t2] - self.y[i][t1:t2]) / self.x[i][t1:t2])
            MSE[i] = self.MSE(self.x[i][t1:t2].unsqueeze(-1), self.y[i][t1:t2].unsqueeze(-1))
            MDPE[i], MDAPE[i], RMSE[i] = self.estimate(PE=PE[i], MSE=MSE[i])

        return MDPE, MDAPE, RMSE

    @staticmethod
    def estimate(PE, MSE):
        """
        :param PE: 每个样本的bis误差（预测bis-真实bis），输入格式：list([样本误差])
        :param MSE: 每个样本的loss， 输入格式：list([样本loss])
        :return: MDPE:误差中位数， MDAPE:绝对误差中位数， RMSE:均方差
        """
        MDPE = np.median(PE) * 100
        MDAPE = np.median(np.abs(PE)) * 100
        RMSE = np.sqrt(MSE)
        return MDPE, MDAPE, RMSE


if __name__ == "__main__":
    x1 = [torch.ones(3000)*10, torch.ones(3000)*110, torch.ones(3000)*56]
    y1 = [torch.ones(3001), torch.ones(3001), torch.ones(3001)]
    e = Evalulate(x1, y1)
    MDPE, MDAPE, RMSE = e.loss()
    e.ratelist()




