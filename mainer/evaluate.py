import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class Evalulate:
    def __init__(self, x, y, istart, istop, case_num):
        # x:预测结果 y:label len:样本长度
        self.len = case_num
        self.case_num = case_num
        self.x = x[:self.len]
        self.y = y[:self.len]
        self.MSE = nn.MSELoss()
        self.MAE = nn.L1Loss()
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
        MAE = [0] * self.len

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
            MAE[i] = self.MAE(self.x[i][t1:t2].unsqueeze(-1), self.y[i][t1:t2].unsqueeze(-1))
            MDPE[i], MDAPE[i], RMSE[i] = self.estimate(PE=PE[i], MSE=MSE[i])

        out = {"MDPE": MDPE,
               "MDAPE": MDAPE,
               "RMSE": RMSE,
               "MAE": MAE,
               "meanMDPE": np.mean(MDPE),
               "meanMDAPE": np.mean(MDAPE),
               "meanRMSE": np.mean(RMSE),
               "meanMAE": np.mean(MAE),
               "SD": [np.std(MDPE), np.std(MDAPE), np.std(RMSE), np.std(MAE)],
               }
        return out

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

    def test_error(self, label_num):
        t_error = []
        out, label = [], []
        for i in range(self.case_num):
            out.extend(self.x[i])
            label.extend(self.y[i])

        for i in range(len(t_error)):
            out[i] = int(out[i])
            label[i] = int(label[i])
        out = np.asarray(out)
        label = np.asarray(label)

        index = np.argsort(label)
        label = label[index]
        out = out[index]
        # t_error = out - label

        """
            label,out: 排好序的向量
            t_error:每个样本点（80万个）的误差
            label_error:从0到100，label的误差均值
        """
        for i in range(len(label)):
            label[i] = int(label[i])
        j = 0
        label_error = list(np.zeros(100))

        for i in range(100):
            label_error[i] = []
            while label[j] == i:
                # label_num[i] += 1
                label_error[i].append(out[j]-label[j])
                j += 1
                if j == len(label)-10:
                    break
            label_error[i] = np.abs(np.mean(label_error[i]))

        """
            误差图
        """
        plt.autoscale(axis='x', tight=True)
        plt.bar(list(range(100)), label_error)
        plt.xlabel("bis index")
        plt.ylabel("label nums")
        plt.show()

        """
            相关性计算
        """
        a = np.asarray(label_num)
        b = np.asarray(label_error)
        for i in range(100):
            if b[i] < 0:
                b[i] = -b[i]
        plt.subplot(2, 1, 1)
        plt.title(f"Pearson correlation: -{np.corrcoef(a, b)[0, 1]}")
        plt.bar(range(100), a, color='lightskyblue')
        plt.ylabel("sample nums")
        plt.subplot(2, 1, 2)
        plt.bar(range(100), b, color='lightcoral')
        plt.xlabel("label space")
        plt.ylabel("test error")
        plt.show()
        """
            卷积
        """
        from scipy.ndimage import convolve1d
        p = a/len(label)
        lds_kernel_window = self.get_lds_kernel_window(kernel='gaussian', ks=10, sigma=8)

        eff_label_dist = convolve1d(p, weights=lds_kernel_window, mode='constant')
        cor = np.corrcoef(eff_label_dist, b)[0, 1]
        plt.bar(range(100), eff_label_dist, color='lightcoral')
        plt.show()
        return t_error, label_error

    @staticmethod
    def get_lds_kernel_window(kernel, ks, sigma):
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal.windows import triang
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks)
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

        return kernel_window



if __name__ == "__main__":
    x1 = [torch.ones(3000)*10, torch.ones(3000)*110, torch.ones(3000)*56]
    y1 = [torch.ones(3001), torch.ones(3001), torch.ones(3001)]
    e = Evalulate(x1, y1)
    MDPE, MDAPE, RMSE = e.loss()
    e.ratelist()




