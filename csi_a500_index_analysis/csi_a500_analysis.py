import pandas as pd
import matplotlib.pyplot as plt


class A500Analysis:
    def __init__(self, data_file_name):
        self.df = pd.read_csv(
            data_file_name,
            skipinitialspace=True,
            parse_dates=['Date'])
        self.df.drop(index=0, inplace=True)
        # self.open_list = self.df['Open'].astype(float).tolist()
        # self.high_list = self.df['High'].astype(float).tolist()
        # self.low_list = self.df['Low'].astype(float).tolist()
        # self.close_list = self.df['Close'].astype(float).tolist()
        # self.volume_list = self.df['Volume'].astype(float).tolist()
        # self.turnover_list = self.df['Turnover'].astype(float).tolist()

    def four_percent_dca(self, dca_rounds: int):
        if dca_rounds < 5:
            raise ValueError("Consider splitting the DCA count into at least 5; otherwise, "
                             "the smoothing effect on volatility may not be achieved.")
        if dca_rounds > 100:
            print("What, using the infinite ammo strategy, huh?")

        close_list: list[float] = self.df['Close'].astype(float).tolist()

        n: int = len(close_list)
        upper_percentage: float = 0.8
        # lower_percentage: float = 1.0
        dcs_unit: float = 10000.0
        reward_list: list[float] = [0.0] * n
        iter_float_list: list[float] = [float(i) for i in range(n)]

        history_max_list: list[float] = [1000.0] * n
        history_min_list: list[float] = [1000.0] * n
        history_percentage_list: list[float] = [close_list[0] / 1000.0] * n

        tp_sell_rate: float = 0.25

        for i in range(1, min(n, 500)):
            history_max_list[i] = max(history_max_list[i - 1], close_list[i])
            history_min_list[i] = min(history_min_list[i - 1], close_list[i])
            history_percentage_list[i] = (close_list[i] - history_min_list[i]) / (history_max_list[i] - history_min_list[i])

        if n > 500:
            for i in range(500, n):
                for j in range(i - 500, i + 1):
                    history_max_list[i] = max(history_max_list[i], close_list[j])
                    history_min_list[i] = min(history_min_list[i], close_list[j])
                history_percentage_list[i] = (close_list[i] - history_min_list[i]) / (
                            history_max_list[i] - history_min_list[i])

        for i in range(n):
            dca_rounds_left: int = dca_rounds
            last_share_price: float = 1000.0
            share_num: float = 0.0
            cash: float = dca_rounds * dcs_unit
            for j in range(i, n):
                if history_percentage_list[j] < upper_percentage:
                    if dca_rounds_left == dca_rounds or close_list[j] <= last_share_price * 0.96:
                        if dca_rounds_left > 0:
                            share_num += dcs_unit / close_list[j]
                            dca_rounds_left -= 1
                            last_share_price = close_list[j]
                            cash -= dcs_unit
                else:
                    if share_num > 0.01:
                        cash += tp_sell_rate * share_num * close_list[j]
                        share_num *= 1.0 - tp_sell_rate

            reward_list[i] = (cash + close_list[n - 1] * share_num) / (dca_rounds * dcs_unit)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        ax1.plot(iter_float_list, close_list, label='CSI A500', color='blue')
        ax1.set_ylabel('Val')
        ax1.set_title('CSI A500')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(iter_float_list, reward_list, label='invest reward', color='red')
        ax2.set_xlabel('iter')
        ax2.set_ylabel('Val')
        ax2.set_title('invest reward')
        ax2.grid(True)
        ax2.legend()

        plt.subplots_adjust(hspace=0.3)

        plt.show()

    def midas_touch(self):

        close_list: list[float] = self.df['Close'].astype(float).tolist()

        n: int = len(close_list)
        buy_percentage = 0.88

        dcs_unit: float = 10000.0
        reward_list: list[float] = [0.0] * n
        iter_float_list: list[float] = [float(i) for i in range(n)]
        average_list: list[float] = [0.0] * n
        tp_sell_rate: float = 0.5

        avg_sum = 0.0
        for i in range(min(n, 120)):
            avg_sum += close_list[i]
            average_list[i] = avg_sum / (i + 1.0)

        if n > 120:
            j: int = 0
            for i in range(120, n):
                avg_sum += close_list[i]
                avg_sum -= close_list[j]
                average_list[i] = avg_sum / 120.0

        for i in range(n):
            share_num: float = 0.0
            cash: float = 0.0
            dca_rounds: int = 0
            for j in range(i, n):
                if close_list[j] < average_list[j] * buy_percentage:
                    share_num += dcs_unit / close_list[j]
                    dca_rounds += 1
                elif close_list[j] >= average_list[j]:
                    if share_num > 0.01:
                        cash += tp_sell_rate * share_num * close_list[j]
                        share_num *= 1.0 - tp_sell_rate

            reward_list[i] = (cash + close_list[n - 1] * share_num) / (dca_rounds * dcs_unit)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        ax1.plot(iter_float_list, close_list, label='CSI A500', color='blue')
        ax1.set_ylabel('Val')
        ax1.set_title('CSI A500')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(iter_float_list, reward_list, label='invest reward', color='red')
        ax2.set_xlabel('iter')
        ax2.set_ylabel('Val')
        ax2.set_title('invest reward')
        ax2.grid(True)
        ax2.legend()

        plt.subplots_adjust(hspace=0.3)

        plt.show()


csi_a500_data_file_name = "000510.csv"
analysis = A500Analysis(csi_a500_data_file_name)
# analysis.four_percent_dca(10)
analysis.midas_touch()


