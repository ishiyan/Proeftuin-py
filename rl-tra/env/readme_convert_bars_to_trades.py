from environment import BarToTradeConverter

spread = 0.001

p, o = BarToTradeConverter._zigzag_open_high_low_close(
                    #0.2, 0.4, 0.1, 0.3, spread)
                    100.2, 100.4, 100.1, 100.3, spread)

for i in range(len(p)):
    print(i, p[i], o[i])
