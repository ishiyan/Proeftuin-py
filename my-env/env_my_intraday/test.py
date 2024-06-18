    
from providers import BarToTradeConverter


#p, o = _zigzag_open_high_low_close(100.2, 100.4, 100.1, 100.3, 0.1)
#p, o = _zigzag_open_high_low_close(0.2, 0.4, 0.1, 0.3, 0.1)
#print(0.3/0.05)

spread = 0.001
p, o = BarToTradeConverter._zigzag_open_high_low_close(
                    100.2, 100.4, 100.1, 100.3, spread)
for i in range(len(p)):
    print(i, p[i], o[i])

# 0.4   4
# 0.3   5
# 0.2  6
# 0.1-> 11
# 0.05-> 18
# 0.01-> 73
# 0.005-> 144
# 0.001-> 703

