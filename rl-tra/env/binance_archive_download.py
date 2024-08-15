from environment import BinanceMonthlyTradesProvider
from environment import BinanceMonthlyKlines1mToTradesProvider

def download_and_convert_monthly_trades(symbol, year, month):
    data_dir = 'data/binance_monthly_trades'
    file = f'{data_dir}/{symbol}-trades-{year:04}-{month:02}.zip'
    BinanceMonthlyTradesProvider.download_month_archive(symbol, year, month, file, 'spot')
    BinanceMonthlyTradesProvider.convert_month_archive(file)

def download_and_convert_monthly_klines(symbol, year, month):
    data_dir = 'data/binance_monthly_klines'
    file = f'{data_dir}/{symbol}-1m-{year:04}-{month:02}.zip'
    BinanceMonthlyKlines1mToTradesProvider.download_month_archive(symbol, year, month, file, 'spot')
    BinanceMonthlyKlines1mToTradesProvider.convert_month_archive(file)

#symbol = 'BTCEUR'
#symbol = 'BTCUSDT'
symbol = 'ETHUSDT'

#for year in range(2021, 2024):
#    for month in range(1, 13):
#        download_and_convert_monthly_trades(symbol, year, month)
#for month in range(1, 7):
#    download_and_convert_monthly_trades(symbol, 2024, month)

#for year in range(2018, 2024):
#    for month in range(1, 13):
#        download_and_convert_monthly_klines(symbol, year, month)
#for month in range(1, 7):
#    download_and_convert_monthly_klines(symbol, 2024, month)

#BinanceMonthlyTradesProvider.convert_month_archive('data/binance_monthly_trades/ETHUSDT-trades-2024-05.zip')