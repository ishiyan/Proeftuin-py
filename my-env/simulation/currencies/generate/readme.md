# Currencies

## Data sources

Sources for data used in `currencies*.csv` file:

- [iso](https://www.currency-iso.org/en/home/tables/table-a1.html)
- [oanda](https://www1.oanda.com/currency/iso-currency-codes/)
- [wikipedia](https://en.wikipedia.org/wiki/ISO_4217)
- [wikipedia cryptocurrencies](https://en.wikipedia.org/wiki/List_of_cryptocurrencies)

## Generating source code

- Comment or uncomment currencies in the `currencies.csv` file.
- Execute `go run generate_currencies_python` in this folder.
- Check the generated `currencies.py` file.
