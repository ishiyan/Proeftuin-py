# The sources for data:
# - [ISO 4217](https://www.currency-iso.org/en/home/tables/table-a1.html)
# - [OANDA ISO currency codes](https://www1.oanda.com/currency/iso-currency-codes/)
# - [Wikipedia ISO 4217](https://en.wikipedia.org/wiki/ISO_4217)
# - [Wikipedia list of cryptocurrencies](https://en.wikipedia.org/wiki/List_of_cryptocurrencies)

from .currency import Currency, _currencies

class SubscriptableType(type):
    def __getitem__(cls, key):
        return cls.__getitem__(key)
    
class Currencies(metaclass=SubscriptableType):
    """The currencies. Sources for data:
    - [ISO 4217](https://www.currency-iso.org/en/home/tables/table-a1.html)
    - [OANDA ISO currency codes](https://www1.oanda.com/currency/iso-currency-codes/)
    - [Wikipedia ISO 4217](https://en.wikipedia.org/wiki/ISO_4217)
    - [Wikipedia list of cryptocurrencies](https://en.wikipedia.org/wiki/List_of_cryptocurrencies)
    """
    def __new__(cls, *args, **kwargs):
        raise TypeError('Currencies is a static class and cannot be instantiated')

    @classmethod
    def __getitem__(cls, key: str) -> Currency:
        """Gets a currency by its symbol."""
        try:
            return _currencies[key]
        except KeyError:
            raise KeyError(f'Currency code {key} is not defined')

    XXX = Currency('XXX', 2, '', 'No currency')
    """No currency, used to denote a transaction involving no currency"""

    XAG = Currency('XAG', 5, '', 'Silver (one troy ounce)')
    """Silver (one troy ounce)"""
    XAU = Currency('XAU', 5, '', 'Gold (one troy ounce)')
    """Gold (one troy ounce)"""
    XPD = Currency('XPD', 5, '', 'Palladium (one troy ounce)')
    """Palladium (one troy ounce)"""
    XPT = Currency('XPT', 5, '', 'Platinum (one troy ounce)')
    """Platinum (one troy ounce)"""

    BTC = Currency('BTC', 8, '₿', 'Bitcoin')
    """Bitcoin (₿), cryptocurrency"""
    BCH = Currency('BCH', 8, '', 'Bitcoin Cash')
    """Bitcoin Cash, cryptocurrency"""
    ETH = Currency('ETH', 18, 'Ξ', 'Ethereum')
    """Ethereum (Ξ), cryptocurrency"""
    ETC = Currency('ETC', 18, '', 'Ethereum Classic')
    """Ethereum Classic, cryptocurrency"""
    USDT = Currency('USDT', 8, '', 'Tether')
    """Tether, cryptocurrency"""
    XRP = Currency('XRP', 6, '', 'Ripple')
    """Ripple, cryptocurrency"""
    NEO = Currency('NEO', 8, '', 'NEO')
    """NEO, cryptocurrency"""
    LTC = Currency('LTC', 8, 'Ł', 'Litecoin')
    """Litecoin (Ł), cryptocurrency"""
    VTC = Currency('VTC', 8, '', 'Vertcoin')
    """Vertcoin, cryptocurrency"""
    XLM = Currency('XLM', 8, '', 'Stellar Lumens')
    """Stellar Lumens, cryptocurrency"""
    XMR = Currency('XMR', 12, '', 'Monero')
    """Monero, cryptocurrency"""
    XTZ = Currency('XTZ', 6, 'ꜩ', 'Tez')
    """Tez, cryptocurrency"""
    DSH = Currency('DSH', 8, '', 'Dash')
    """Dash, cryptocurrency"""
    ZEC = Currency('ZEC', 8, '', 'Zcash')
    """Zcash, cryptocurrency"""
    EOS = Currency('EOS', 4, 'EOS.IO', 'Zcash')
    """EOS.IO, cryptocurrency"""
    LINK = Currency('LINK', 8, '', 'Chainlink')
    """Chainlink, cryptocurrency"""
    ATOM = Currency('ATOM', 8, '', 'Cosmos')
    """Cosmos, cryptocurrency"""
    DAI = Currency('DAI', 8, '', 'Dai')
    """Dai, cryptocurrency"""

    EUR = Currency('EUR', 2, '€', 'Euro')
    """Euro (€)"""
    EUX = Currency('EUX', 0, '', 'Euro cent')
    """Euro cent, 1⁄100 of EUR"""
    USD = Currency('USD', 2, '$', 'US Dollar')
    """US Dollar ($)"""
    USX = Currency('USX', 0, '¢', 'US cent')
    """US cent, 1⁄100 of USD"""
    GBP = Currency('GBP', 2, '£', 'Pound sterling')
    """Pound sterling (£)"""
    GBX = Currency('GBX', 0, 'p', 'Penny sterling')
    """Penny sterling (p), 1⁄100 of a pound, but historically was 1⁄240 of a pound (old penny sterling)"""
    ZAR = Currency('ZAR', 2, 'R', 'South African rand')
    """South African rand (R)"""
    ZAC = Currency('ZAC', 0, 'c', 'South African cent')
    """South African cent, 1⁄100 of ZAR"""
    CHF = Currency('CHF', 2, 'Fr', 'Swiss franc')
    """Swiss franc (Fr)"""
    CAD = Currency('CAD', 2, 'C$', 'Canadian dollar')
    """Canadian dollar (C$)"""
    AUD = Currency('AUD', 2, 'A$', 'Australian dollar')
    """Australian dollar (A$)"""
    NZD = Currency('NZD', 2, '$', 'New Zeeland dollar')
    """New Zeeland dollar ($)"""
    DKK = Currency('DKK', 2, 'kr', 'Danish krone')
    """Danish krone (kr)"""
    SEK = Currency('SEK', 2, 'kr', 'Swedish krona')
    """Swedish krona (kr)"""
    NOK = Currency('NOK', 2, 'kr', 'Norwegian krone')
    """Norwegian krone (kr)"""
    ISK = Currency('ISK', 0, 'Íkr', 'Icelandic krona')
    """Icelandic krona (kr, Íkr)"""
    PLN = Currency('PLN', 2, 'zł', 'Poland zloty')
    """Poland zloty (zł)"""
    HUF = Currency('HUF', 2, 'Ft', 'Hungary forint')
    """Hungary forint (Ft)"""
    JPY = Currency('JPY', 0, '¥', 'Japanese Yen')
    """Japanese Yen (¥)"""
    SGD = Currency('SGD', 2, 'S$', 'Singapore dollar')
    """Singapore dollar (S$)"""
    HKD = Currency('HKD', 2, 'HK$', 'Hong Kong dollar')
    """Hong Kong dollar (HK$)"""
    KRW = Currency('KRW', 0, '₩', 'South Korean won')
    """South Korean won (₩)"""
    TWD = Currency('TWD', 2, 'NT$', 'Taiwan new dollar')
    """Taiwan new dollar (NT$)"""
    CNY = Currency('CNY', 2, '¥', 'Chinese onshore yuan renminbi')
    """Chinese onshore yuan renminbi (traded within Mainland China only)"""
    CNH = Currency('CNH', 2, '¥', 'Chinese offshore yuan renminbi')
    """Chinese offshore yuan renminbi (traded outside of Mainland China)"""
    INR = Currency('INR', 2, '₨', 'Indian rupee')
    """Indian rupee (₨, ₹, ৳, रु)"""
