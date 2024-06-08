_mics = {}
_mics_operational = {}

class MIC:
    """MIC is a Market Identifier Code according to ISO 10383.

    See https://www.iso20022.org/market-identifier-codes

    Parameters
    ----------
    mic : str
        The MIC. May be either an operational MIC (exchange) or a segment MIC (market).
        
        A market segment MIC is a MIC which operates in specific instruments.
        Each market segment MIC has a parent MIC which is called the “operating MIC”.
    operational_mic : str
        The operational MIC related to this market segment MIC.
        This is an operatonal MIC if it is not a market segment.
    country_code : str
        The ISO 3166 alpha-2 country code.
    time_zone_seconds : int
        The time zone offset in seconds from UTC.
    """

    def __init__(self, mic: str, operational_mic: str, country_code: str, time_zone_seconds: int) -> None:
        self.mic = mic
        self.operational_mic = operational_mic
        self.country_code = country_code
        self.time_zone_seconds = time_zone_seconds
        if mic in _mics:
            raise ValueError(f'MIC {mic} is already defined')
        _mics[mic] = self
        if mic == operational_mic:
            return
        if operational_mic in _mics_operational:
            _mics_operational[operational_mic].append(self)
        else:
            _mics_operational[operational_mic] = [self]

    def get_segments(self):
        """
        Gets all segments for this operational MIC.
        If this mic is not operational one, the segments
        of it's operational MIC are returned.
        """
        return _mics_operational[self.operational_mic]

    def __eq__(self, other) -> bool:
        """Checks if two MICs are equal.

        Parameters
        ----------
        other : `Any`
            The MIC being compared.

        Returns
        -------
        bool
            Whether the MICs are equal.
        """
        if not isinstance(other, MIC):
            return False
        if self.mic != other.mic:
            return False
        if self.operational_mic != other.operational_mic:
            return False
        if self.country_code != other.country_code:
            return False
        if self.time_zone_seconds != other.time_zone_seconds:
            return False
        return True

    def __ne__(self, other) -> bool:
        """Checks if two MICs are not equal.

        Parameters
        ----------
        other : `Any`
            The MIC being compared.

        Returns
        -------
        bool
            Whether the MICs are not equal.
        """
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.mic)

    def __str__(self):
        return str(self.mic)

    def __repr__(self):
        attributes = ['mic', 'operational_mic', 'country_code', 'time_zone_seconds']
        attr_strings = [f'{attr}={getattr(self, attr)}' for attr in attributes ]
        return 'Instrument(' + ', '.join(attr_strings) + ')'
