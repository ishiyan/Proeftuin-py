from itertools import groupby
from typing import List

from .roundtrip import Roundtrip

class RoundtripPerformance:
    """Calculates the roundtrip performance statistics."""

    def __init__(self):        
        self.roundtrips: List[Roundtrip] = []

    def add_roundtrip(self, roundtrip: Roundtrip):
        """Adds a roundtrip to the performance tracker."""
        self.roundtrips.append(roundtrip)

    @property
    def total_count(self) -> int:
        """Returns the total number of roundtrips."""
        return len(self.roundtrips)

    @property
    def gross_winning_count(self) -> int:
        """Returns the number of gross winning roundtrips."""
        return len([r for r in self.roundtrips if r.gross_pnl > 0])

    @property
    def gross_loosing_count(self) -> int:
        """Returns the number of gross loosing roundtrips."""
        return len([r for r in self.roundtrips if r.gross_pnl < 0])

    @property
    def net_winning_count(self) -> int:
        """Returns the number of net winning roundtrips."""
        return len([r for r in self.roundtrips if r.net_pnl > 0])

    @property
    def net_loosing_count(self) -> int:
        """Returns the number of net loosing roundtrips."""
        return len([r for r in self.roundtrips if r.net_pnl < 0])
    
    @property
    def gross_winning_percentage(self) -> float:
        """Returns the percentage of gross winning roundtrips."""
        return 100.0 * self.gross_winning_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def gross_loosing_percentage(self) -> float:
        """Returns the percentage of gross loosing roundtrips."""
        return 100.0 * self.gross_loosing_count / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def net_winning_percentage(self) -> float:
        """Returns the percentage of net winning roundtrips."""
        return 100.0 * self.net_winning_count / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def net_loosing_percentage(self) -> float:
        """Returns the percentage of net loosing roundtrips."""
        return 100.0 * self.net_loosing_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def total_gross_pnl(self) -> float:
        """Returns the total gross profit and loss of all roundtrips."""
        return sum([r.gross_pnl for r in self.roundtrips])

    @property
    def total_net_pnl(self) -> float:
        """Returns the total net profit and loss of all roundtrips (taking commission into account)."""
        return sum([r.net_pnl for r in self.roundtrips])

    @property
    def winning_gross_pnl(self) -> float:
        """Returns the total gross profit of all winning roundtrips."""
        return sum([r.gross_pnl for r in self.roundtrips if r.gross_pnl > 0])
    
    @property
    def loosing_gross_pnl(self) -> float:
        """Returns the total gross loss of all loosing roundtrips."""
        return sum([r.gross_pnl for r in self.roundtrips if r.gross_pnl < 0])
    
    @property
    def winning_net_pnl(self) -> float:
        """Returns the total net profit of all winning roundtrips (taking commission into account)."""
        return sum([r.net_pnl for r in self.roundtrips if r.net_pnl > 0])
    
    @property
    def loosing_net_pnl(self) -> float:
        """Returns the total net loss of all loosing roundtrips (taking commission into account)."""
        return sum([r.net_pnl for r in self.roundtrips if r.net_pnl < 0])
    
    @property
    def average_gross_pnl(self) -> float:
        """Returns the average gross profit and loss of a roundtrip."""
        return self.total_gross_pnl / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def average_net_pnl(self) -> float:
        """Returns the average net profit and loss of a roundtrip (taking commission into account)."""
        return self.total_net_pnl / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def average_winning_gross_pnl(self) -> float:
        """Returns the average gross profit of a winning roundtrip."""
        return self.winning_gross_pnl / self.gross_winning_count if self.gross_winning_count > 0 else 0.0
    
    @property
    def average_loosing_gross_pnl(self) -> float:
        """Returns the average gross loss of a loosing roundtrip."""
        return self.loosing_gross_pnl / self.gross_loosing_count if self.gross_loosing_count > 0 else 0.0
    
    @property
    def average_winning_net_pnl(self) -> float:
        """Returns the average net profit of a winning roundtrip (taking commission into account)."""
        return self.winning_net_pnl / self.net_winning_count if self.net_winning_count > 0 else 0.0
    
    @property
    def average_loosing_net_pnl(self) -> float:
        """Returns the average net loss of a loosing roundtrip (taking commission into account)."""
        return self.loosing_net_pnl / self.net_loosing_count if self.net_loosing_count > 0 else 0.0
    
    @property
    def average_gross_winning_loosing_percentage(self) -> float:
        """Returns the average percentage of a gross winning or loosing roundtrip."""
        w = self.winning_gross_pnl / self.gross_winning_count if self.gross_winning_count > 0 else 0.0
        l = -self.loosing_gross_pnl / self.gross_loosing_count if self.gross_loosing_count > 0 else 0.0
        return 100.0 * w / l if l != 0.0 else 0.0 # very big number if l == 0.0

    @property
    def average_net_winning_loosing_percentage(self) -> float:
        """Returns the average percentage of a net winning or loosing roundtrip (taking commission into account)."""
        w = self.winning_net_pnl / self.net_winning_count if self.net_winning_count > 0 else 0.0
        l = -self.loosing_net_pnl / self.net_loosing_count if self.net_loosing_count > 0 else 0.0
        return 100.0 * w / l if l != 0.0 else 0.0 # very big number if l == 0.0
    
    @property
    def gross_profit_pnl_percentage(self) -> float:
        """Returns the PnL percentage of the gross winning roundtrips over the gross loosing roundtrips."""
        return 100.0 * self.winning_gross_pnl / self.total_gross_pnl if self.total_gross_pnl != 0.0 else 0.0

    @property
    def net_profit_pnl_percentage(self) -> float:
        """Returns the PnL percentage of the net winning roundtrips over the net loosing roundtrips."""
        return 100.0 * self.winning_net_pnl / self.total_net_pnl if self.total_net_pnl != 0.0 else 0.0

    @property
    def average_duration_seconds(self) -> float:
        """Returns the average duration of a roundtrip in seconds."""
        return sum([r.duration.total_seconds() for r in self.roundtrips]) / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def average_gross_winning_duration_seconds(self) -> float:
        """Returns the average duration of a gross winning roundtrip in seconds."""
        return sum([r.duration.total_seconds() for r in self.roundtrips if r.gross_pnl > 0]) / self.gross_winning_count if self.gross_winning_count > 0 else 0.0
    
    @property
    def average_gross_loosing_duration_seconds(self) -> float:
        """Returns the average duration of a gross loosing roundtrip in seconds."""
        return sum([r.duration.total_seconds() for r in self.roundtrips if r.gross_pnl < 0]) / self.gross_loosing_count if self.self.gross_loosing_count > 0 else 0.0
    
    @property
    def average_net_winning_duration_seconds(self) -> float:
        """Returns the average duration of a net winning roundtrip in seconds."""
        return sum([r.duration.total_seconds() for r in self.roundtrips if r.net_pnl > 0]) / self.net_winning_count if self.net_winning_count > 0 else 0.0
    
    @property
    def average_net_loosing_duration_seconds(self) -> float:
        """Returns the average duration of a net loosing roundtrip in seconds."""
        return sum([r.duration.total_seconds() for r in self.roundtrips if r.net_pnl < 0]) / self.net_loosing_count if self.net_loosing_count > 0 else 0.0
    
    @property
    def minimum_duration_seconds(self) -> float:
        """Returns the minimum duration of a roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips])
    
    @property
    def maximum_duration_seconds(self) -> float:
        """Returns the maximum duration of a roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips])
    
    @property
    def minimum_gross_winning_duration_seconds(self) -> float:
        """Returns the minimum duration of a gross winning roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips if r.gross_pnl > 0])
    
    @property
    def maximum_gross_winning_duration_seconds(self) -> float:
        """Returns the maximum duration of a gross winning roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips if r.gross_pnl > 0])
    
    @property
    def minimum_gross_loosing_duration_seconds(self) -> float:
        """Returns the minimum duration of a gross loosing roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips if r.gross_pnl < 0])
    
    @property
    def maximum_gross_loosing_duration_seconds(self) -> float:
        """Returns the maximum duration of a gross loosing roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips if r.gross_pnl < 0])
    
    @property
    def minimum_net_winning_duration_seconds(self) -> float:
        """Returns the minimum duration of a net winning roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips if r.net_pnl > 0])
    
    @property
    def maximum_net_winning_duration_seconds(self) -> float:
        """Returns the maximum duration of a net winning roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips if r.net_pnl > 0])
    
    @property
    def minimum_net_loosing_duration_seconds(self) -> float:
        """Returns the minimum duration of a net loosing roundtrip in seconds."""
        return min([r.duration.total_seconds() for r in self.roundtrips if r.net_pnl < 0])
    
    @property
    def maximum_net_loosing_duration_seconds(self) -> float:
        """Returns the maximum duration of a net loosing roundtrip in seconds."""
        return max([r.duration.total_seconds() for r in self.roundtrips if r.net_pnl < 0])
    
    @property
    def average_maximum_adverse_excursion(self) -> float:
        """Returns the average maximum adverse excursion of all roundtrips in percentage."""
        return sum([r.maximum_adverse_excursion for r in self.roundtrips]) / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def average_maximum_favorable_excursion(self) -> float:
        """Returns the average maximum favorable excursion of all roundtrips in percentage."""
        return sum([r.maximum_favorable_excursion for r in self.roundtrips]) / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def average_entry_efficiency(self) -> float:
        """Returns the average entry efficiency of all roundtrips in percentage."""
        return sum([r.entry_efficiency for r in self.roundtrips]) / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def average_exit_efficiency(self) -> float:
        """Returns the average exit efficiency of all roundtrips in percentage."""
        return sum([r.exit_efficiency for r in self.roundtrips]) / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def average_total_efficiency(self) -> float:
        """Returns the average total efficiency of all roundtrips in percentage."""
        return sum([r.total_efficiency for r in self.roundtrips]) / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def max_consecutive_gross_winners(self) -> int:
        """Returns the maximum number of consecutive gross winners."""
        return max([len(list(g)) for k, g in groupby([r.gross_pnl > 0 for r in self.roundtrips]) if k], default=0)
    
    @property
    def max_consecutive_gross_loosers(self) -> int:
        """Returns the maximum number of consecutive gross looser."""
        return max([len(list(g)) for k, g in groupby([r.gross_pnl < 0 for r in self.roundtrips]) if k], default=0)
    
    @property
    def max_consecutive_net_winners(self) -> int:
        """Returns the maximum number of consecutive net winners."""
        return max([len(list(g)) for k, g in groupby([r.net_pnl > 0 for r in self.roundtrips]) if k], default=0)
    
    @property
    def max_consecutive_net_loosers(self) -> int:
        """Returns the maximum number of consecutive net looser."""
        return max([len(list(g)) for k, g in groupby([r.net_pnl < 0 for r in self.roundtrips]) if k], default=0)
    
    
    