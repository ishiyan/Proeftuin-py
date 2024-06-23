from __future__ import annotations
from typing import TYPE_CHECKING
from numbers import Real

if TYPE_CHECKING:
    from ..environment import Environment
from ..accounts import Account
from .reward_scheme import RewardScheme

class BalanceReturnReward(RewardScheme):
    """
    A reward scheme that calculates a reward based on the return
    on the account balance.

    A penalty is applied for consecutive steps without position.
    """
    def __init__(self,
                 return_cap: Real = 1.0,
                 return_multiplier: Real = 100.0,
                 consecutive_steps_without_position_threshold: int = 10,
                 consecutive_steps_without_position_penalty_per_day: Real = 0.1,
                 consecutive_steps_without_position_penalty_cap: Real = 10,
                ):
        """
        Initializes the reward scheme.

        Reward is calculated as the return on the account balance.
        ```
        #
        #   reward = min(return, return_cap) * return_multiplier
        #
        #               /-- return_cap
        #              /|
        #             / |
        #            /  |
        #           /|  |
        #          / |  |
        # --------+--+--+---> return
        #         0  1  2
        #
        ```

        For consecutive steps without position, a penalty is applied.                        
        ```
        #   penalty = 0 if days < threshold else min((days - threshold)*penalty_per_day, penalty_cap)
        #
        #                /-- penalty_cap
        #               /
        #              / penalty_per_day
        #             /|
        #  threshold / |
        # ----------+--+--+---> return
        #           0  1
        ```
        Args:
            norm_factor Union[None, Real, str]:
                A normalization factor.
                
                If it is a number, the reward is divided by this number.
                
                If it is a string, the reward is divided by the value of
                the attribute with this name in the last frame.

                If it is `None`, the reward is not normalized.
        """
        super().__init__()
        self.last_balance = {}

        #
        #   reward = min(return, return_cap) * return_multiplier
        #
        #               /-- return_cap
        #              /|
        #             / |
        #            /  |
        #           /|  |
        #          / |  |
        # --------+--+--+---> return
        #         0  1  2
        #
        self.return_cap = return_cap
        self.return_multiplier = return_multiplier

        #
        #   penalty = 0 if days < threshold else min((days - threshold)*penalty_per_day, penalty_cap)
        #
        #                /-- penalty_cap
        #               /
        #              / penalty_per_day
        #             /|
        #  threshold / |
        # ----------+--+--+---> return
        #           0  1
        #
        self.consecutive_steps_without_position = 0
        self.consecutive_steps_without_position_threshold = \
            consecutive_steps_without_position_threshold
        self.consecutive_steps_without_position_penalty_per_day = \
            consecutive_steps_without_position_penalty_per_day
        self.consecutive_steps_without_position_penalty_cap = \
            consecutive_steps_without_position_penalty_cap
    
    def reset(self):
        self.last_balance.clear()
        self.consecutive_steps_without_position = 0
        
    def get_reward(self, env: 'Environment', account: Account) -> float:
        if account not in self.last_balance:
            self.last_balance[account] = account.initial_balance
        new_balance = account.balance
        old_balance = self.last_balance[account]
        self.last_balance[account] = new_balance
        ret = new_balance / old_balance - 1 if old_balance > 0 else 0
        reward = min(abs(ret), self.return_cap) * self.return_multiplier
        if ret < 0:
            reward = -reward

        # If account.is_halted -> reward = ???
        # 'pnl': account.report.net_profit / account.initial_balance,

        if account.has_no_position:
            self.consecutive_steps_without_position += 1
        else:
            self.consecutive_steps_without_position = 0

        if self.consecutive_steps_without_position < \
            self.consecutive_steps_without_position_threshold:
            penalty = 0
        else:
            penalty = min((self.consecutive_steps_without_position - \
                self.consecutive_steps_without_position_threshold) * \
                self.consecutive_steps_without_position_penalty_per_day, \
                self.consecutive_steps_without_position_penalty_cap)
        reward -= penalty

        return reward

# Designing Rewards for Fast Learning
# https://arxiv.org/pdf/2205.15400

# https://medium.com/@BonsaiAI/deep-reinforcement-learning-models-tips-tricks-for-writing-reward-functions-a84fe525e8e0

# https://github.com/stefan-jansen/synthetic-data-for-finance
# https://github.com/stefan-jansen/machine-learning-for-trading/tree/main/22_deep_reinforcement_learning

# https://andreybabynin.medium.com/reinforcement-learning-in-trading-part-1-d0920d69b526
# https://andreybabynin.medium.com/log-periodic-power-law-singularity-model-and-tesla-stock-b775a8b44d26

# https://ai.stackexchange.co   m/questions/10082/suitable-reward-function-for-trading-buy-and-sell-orders

# https://github.com/tatsath/fin-ml/tree/master/Chapter%209%20-%20Reinforcement%20Learning/Case%20Study%201%20-%20Reinforcement%20Learning%20based%20Trading%20Strategy
