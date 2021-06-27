from dataclasses import dataclass
from itertools import accumulate
from numpy.random import default_rng
import matplotlib.pyplot as plt

rng = default_rng()


@dataclass
class Bandit:
    params: list[tuple]

    def __getitem__(self, i):
        μ, σ = self.params[i]
        return rng.normal(μ, σ)

    def __len__(self):
        return len(self.params)


def greedy_strategy(bandit: Bandit,
                    runs: int,
                    ε: float = 0.0,
                    initial_explore=False) -> tuple[list[float], list[float]]:
    arms = len(bandit)
    action_values = [0] * arms
    counts = [0] * arms
    rewards = list()

    if initial_explore:
        # Do one full exploration
        action_values = [bandit[i] for i in range(arms)]
        counts = [1] * arms
        rewards = action_values

    for _ in range(runs - len(rewards)):
        uni = rng.uniform()
        if uni < 1 - ε:
            mx = max(action_values)
            argmax = action_values.index(mx)
        else:
            argmax = rng.integers(arms)
        reward = bandit[argmax]
        rewards.append(reward)
        counts[argmax] += 1
        action_values[argmax] += (1 / counts[argmax]) * (reward - action_values[argmax])
    return rewards, action_values


def best_reward_perc(bandit: Bandit,
                     actual_rewards: list[float]) -> list[float]:
    best_reward = max([μ for μ, _ in bandit.params])
    rewards_cum = accumulate(actual_rewards)
    return [r / (idx * best_reward) for idx, r in enumerate(rewards_cum, 1)]


if __name__ == "__main__":
    PARAMS = [(10, 1), (10.1, 1), (10.25, 1), (10.5, 1)]
    EPSILONS = (0.0, 0.01, 0.1)
    RUNS = 10000

    bandit = Bandit(params=PARAMS)
    print(bandit, "\n")

    for ε in EPSILONS:
        rewards, found = greedy_strategy(bandit, RUNS, ε)
        print(f"ε = {ε}")
        print(f"Total reward: {sum(rewards):.3f}")
        print(f"Action values: {', '.join([f'{v:.3f}' for v in found])}\n")
        plt.plot(best_reward_perc(bandit, rewards), label=f"ε = {ε}")
    plt.legend(loc="lower right")
    plt.show()
