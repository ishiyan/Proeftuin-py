
# FinRock

Reinforcement Learning package for Finance
[github](https://github.com/pythonlessons/FinRock)

## Install requirements

- numpy
- pandas
- matplotlib
- rockrl==0.4.4
- tensorflow==2.10.0

```bash
pip install -r requirements.txt
pip install pygame
```

## Execute

- Create sinusoid data.
  Open `Create sinusoid data.ipynb` and run it.
- Play the data in the environment.
  Open `Play sinusoid data.ipynb` and run it.
- Train RL (PPO) agent on discrete actions.
  Open `Training ppo sinusoid discrete.ipynb` and run it.
- Test trained agent.
  - Open `Testing ppo sinusoid discrete.ipynb`
  - Change path to the saved model
  - Run it
- Train RL (PPO) agent on continuous actions.
  Open `Training ppo sinusoid continuous.ipynb` and run it.
- Test trained agent.
  - Open `Testing ppo sinusoid continuous.ipynb`
  - Change path to the saved model
  - Run it

## Links to YouTube videos:

- [Introduction to FinRock package](https://youtu.be/xU_YJB7vilA)
- [Complete Trading Simulation Backbone](https://youtu.be/1z5geob8Yho)
- [Training RL agent on Sinusoid data](https://youtu.be/JkA4BuYvWyE)
- [Included metrics and indicators into environment](https://youtu.be/bGpBEnKzIdo)
