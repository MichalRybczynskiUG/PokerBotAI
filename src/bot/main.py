from trainer import NFSPTrainer
import os

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("models", exist_ok=True)

STATE_DIM = 375
NUM_ACTIONS = 5

trainer = NFSPTrainer(STATE_DIM, NUM_ACTIONS)

trainer.train(episodes=500100)
