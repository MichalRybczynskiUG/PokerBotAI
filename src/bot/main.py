from trainer import NFSPTrainer
import os

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("models", exist_ok=True)

STATE_DIM = 302
NUM_ACTIONS = 3

trainer = NFSPTrainer(STATE_DIM, NUM_ACTIONS)

trainer.train(episodes=200000, eval_every=25000)
