from trainer import NFSPTrainer
import os

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("models", exist_ok=True)

STATE_DIM = 382
NUM_ACTIONS = 5

trainer = NFSPTrainer(STATE_DIM, NUM_ACTIONS)

trainer.train(episodes=2001000,eval_every=25000)
