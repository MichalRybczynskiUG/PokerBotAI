from trainer import NFSPTrainer

STATE_DIM = 20
NUM_ACTIONS = 4

trainer = NFSPTrainer(STATE_DIM, NUM_ACTIONS)
trainer.train(episodes=5000)