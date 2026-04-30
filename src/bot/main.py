from trainer import NFSPTrainer

STATE_DIM = 20
NUM_ACTIONS = 8

trainer = NFSPTrainer(STATE_DIM, NUM_ACTIONS)
trainer.train(episodes=25000)