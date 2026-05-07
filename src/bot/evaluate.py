from evaluate_metrics import load_model, evaluate_vs_random, evaluate_model_vs_model

STATE_DIM = 414
NUM_ACTIONS = 8

model_50k = load_model("models/stare/model_250000.pt",STATE_DIM,NUM_ACTIONS)
model_250k = load_model("models/stare/model_250000_step20.pt", STATE_DIM, NUM_ACTIONS)

#evaluate_vs_random(model_50k, episodes=3000)

evaluate_model_vs_model(model_50k, model_250k, episodes=10000)

