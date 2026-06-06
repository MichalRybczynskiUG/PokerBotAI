from evaluate_metrics import load_model, evaluate_vs_random, evaluate_model_vs_model

STATE_DIM = 375
NUM_ACTIONS = 5

model_50k = load_model("models/model_350000.pt", STATE_DIM, NUM_ACTIONS)
model_100k = load_model("models/model_500000.pt", STATE_DIM, NUM_ACTIONS)

#evaluate_vs_random(model_50k, episodes=3000)

evaluate_model_vs_model(model_50k, model_100k, episodes=10000)

