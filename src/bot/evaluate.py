from evaluate_metrics import load_model, evaluate_vs_random, evaluate_model_vs_model, analyze_starting_hands, inspect_q, compare_hands
from src.poker_enviroment.poker_env import PokerEnv

STATE_DIM = 222 #375
NUM_ACTIONS = 5

model_1 = load_model("models/model_200000.pt", STATE_DIM, NUM_ACTIONS)
model_2 = load_model("models/model_270000.pt", STATE_DIM, NUM_ACTIONS)

#evaluate_vs_random(model_2, episodes=3000)

evaluate_model_vs_model(model_1, model_2, episodes=10000)

model = load_model(
    "models/model_270000.pt",
    222,
    NUM_ACTIONS
)

env = PokerEnv()
env.reset()

env.board = []

env.p1.hand = ["As", "Ah"]
inspect_q(model, env, env.p1)

env.p1.hand = ["7c", "2d"]
inspect_q(model, env, env.p1)

for checkpoint in [
    "models/model_270000.pt",
]:
    print()
    print("=" * 80)
    print(checkpoint)
    print("=" * 80)

    model = load_model(
        checkpoint,
        STATE_DIM,
        NUM_ACTIONS
    )

    compare_hands(
        model,
        PokerEnv()
    )
