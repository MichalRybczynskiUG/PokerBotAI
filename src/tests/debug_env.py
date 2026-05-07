from src.poker_enviroment.poker_env import PokerEnv
from src.bot.random_bot import RandomBot

def debug_random_vs_random(hands=5):
    print("=== DEBUG RANDOM vs RANDOM ===")

    env = PokerEnv()
    bot = RandomBot()

    for i in range(hands):
        print(f"\n---- HAND {i+1} ----")

        state = env.reset()

        start_p1 = env.initial_stack_p1
        start_p2 = env.initial_stack_p2

        print("START (true):", start_p1, start_p2)

        print("START (after blinds):", env.p1.stack, env.p2.stack)

        done = False
        steps = 0

        while not done:
            current = env.current_player

            if current.all_in:
                state, _, done, _ = env.step(1, None)
                continue

            action, raise_amount = bot.select_action(env)
            state, reward, done, _ = env.step(action, raise_amount)

            steps += 1
            if steps > 50:
                print("too many steps")
                break

        end_p1 = env.p1.stack
        end_p2 = env.p2.stack

        print("END STACKS:", end_p1, end_p2)

        profit_p1 = end_p1 - start_p1
        profit_p2 = end_p2 - start_p2

        print("PROFIT P1:", profit_p1)
        print("PROFIT P2:", profit_p2)

        total_profit = profit_p1 + profit_p2
        print("PROFIT SUM:", total_profit)

        if total_profit != 0:
            print("PROFIT MISMATCH")

        total_chips = end_p1 + end_p2 + env.engine.pot
        expected_total = start_p1 + start_p2

        print("TOTAL CHIPS:", total_chips)

        if total_chips != expected_total:
            print("CHIP LEAK")

        if abs(profit_p1) > 1000 or abs(profit_p2) > 1000:
            print("LARGE POT (ALL-IN?)")


if __name__ == "__main__":
    debug_random_vs_random(10)
