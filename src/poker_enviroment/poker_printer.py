from poker_env import *

def main():
    env = PokerEnv()
    env.reset()

    print("Gra w pokera")

    while not env.done:
        obs = env.get_debug_observation(env.current_player)

        print(f"--{STREET_NAMES[obs['street']]}--")
        print(f"Tura gracza: {env.current_player.name}")
        print(f"Twoje karty: {obs['hand']}")
        print(f"Board: {obs['board']}")
        print(f"Stack: {obs['stack_self']}")
        print(f"Pot: {obs['pot']}")
        print(f"Do sprawdzenia: {obs['to_call'] - env.current_player.street_bet}")
        print(f"Dostępne akcje: {[ACTION_NAMES[a] for a in obs['legal_actions']]}")

        action = int(input("Wybierz akcję (0 fold, 1 call/check, 2 raise, 3 all-in): "))

        if action == ACTION_RAISE:
            raise_amount = int(input("Kwota raise: "))
            env.step(action, raise_amount=raise_amount)
        else:
            env.step(action)

    winner_stack = max([player.stack for player in env.players])
    winners = [p.name for p in env.players if p.stack == winner_stack]

    print(f"Wygrani: {winners}")


if __name__ == "__main__":
    main()