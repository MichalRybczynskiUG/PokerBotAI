from poker_env import *

def main():
    env = PokerEnv()
    obs = env.reset()

    print("Gra w pokera")
    while not env.done:
        print(f"--{STREET_NAMES[obs['street']]}--")
        print(f"Tura gracza: {env.current_player.name}")
        print(f"Twoje karty: {obs['hand']}")
        print(f'Board: {obs["board"]}')
        print(f'Stack: {obs["stack_self"]}')
        print(f'Pot: {obs["pot"]}')
        print(f'Do sprawdzenia: {obs["to_call"] - env.current_player.street_bet}')
        print(f"Dostępne akcje: {[ACTION_NAMES[action] for action in obs['legal_actions']]}")
        action =int(input(f"Wybierz akcję {ACTION_NAMES}:"))
        if action == 2:
            raise_amount = int(input(f"Wprowadź kwotę (min 10):"))
            obs = env.step(action, raise_amount = raise_amount)[0]
        else:
            obs = env.step(action)[0]

    winner_stack = max([player.stack for player in env.players])
    winner = []
    for player in env.players:
        if player.stack == winner_stack:
            winner.append(player.name)
    print(f"Wygrani: {winner}")

if __name__ == "__main__":
    main()