from src.poker_enviroment.constants import *

def get_legal_actions(player, to_call, raises_this_round):

    if player.all_in:
        return []

    actions = []

    call_amount = max(
        0,
        to_call - player.street_bet
    )

    if call_amount > 0:
        actions.append(ACTION_FOLD)

    actions.append(ACTION_CALL)

    if (
        player.stack > call_amount
        and raises_this_round < MAX_RAISES
    ):
        actions.append(ACTION_RAISE)

    return actions

class PokerEngine:
    def __init__(self, players):
        self.players = players
        self.pot = 0
        self.to_call = 0
        self.current_player_idx = 0
        self.actions_without_raise = 0

        self.raises_this_round = 0
        self.raise_size = SMALL_BET

    def next_player(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

    def step_betting(self, action):

        p = self.players[self.current_player_idx]

        if p.folded or p.all_in:
            self.next_player()
            return

        legal = get_legal_actions(
            p,
            self.to_call,
            self.raises_this_round
        )

        if action not in legal:
            raise ValueError(
                f"Illegal action: {action}, legal: {legal}"
            )

        if action == ACTION_FOLD:

            p.folded = True

        elif action == ACTION_CALL:

            call_amount = max(
                0,
                self.to_call - p.street_bet
            )

            call_amount = min(
                call_amount,
                p.stack
            )

            p.stack -= call_amount
            p.bet += call_amount
            p.street_bet += call_amount

            self.pot += call_amount

            if p.stack == 0:
                p.all_in = True

            self.actions_without_raise += 1

        elif action == ACTION_RAISE:

            call_amount = max(
                0,
                self.to_call - p.street_bet
            )

            total = call_amount + self.raise_size

            total = min(
                total,
                p.stack
            )

            p.stack -= total
            p.bet += total
            p.street_bet += total

            self.pot += total

            self.to_call = p.street_bet

            self.actions_without_raise = 0
            self.raises_this_round += 1

            if p.stack == 0:
                p.all_in = True

        self.next_player()

    def betting_round_finished(self):

        active = [
            p for p in self.players
            if not p.folded and not p.all_in
        ]

        if len(active) <= 1:
            return True

        if self.actions_without_raise >= len(active):
            return True

        return False

