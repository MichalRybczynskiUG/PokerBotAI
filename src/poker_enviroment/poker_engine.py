from src.poker_enviroment.constants import *

def get_legal_actions(player, to_call):

    if player.all_in:
        return [ACTION_CALL]
    actions = []

    # Fold
    if to_call > player.street_bet:
        actions.append(ACTION_FOLD)

    # Call
    if to_call > player.street_bet and player.stack > 0:
        actions.append(ACTION_CALL)

    # Check
    if to_call == player.street_bet:
        actions.append(ACTION_CALL)

    # Raise
    if player.stack > (to_call - player.street_bet):
        actions.append(ACTION_RAISE)

    # All-in
    if player.stack > 0:
        actions.append(ACTION_ALL_IN)

    return actions


class PokerEngine:
    def __init__(self, players):
        self.players = players
        self.pot = 0
        self.to_call = 0
        self.current_player_idx = 0
        self.actions_without_raise = 0

    def next_player(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

    def betting_round_finished(self):
        active = [p for p in self.players if not p.folded and not p.all_in]

        if len(active) <= 1:
            return True

        if self.actions_without_raise >= len(active):
            return True

        return False

    def step_betting(self, action, raise_amount=None):
        p = self.players[self.current_player_idx]

        if p.folded or p.all_in:
            self.next_player()
            return

        legal = get_legal_actions(p, self.to_call)
        if action not in legal:
            raise ValueError("Illegal action")

        # --- FOLD ---
        if action == ACTION_FOLD:
            p.folded = True

        # --- CALL / CHECK ---
        elif action == ACTION_CALL:
            call_amount = max(0, self.to_call - p.street_bet)
            call_amount = min(call_amount, p.stack)

            p.stack -= call_amount
            p.bet += call_amount
            p.street_bet += call_amount
            self.pot += call_amount

            if p.stack == 0:
                p.all_in = True

            self.actions_without_raise += 1

        # --- RAISE ---
        elif action == ACTION_RAISE:
            if raise_amount is None:
                raise ValueError("Raise requires raise_amount")

            call_amount = max(0, self.to_call - p.street_bet)

            max_raise = max(0, p.stack - call_amount)
            raise_amount = max(0, raise_amount)
            raise_amount = min(raise_amount, max_raise)

            total = call_amount + raise_amount

            p.stack -= total
            p.bet += total
            p.street_bet += total
            self.pot += total

            if p.stack == 0:
                p.all_in = True

            self.to_call = p.street_bet
            self.actions_without_raise = 0

        # --- ALL-IN ---
        elif action == ACTION_ALL_IN:
            amount = p.stack

            p.stack = 0
            p.bet += amount
            p.street_bet += amount
            self.pot += amount
            p.all_in = True

            if p.street_bet > self.to_call:
                self.to_call = p.street_bet
                self.actions_without_raise = 0

        active_players = [pl for pl in self.players if not pl.folded]

        if all(pl.street_bet == self.to_call or pl.all_in for pl in active_players):
            self.to_call = 0

        self.next_player()