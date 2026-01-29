from constants import *

def get_legal_actions(player, to_call) -> list[str]:

    actions = []

    # Fold zawsze możliwy (jeśli jest bet)
    if to_call > player.street_bet:
        actions.append(ACTION_FOLD)

    # Call tylko jeśli jest co callować
    if to_call > player.street_bet and player.stack > 0:
        actions.append(ACTION_CALL)

    # Check tylko jeśli nic nie trzeba dopłacać
    if to_call == player.street_bet:
        actions.append(ACTION_CALL) #check

    # Raise tylko jeśli gracz ma środki
    if player.stack > (to_call - player.street_bet):
        actions.append(ACTION_RAISE)

    # All-in zawsze możliwy jeśli gracz ma stack
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
            call_amount = max(0, self.to_call - p.street_bet)  # call_amount nie może być ujemne
            call_amount = min(call_amount, p.stack)  # call_amount nie może być większe niż stack

            p.stack -= call_amount
            p.bet += call_amount
            p.street_bet += call_amount
            self.pot += call_amount
            self.actions_without_raise += 1

            if p.stack == 0:
                p.all_in = True

        # --- RAISE ---
        elif action == ACTION_RAISE:
            if raise_amount is None:
                raise ValueError("Raise requires raise_amount")

            call_amount = max(0, self.to_call - p.street_bet)
            total = call_amount + raise_amount

            if total >= p.stack:
                total = p.stack
                p.all_in = True

            p.stack -= total
            p.bet += total
            p.street_bet += total
            self.pot += total

            self.to_call = p.street_bet
            self.actions_without_raise = 0
            # po raisie nie sprawdzamy końca  rundy

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

        self.next_player()