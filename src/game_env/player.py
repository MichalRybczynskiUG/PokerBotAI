from typing import List, Optional
from .deck import Card

class Player:
    """Represents a single player (or bot) in the poker game."""

    def __init__(self, name: str, stack: int = 1000, position: Optional[int] = None) -> None:
        """
        Initialize a player.

        Args:
            name (str): Player name or identifier.
            stack (int): Starting chip stack.
            position (Optional[int]): Seat position relative to dealer (0 = dealer).
        """
        self.name: str = name
        self.stack: int = stack
        self.position: Optional[int] = position
        self.hand: List[Card] = []
        self.folded: bool = False
        self.all_in: bool = False
        self.current_bet: int = 0
        self.total_bet: int = 0
        self.last_action: Optional[str] = None

    def reset_for_new_hand(self) -> None:
        """Reset the player's state for a new hand."""
        self.hand.clear()
        self.folded = False
        self.all_in = False
        self.current_bet = 0
        self.total_bet = 0
        self.last_action = None

    def bet(self, amount: int) -> int:
        """
        Place a bet by removing chips from the stack.

        Args:
            amount (int): Amount to bet.

        Returns:
            int: The actual amount bet (useful if player goes all-in).

        Raises:
            ValueError: If the player has folded or is all-in.
        """
        if self.folded:
            raise ValueError(f"{self.name} cannot bet after folding.")
        if self.all_in:
            raise ValueError(f"{self.name} cannot bet because they are already all-in.")

        bet_amount = min(amount, self.stack)
        self.stack -= bet_amount
        self.current_bet += bet_amount
        self.total_bet += bet_amount

        if self.stack == 0:
            self.all_in = True

        return bet_amount

    def is_active(self) -> bool:
        """Check if the player is still in the hand (not folded or all-in)."""
        return not self.folded and not self.all_in
