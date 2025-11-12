from __future__ import annotations
import random
from typing import List


class Card:
    """Represents a single playing card"""

    ranks: List[str] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits: List[str] = ['Spades', 'Hearts', 'Diamonds', 'Clubs']

    def __init__(self, rank: str, suit: str) -> None:
        """
        Initialize a card.

        Args:
            rank (str): Card rank
            suit (str): Card suit
        """
        self.rank: str = rank
        self.suit: str = suit

    def __repr__(self) -> str:
        """
        Return a short string representation of the card
        """
        return f"{self.rank}{self.suit[0]}"


class Deck:
    """Represents a full deck of 52 playing cards"""

    def __init__(self) -> None:
        """Initialize and shuffle a new deck"""
        self.cards: List[Card] = [Card(r, s) for r in Card.ranks for s in Card.suits]
        random.shuffle(self.cards)

    def draw(self, n: int = 1) -> List[Card]:
        """
        Draw n cards from the deck

        Args:
            n (int): Number of cards to draw

        Returns:
            List[Card]: A list of drawn cards
        """
        return [self.cards.pop() for _ in range(n)]

    def show_cards(self) -> List[Card]:
        """
        Return the list of all remaining cards in the deck

        Returns:
            List[Card]: A list of Card objects still in the deck
        """
        return self.cards