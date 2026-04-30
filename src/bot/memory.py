import random
from collections import deque

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions.

    Stores tuples of experiences (e.g., state, action, reward, next_state, done)
    and allows random mini-batch sampling for stable training.
    """

    def __init__(self, capacity):
        """Initialize the buffer.

        Args:
            capacity (int): Maximum number of stored transitions.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """Add a transition to the buffer.

        Args:
            *args: Elements of a transition (e.g., state, action, reward, ...).
        """
        self.buffer.append(args)

    def sample(self, batch_size):
        """Sample a random batch of transitions.

        Args:
            batch_size (int): Number of samples to return.

        Returns:
            list: Batch of transitions grouped by element (zipped).
        """
        batch = random.sample(self.buffer, batch_size)
        return list(zip(*batch))

    def __len__(self):
        """Return current size of the buffer."""
        return len(self.buffer)

class ReservoirBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.n_seen = 0

    def push(self, state, action):
        self.n_seen += 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action))
        else:
            idx = random.randint(0, self.n_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = (state, action)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch

    def __len__(self):
        return len(self.buffer)