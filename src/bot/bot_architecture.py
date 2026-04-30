import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared Encoder

class StateEncoder(nn.Module):
    """Multi-layer perceptron (MLP) that encodes input states into a latent feature space. """

    def __init__(self, input_dim, hidden_dim=256):
        """Initialize the encoder network.

        The architecture consists of two fully connected layers with ReLU activations.

        Args:
            input_dim (int): Dimensionality of the input state.
            hidden_dim (int, optional): Size of hidden layers and output embedding.
                Defaults to 256.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

    def forward(self, x):
        """Compute latent representation of the input.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Encoded features of shape (batch_size, hidden_dim).
        """
        return self.net(x)


#Q-Network (Best Response)
class QNetwork(nn.Module):
    """Q-network that maps states to action values using a shared encoder.

    Combines a state encoder with a linear head to produce Q-values
    for each possible action.
    """

    def __init__(self, state_dim, num_actions):
        """Initialize the Q-network.

        Args:
            state_dim (int): Dimensionality of the input state.
            num_actions (int): Number of discrete actions.
        """
        super().__init__()

        self.encoder = StateEncoder(state_dim)
        self.head = nn.Linear(512, num_actions)

    def forward(self, x):
        """Compute Q-values for given states.

        Args:
            x (Tensor): Input tensor of shape (batch_size, state_dim).

        Returns:
            Tensor: Q-values of shape (batch_size, num_actions).
        """
        z = self.encoder(x)
        q = self.head(z)
        return q

#Policy Network (Average Strategy)

class PolicyNetwork(nn.Module):
    """Policy network that maps states to action probabilities.

    Uses a shared state encoder followed by a linear layer and softmax
    to produce a probability distribution over actions.
    """

    def __init__(self, state_dim, num_actions):
        """Initialize the policy network.

        Args:
            state_dim (int): Dimensionality of the input state.
            num_actions (int): Number of discrete actions.
        """
        super().__init__()

        self.encoder = StateEncoder(state_dim)
        self.head = nn.Linear(512, num_actions)

    def forward(self, x):
        """Compute action probabilities for given states.

        Args:
            x (Tensor): Input tensor of shape (batch_size, state_dim).

        Returns:
            Tensor: Action probabilities of shape (batch_size, num_actions).
        """
        z = self.encoder(x)
        logits = self.head(z)
        return logits



# NFSP Model Wrapper
class NFSPModel(nn.Module):
    """Neural Fictitious Self-Play model.

    Combines:
    - Q-network (best response)
    - Target Q-network (stabilization)
    - Policy network (average strategy)
    """

    def __init__(self, state_dim, num_actions):
        """Initialize NFSP components.

        Args:
            state_dim (int): Dimensionality of the input state.
            num_actions (int): Number of discrete actions.
        """
        super().__init__()

        self.q_net = QNetwork(state_dim, num_actions)
        self.target_q_net = QNetwork(state_dim, num_actions)

        self.policy_net = PolicyNetwork(state_dim, num_actions)

        self.update_target()

    def update_target(self):
        """Synchronize target Q-network with the current Q-network."""
        self.target_q_net.load_state_dict(self.q_net.state_dict())


#Action Masking Helper

def mask_illegal_actions(logits, legal_mask):
    """Mask illegal actions by assigning them very low logits.

    Adds a large negative value to logits corresponding to illegal actions,
    so that after softmax they receive near-zero probability.

    Args:
        logits (Tensor): Raw action scores of shape
        legal_mask (Tensor): Binary mask of same shape as logits, where
            1 indicates a legal action and 0 indicates an illegal action.

    Returns:
        Tensor: Adjusted logits with illegal actions effectively suppressed.
    """
    return logits + (legal_mask - 1) * 1e9


# Action Selection
def select_action_nfsp(model, state, legal_mask, eta=0.1, epsilon=0.05):
    """Select an action using NFSP (best response or average policy).

    With probability `eta`, selects an action using the Q-network (best response)
    with epsilon-greedy exploration. Otherwise, samples from the policy network
    (average strategy). Illegal actions are masked out in both cases.

    Args:
        model (NFSPModel): Model containing Q-network and policy network.
        state (array-like): Environment state.
        legal_mask (array-like): Binary mask (1 = legal, 0 = illegal).
        eta (float, optional): Probability of using best response (Q-network).
        epsilon (float, optional): Exploration rate for epsilon-greedy.

    Returns:
        tuple:
            int: Selected action index.
            str: Mode used ("BR" for best response, "AVG" for average policy).
    """
    state = torch.FloatTensor(state).unsqueeze(0)
    legal_mask = torch.FloatTensor(legal_mask)

    with torch.no_grad():
        # POLICY π
        logits = model.policy_net(state).squeeze(0)
        masked_logits = mask_illegal_actions(logits, legal_mask)
        pi_probs = torch.softmax(masked_logits, dim=-1)

        # BEST RESPONSE β (epsilon-greedy)
        q_values = model.q_net(state).squeeze(0)
        masked_q = mask_illegal_actions(q_values, legal_mask)

        greedy_action = torch.argmax(masked_q).item()

        beta_probs = torch.zeros_like(pi_probs)
        legal_actions = torch.where(legal_mask == 1)[0]

        beta_probs = torch.zeros_like(pi_probs)

        legal_actions = torch.where(legal_mask == 1)[0]

        beta_probs[legal_actions] = epsilon / len(legal_actions)

        beta_probs[greedy_action] = (1 - epsilon) + (epsilon / len(legal_actions))

    use_br = torch.rand(1).item() < eta

    if use_br:
        probs = beta_probs
        mode = "BR"
    else:
        probs = pi_probs
        mode = "AVG"

    action = torch.multinomial(probs, 1).item()

    return action, mode

def compute_q_loss(model, batch, gamma=0.99):
    """Compute MSE loss for the Q-network (DQN-style update).

    Uses the target Q-network to compute stable TD targets and applies
    mean squared error between predicted Q-values and targets.

    Args:
        model (NFSPModel): Model containing Q-network and target Q-network.
        batch (tuple): Tuple of (states, actions, rewards, next_states, dones).
        gamma (float, optional): Discount factor. Defaults to 0.99.

    Returns:
        Tensor: Scalar loss value for Q-network training.
    """
    states, actions, rewards, next_states, dones, next_legal_masks = batch

    states = torch.stack(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.stack(next_states)
    dones = torch.FloatTensor(dones)
    next_legal_masks = torch.stack(next_legal_masks)

    q_values = model.q_net(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()

    with torch.no_grad():
        next_q_values = model.target_q_net(next_states)

        next_q_values = next_q_values.masked_fill(next_legal_masks == 0, -1e9)

        next_q = next_q_values.max(1)[0]
        target = rewards + gamma * next_q * (1 - dones)

    loss = F.mse_loss(q_value, target)
    return loss

def compute_policy_loss(model, batch):
    """Compute cross-entropy loss for the policy network.

    Trains the policy network to imitate actions (average strategy)
    by maximizing the likelihood of selected actions.

    Args:
        model (NFSPModel): Model containing the policy network.
        batch (tuple): Tuple of (states, actions).

    Returns:
        Tensor: Scalar loss value for policy network training.
    """
    states, actions = zip(*batch)

    states = torch.stack([
        s if isinstance(s, torch.Tensor) else torch.FloatTensor(s)
        for s in states
    ])
    actions = torch.LongTensor(actions)

    logits = model.policy_net(states)
    logits = torch.clamp(logits, -10, 10)

    log_probs = F.log_softmax(logits, dim=-1)
    loss = F.nll_loss(log_probs, actions)
    return loss

def build_optimizers(model, lr=1e-3):
    """Create optimizers for Q-network and policy network.

    Uses Adam optimizer for both components of the NFSP model.

    Args:
        model (NFSPModel): Model containing Q-network and policy network.
        lr (float, optional): Learning rate. Defaults to 1e-3.

    Returns:
        tuple:
            Optimizer: Optimizer for Q-network.
            Optimizer: Optimizer for policy network.
    """
    q_opt = torch.optim.Adam(model.q_net.parameters(), lr=lr)
    policy_opt = torch.optim.Adam(model.policy_net.parameters(), lr=lr)
    return q_opt, policy_opt