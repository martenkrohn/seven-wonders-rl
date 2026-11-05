from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


MODEL_ARCHITECTURE_VERSION = 2
MULTIHEAD_ARCHITECTURE_VERSION = 3
MEMORY_ARCHITECTURE_VERSION = 4
MEMORY_MULTIHEAD_ARCHITECTURE_VERSION = 5


class ActorCritic(nn.Module):
    def __init__(self, global_dim: int, action_feat_dim: int, hidden: int = 128,
                 *, architecture_version: int = MODEL_ARCHITECTURE_VERSION, public_information=False):
        super().__init__()
        if architecture_version not in (1, MODEL_ARCHITECTURE_VERSION):
            raise ValueError(f"Unsupported model architecture version: {architecture_version}")
        self.architecture_version = architecture_version
        self.num_heads = 1
        self.public_information = public_information
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(action_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # A linear head here makes the global contribution identical for every
        # action, so it cancels in softmax. Joint nonlinear processing lets the
        # policy learn interactions between the state and each candidate action.
        # Version 1 is retained only for explicitly requested historical inference.
        self.policy_head = (nn.Linear(hidden * 2, 1) if architecture_version == 1 else
                            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                          nn.Linear(hidden, 1)))
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _global_context(self, global_flat):
        return self.global_mlp(global_flat)

    def forward_values(self, global_flat: torch.Tensor):
        """Critic-only bootstrap; keep the original batch shape and arithmetic."""
        return self.value_head(self._global_context(global_flat)).squeeze(-1)

    def forward_batch(self, global_flat: torch.Tensor, action_feats: torch.Tensor):
        # global_flat: (B, G), action_feats: (B, A, F)
        global_ctx = self._global_context(global_flat)  # (B, H)
        action_emb = self.action_mlp(action_feats)  # (B, A, H)

        global_expand = global_ctx.unsqueeze(1).expand(-1, action_emb.size(1), -1)
        logits = self.policy_head(torch.cat([action_emb, global_expand], dim=-1)).squeeze(-1)  # (B, A)
        values = self.value_head(global_ctx).squeeze(-1)  # (B,)
        return logits, values


class MultiHeadActorCritic(nn.Module):
    """Episode-selected PPO heads; independent priors see raw observations.

    Head IDs are explicit inputs, never mutable model state. Only the selected
    private head receives gradients; the two shared encoders learn from all heads.
    """
    def __init__(self, global_dim, action_feat_dim, hidden=128, *, num_heads=4,
                 prior_strength=0.0, prior_hidden=32, public_information=False,
                 prior_global_dim=None, prior_action_dim=None):
        super().__init__()
        if num_heads < 2 or not math.isfinite(prior_strength) or prior_strength < 0:
            raise ValueError('Multi-head models require at least two heads and a finite nonnegative prior strength')
        self.architecture_version = MULTIHEAD_ARCHITECTURE_VERSION
        self.num_heads, self.prior_strength, self.prior_hidden = num_heads, prior_strength, prior_hidden
        self.public_information = public_information
        self.prior_global_dim = global_dim if prior_global_dim is None else prior_global_dim
        self.prior_action_dim = action_feat_dim if prior_action_dim is None else prior_action_dim
        if not (0 < self.prior_global_dim <= global_dim and 0 < self.prior_action_dim <= action_feat_dim):
            raise ValueError('Prior input dimensions must be positive prefixes of model inputs')
        base = ActorCritic(global_dim, action_feat_dim, hidden)
        self.global_mlp, self.action_mlp = base.global_mlp, base.action_mlp
        self.policy_heads = nn.ModuleList([copy.deepcopy(base.policy_head) for _ in range(num_heads)])
        self.value_heads = nn.ModuleList([copy.deepcopy(base.value_head) for _ in range(num_heads)])
        # Independent initial heads when training from scratch. Warm starts below
        # explicitly copy the same trained head into each member instead.
        for head in list(self.policy_heads)[1:] + list(self.value_heads)[1:]:
            for layer in head:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        self.priors = nn.ModuleList([nn.Sequential(
            nn.Linear(self.prior_global_dim + self.prior_action_dim, prior_hidden), nn.Tanh(),
            nn.Linear(prior_hidden, 1), nn.Tanh()) for _ in range(num_heads)])
        self.priors.requires_grad_(False)

    def copy_single_head(self, source):
        if source.architecture_version != MODEL_ARCHITECTURE_VERSION:
            raise ValueError('Warm start requires a current single-head checkpoint')
        self.global_mlp.load_state_dict(source.global_mlp.state_dict())
        self.action_mlp.load_state_dict(source.action_mlp.state_dict())
        for actor, critic in zip(self.policy_heads, self.value_heads):
            actor.load_state_dict(source.policy_head.state_dict())
            critic.load_state_dict(source.value_head.state_dict())

    def _ids(self, global_flat, head_ids):
        if head_ids is None:
            raise ValueError('Multi-head inference requires explicit per-game head IDs')
        ids = torch.as_tensor(head_ids, dtype=torch.long, device=global_flat.device)
        if ids.shape != (len(global_flat),) or ((ids < 0) | (ids >= self.num_heads)).any():
            raise ValueError('Invalid head IDs')
        return ids

    def _global_context(self, global_flat):
        return self.global_mlp(global_flat)

    def forward_values(self, global_flat, head_ids=None):
        ids = self._ids(global_flat, head_ids)
        context = self._global_context(global_flat)
        values = context.new_zeros(len(context))
        for h in ids.unique().tolist():
            rows = ids == h
            values[rows] = self.value_heads[h](context[rows]).squeeze(-1)
        return values

    def forward_batch(self, global_flat, action_feats, head_ids=None):
        ids = self._ids(global_flat, head_ids)
        context = self._global_context(global_flat)
        actions = self.action_mlp(action_feats)
        joint = torch.cat([actions, context.unsqueeze(1).expand(-1, actions.size(1), -1)], -1)
        logits = context.new_zeros(action_feats.shape[:2])
        values = context.new_zeros(len(context))
        for h in ids.unique().tolist():
            rows = ids == h
            scores = self.policy_heads[h](joint[rows]).squeeze(-1)
            if self.prior_strength:
                raw = torch.cat([action_feats[rows, :, :self.prior_action_dim], global_flat[rows, :self.prior_global_dim].unsqueeze(1).expand(
                    -1, actions.size(1), -1)], -1)
                with torch.no_grad():
                    prior = self.priors[h](raw).squeeze(-1)
                scores = scores + self.prior_strength * prior
            logits[rows] = scores
            values[rows] = self.value_heads[h](context[rows]).squeeze(-1)
        return logits, values


def extend_public_inputs(source):
    """Neutral append-only migration. Priors retain their original inputs/weights."""
    from sevenwonders_rl.features.public_state import PUBLIC_GLOBAL_DIM, PUBLIC_ACTION_DIM
    if getattr(source, 'public_information', False):
        return source
    if (source.global_mlp[0].in_features, source.action_mlp[0].in_features) != (102, 99):
        raise ValueError('Public input migration requires the original 102/99 feature contract')
    model = copy.deepcopy(source)
    for name, extra in (('global_mlp', PUBLIC_GLOBAL_DIM), ('action_mlp', PUBLIC_ACTION_DIM)):
        encoder = getattr(model, name)
        old = encoder[0]
        with torch.random.fork_rng(devices=[]):
            new = nn.Linear(old.in_features + extra, old.out_features,
                            device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_features].copy_(old.weight)
            new.bias.copy_(old.bias)
        encoder[0] = new
    model.public_information = True
    return model


class SharedInformationEncoder(nn.Module):
    """Current hand, full action information and an ordered observation history.

    The final projection starts at zero for neutral weight migration. Invalid
    history slots are masked after encoding, including the encoder's bias.
    """
    def __init__(self, hidden):
        super().__init__()
        from sevenwonders_rl.features.observation_history import (
            CARD_COUNT, ACTION_DIM, HISTORY_LENGTH, FRAME_DIM)
        self.hand = nn.Sequential(nn.Linear(CARD_COUNT + 1, 32), nn.ReLU())
        self.opportunities = nn.Sequential(nn.Linear(ACTION_DIM, 32), nn.ReLU(),
                                           nn.Linear(32, 32), nn.ReLU())
        self.frame = nn.Sequential(nn.Linear(FRAME_DIM, 32), nn.ReLU())
        self.sequence = nn.Sequential(nn.Linear(HISTORY_LENGTH * 32, hidden), nn.ReLU())
        self.projection = nn.Linear(hidden + 64, hidden)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, inputs):
        from sevenwonders_rl.features.observation_history import (
            CURRENT_HAND, MASK, ACTION_FEATURES, FRAMES, VALID, HISTORY_LENGTH, FRAME_DIM,
            N_ACTIONS, ACTION_DIM)
        hand = self.hand(inputs[:, CURRENT_HAND])
        action_rows = inputs[:, ACTION_FEATURES].reshape(-1, N_ACTIONS, ACTION_DIM)
        legal = inputs[:, MASK].unsqueeze(-1)
        # A hand has no meaningful slot order. Sum valid choice embeddings so
        # permuting card slots does not alter the shared actor/critic context.
        opportunities = (self.opportunities(action_rows) * legal).sum(1) / 7.
        frames = inputs[:, FRAMES].reshape(-1, HISTORY_LENGTH, FRAME_DIM)
        encoded = self.frame(frames) * inputs[:, VALID].unsqueeze(-1)
        history = self.sequence(encoded.flatten(1))
        return self.projection(torch.cat((hand, opportunities, history), dim=-1))


class _MemoryContext:
    def _initialize_memory(self, global_dim, action_feat_dim, hidden):
        from sevenwonders_rl.features.observation_history import BASE_GLOBAL_DIM, ACTION_DIM, MEMORY_DIM
        if global_dim != BASE_GLOBAL_DIM + MEMORY_DIM or action_feat_dim != ACTION_DIM:
            raise ValueError('Observation-memory model dimensions do not match its schema')
        self.memory_enabled = True
        self.global_dim = global_dim
        self.information_encoder = SharedInformationEncoder(hidden)

    def _global_context(self, global_flat):
        from sevenwonders_rl.features.observation_history import BASE_GLOBAL_DIM
        if global_flat.shape[-1] != self.global_dim:
            raise ValueError('Observation-memory input is missing or has the wrong size')
        base = super()._global_context(global_flat[:, :BASE_GLOBAL_DIM])
        # Both actor and critic consume this same context, including the legal
        # mask and all current action features in the explicitly packed payload.
        return base + self.information_encoder(global_flat[:, BASE_GLOBAL_DIM:])


class MemoryActorCritic(_MemoryContext, ActorCritic):
    def __init__(self, global_dim, action_feat_dim, hidden=128, *, public_information=True):
        from sevenwonders_rl.features.observation_history import BASE_GLOBAL_DIM
        if not public_information:
            raise ValueError('Observation memory requires public information')
        super().__init__(BASE_GLOBAL_DIM, action_feat_dim, hidden, public_information=True)
        self._initialize_memory(global_dim, action_feat_dim, hidden)
        self.architecture_version = MEMORY_ARCHITECTURE_VERSION


class MemoryMultiHeadActorCritic(_MemoryContext, MultiHeadActorCritic):
    def __init__(self, global_dim, action_feat_dim, hidden=128, *, public_information=True, **kwargs):
        from sevenwonders_rl.features.observation_history import BASE_GLOBAL_DIM
        if not public_information:
            raise ValueError('Observation memory requires public information')
        super().__init__(BASE_GLOBAL_DIM, action_feat_dim, hidden, public_information=True, **kwargs)
        self._initialize_memory(global_dim, action_feat_dim, hidden)
        self.architecture_version = MEMORY_MULTIHEAD_ARCHITECTURE_VERSION


def extend_observation_memory(source):
    """Preserve existing predictions and frozen priors; add a trainable residual."""
    from sevenwonders_rl.features.observation_history import BASE_GLOBAL_DIM, MEMORY_DIM
    if getattr(source, 'memory_enabled', False):
        return source
    if not getattr(source, 'public_information', False):
        raise ValueError('Add public inputs before observation memory')
    options = dict(public_information=True)
    if source.num_heads > 1:
        options.update(num_heads=source.num_heads, prior_strength=source.prior_strength,
            prior_hidden=source.prior_hidden, prior_global_dim=source.prior_global_dim,
            prior_action_dim=source.prior_action_dim)
    with torch.random.fork_rng(devices=[]):
        cls = MemoryMultiHeadActorCritic if source.num_heads > 1 else MemoryActorCritic
        model = cls(BASE_GLOBAL_DIM + MEMORY_DIM, source.action_mlp[0].in_features,
            hidden=source.global_mlp[0].out_features, **options)
    parameter = next(source.parameters())
    model = model.to(device=parameter.device, dtype=parameter.dtype)
    missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
    if unexpected or any(not key.startswith('information_encoder.') for key in missing):
        raise ValueError('Unsupported source architecture for observation-memory migration')
    return model.train(source.training)
