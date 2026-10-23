"""Policy regressions: state/action interactions must survive normalization."""
import pytest
import torch

from sevenwonders_rl.utils.ppo_model import ActorCritic
from sevenwonders_rl.utils.ppo_utils import masked_logits

pytestmark = pytest.mark.unit


def interaction_model():
    """Construct logit(action, state) = abs(action - state) in the joint head."""
    model = ActorCritic(1, 1, hidden=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for encoder in (model.global_mlp, model.action_mlp):
            encoder[0].weight[0, 0] = 1
            encoder[2].weight[0, 0] = 1
        # Concatenation order is [action embedding, global embedding].
        model.policy_head[0].weight[0, 0] = 1
        model.policy_head[0].weight[0, 2] = -1
        model.policy_head[0].weight[1, 0] = -1
        model.policy_head[0].weight[1, 2] = 1
        model.policy_head[2].weight.fill_(1)
    return model


def test_global_context_can_reverse_action_preferences_with_fixed_action_features():
    model = interaction_model()
    features = torch.tensor([[[0.], [2.], [100.]]]).expand(2, -1, -1)
    logits, values = model.forward_batch(torch.tensor([[0.], [2.]]), features)
    probabilities = masked_logits(logits, torch.tensor([[1, 1, 0], [1, 1, 0]])).softmax(-1)
    assert values.shape == (2,)
    assert probabilities.argmax(-1).tolist() == [1, 0]
    assert torch.all(probabilities[:, 2] == 0)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))


def test_policy_loss_alone_trains_global_encoder():
    model = interaction_model()
    logits, _ = model.forward_batch(torch.tensor([[.5]]), torch.tensor([[[0.], [2.]]]))
    loss = -logits.log_softmax(-1)[0, 0]
    loss.backward()
    assert model.global_mlp[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in model.value_head.parameters())
