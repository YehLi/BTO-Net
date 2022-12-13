from losses.cross_entropy import CrossEntropy
from losses.label_smoothing import LabelSmoothing
from losses.reward_criterion import RewardCriterion, ObjRewardCriterion
from losses.nce_loss import NCELoss
from losses.multi_label_loss import MultiLabelLoss
from losses.attention_loss import AttentionLoss, ObjAttentionLoss
from losses.mil_losses import MilLoss

__factory = {
    'CrossEntropy': CrossEntropy,
    'LabelSmoothing': LabelSmoothing,
    'RewardCriterion': RewardCriterion,
    'ObjRewardCriterion': ObjRewardCriterion,
    'MultiLabel': MultiLabelLoss,
    'NCE': NCELoss,
    'Attention': AttentionLoss,
    'ObjAttention': ObjAttentionLoss,
    'MIL': MilLoss
}

def names():
    return sorted(__factory.keys())

def create(name):
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name]()