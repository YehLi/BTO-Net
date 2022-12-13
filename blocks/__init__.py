from blocks.feedforward_block import FeedForwardBlock
from blocks.lowrank_bilinear_block import LowRankBilinearEncBlock, LowRankBilinearDecBlock
from blocks.multihead_attention_block import MultiheadAttentionBlock
from blocks.multimodal_mh_block import MultimodalMHBlock
from blocks.multimodal_mh_block import MultimodalSEBlock, MultimodalSEBlock2

__factory = {
    'MultiheadAttention': MultiheadAttentionBlock,
    'MultimodalMH': MultimodalMHBlock,
    'MultimodalSE': MultimodalSEBlock,
    'MultimodalSE2': MultimodalSEBlock2,
    'FeedForward': FeedForwardBlock,
    'LowRankBilinearEnc': LowRankBilinearEncBlock,
    'LowRankBilinearDec': LowRankBilinearDecBlock,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown blocks:", name)
    return __factory[name](*args, **kwargs)