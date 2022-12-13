from models.updown import UpDown
from models.xlan import XLAN
from models.xtransformer import XTransformer
from models.transformer import Transformer
from models.btoformer import Btoformer, Objformer

__factory = {
    'UpDown': UpDown,
    'XLAN': XLAN,
    'XTransformer': XTransformer,
    'Transformer': Transformer,
    'Btoformer': Btoformer,
    'Objformer': Objformer
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)