from layers.low_rank import LowRank
from layers.basic_att import BasicAtt
from layers.sc_att import SCAtt
from layers.objencoderptr import ObjEncoderPtr
from layers.objencoderptr_onelayer import ObjEncoderPtrOneLayer
from layers.objencoderptrw import ObjEncoderPtrw

__factory = {
    'LowRank': LowRank,
    'BasicAtt': BasicAtt,
    'SCAtt': SCAtt,
    'OBJ_ENC_TWO': ObjEncoderPtr,
    'OBJ_ENC_ONE': ObjEncoderPtrOneLayer,
    'OBJW_ENC_TWO': ObjEncoderPtrw
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown layer:", name)
    return __factory[name](*args, **kwargs)