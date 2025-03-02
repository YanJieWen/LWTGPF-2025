'''
@File: __init__.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 22, 2024
@HomePage: https://github.com/YanJieWen
'''

from .market1501 import Market1501
from .dukemtmc import DukeMTMC
from .cuhk03 import CUHK03
from .occduke import OccDuke
__factory = {
    'Market-1501': Market1501,
    'DukeMTMC': DukeMTMC,
    'CUHK03': CUHK03,
    'Occ_Duke': OccDuke,
    # 'MSMT17': MSMT17,
    # 'personx': PersonX,
    # 'VeRi': VeRi,
    # 'vehicleid': VehicleID,
    # 'vehiclex': VehicleX
}


def names():
    return sorted(__factory.keys())


def create(name,root,*args,**kwargs):
    if name not in __factory:
        raise KeyError('Unkown datast:', name)
    return __factory[name](root,*args,**kwargs)