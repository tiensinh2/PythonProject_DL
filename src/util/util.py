from collections import namedtuple

import collections
import copy
import datetime
import gc
import time
import numpy as np
from util.logconf import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
#IRC: Index, Row, Cow
IRCTuple = collections.namedtuple(
    'IRCTuple',
    ['index', 'rol', 'col']
)
#XYZ: real cordination
XYZTuple = collections.namedtuple(
    'XYZTuple',
    ['x', 'y', 'z']
)
def irc_to_xyz(irc_tuple, origin_xyz, vxSize_xyz, direction_a):
    cri_tuple = np.array(irc_tuple)[::-1]
    origin_tuple = np.array(origin_xyz)
    vxSize_tuple = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_tuple * vxSize_tuple) + origin_tuple)
    return XYZTuple(*coords_xyz)
def xyz_to_irc(xyz_tuple, origin_xyz, vxSize_xyz, direction_a):
    origin_xyz = np.array(origin_xyz)
    xyz = np.array(xyz_tuple)
    vxSize_xyz = np.array(vxSize_xyz)
    direction_a = np.array(direction_a)
    coors_irc = ((xyz - origin_xyz) @ np.linalg.inv(direction_a)) / vxSize_xyz
    coords_irc = np.round(coors_irc).astype(int)
    return IRCTuple(coords_irc[2], coords_irc[1], coords_irc[0])
