from collections import namedtuple

import collections
import copy
import datetime
import gc
import time
import numpy as np
from src.util.logconf import logging
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

def importstr(module_str, from_ = None):
    if from_ is None and ":" in module_str:
        module_name, from_ = module_str.rsplit(":")
    module =  __import__(module_name, fromlist=from_)
    for sub_str in module_str.split(".")[1:]:
        module = getattr(module, sub_str)
    if from_ is None:
        try:
            return getattr(module, sub_str)
        except:
            raise ImportError('{}.{}'.format(from_, module_str))
    return module
def enumerateWithEstimate(
        iter,
        desc_str,
        start_ndx,
        print_ndx = 4,
        backoff = None,
        iter_len = None
):
    if iter_len is None:
        iter_len = len(iter)
    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2
        assert backoff >= 2
        while print_ndx < start_ndx * backoff:
            print_ndx *= backoff
        log.warning("{} --- /{}. starting".format(desc_str, iter_len))
        start = time.time()
        for (current_ndx, item) in enumerate(iter):
            yield(current_ndx, item)
            if current_ndx == print_ndx:
                duration_sec = ((time.time() - start) / (current_ndx - start_ndx + 1) * (iter_len - start_ndx))
                done_dt = datetime.datetime.fromtimestamp(start + duration_sec)
                done_td = datetime.timedelta(seconds=duration_sec)
                log.info("{} {:-4}/{}, done at {}, {}".format(
                    desc_str,
                    current_ndx,
                    iter_len,
                    str(done_dt).rsplit('.', 1)[0],
                    str(done_td).rsplit('.', 1)[0],
                ))

                print_ndx *= backoff

            if current_ndx + 1 == start_ndx:
                start = time.time()

            log.warning("{} ----/{}, done at {}".format(
                desc_str,
                iter_len,
                str(datetime.datetime.now()).rsplit('.', 1)[0],
            ))
