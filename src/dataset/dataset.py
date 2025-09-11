import copy
import csv
import functools
import glob
import os

from collections import namedtuple

import SimpleITK as sitk
import numpy as np

import torch
import torch.cuda
from torch.utils.data import Dataset

from src.util.disk import getCache
from src.util.util import XYZTuple, xyz_to_irc
from src.util.logconf import logging


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = getCache('/kaggle/working/raw_data_cache')
#
#
#GET CACHE
CandidateInfoTuple = namedtuple('CandidateInfoTuple',
                                'isNodule_bool, diameter_mm, series_uid, center_xyz')

@functools.lru_cache(1)
def getCandidateInfoList(requiredOnDisk_bool = True):
    mhd_list = glob.glob('/kaggle/input/luna16/subset0/subset0/*.mhd') #replace ... by data path in kaggle
    presentOnDisk_set = {os.path.split(x)[-1][:-4] for x in mhd_list}


    diameter_dict = {}
    with open('/kaggle/input/luna16/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple(float(x) for x in row[1:4])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append((annotationCenter_xyz, annotationDiameter_mm))

    candidateInfo_list = []
    with open('/kaggle/input/luna16/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in presentOnDisk_set and requiredOnDisk_bool:
                continue
            candidateCenter_xyz = tuple(float(x) for x in row[1:4])
            isNodule_bool = bool(int(row[4]))

            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(annotationCenter_xyz[i] - candidateCenter_xyz[i])
                    if delta_mm < annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break
            candidateInfo_list.append(CandidateInfoTuple(isNodule_bool, candidateDiameter_mm, series_uid, candidateCenter_xyz))
    candidateInfoList = sorted(candidateInfo_list, key = lambda x: x.isNodule_bool, reverse = True)
    return candidateInfoList

class Ct:
    def __init__(self, series_uid):
        mhd_path_files = glob.glob('/kaggle/input/luna16/subset0/subset0/{}.mhd'.format(series_uid))
        if not mhd_path_files:
            raise FileNotFoundError
        else:
            mhd_path = mhd_path_files[0]
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype = np.float32)
        ct_a.clip(-1000, 1000, ct_a)
        self.series_uid = series_uid
        self.hu_a = ct_a
        self.origin_xyz = XYZTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XYZTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz_to_irc(center_xyz,
                                self.origin_xyz,
                                self.vxSize_xyz,
                                self.direction_a)
        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(round(start_ndx + width_irc[axis]))
            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])
            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])
            slice_list.append(slice(start_ndx, end_ndx))
        ct_chunk = self.hu_a[tuple(slice_list)]
        return ct_chunk, center_irc

@functools.lru_cache(1)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

class LunaDataset(Dataset):
    def __init__(self,
                 val_stride = 0,
                 isValSet_bool = None,
                 series_uid = None):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfoList = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfoList[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            assert val_stride > 0, val_stride
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
        ))
    def __len__(self):
        return len(self.candidateInfo_list)
    def __getitem__(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)
        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = torch.unsqueeze(0)

        pos_t = torch.tensor([
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
        ],
               dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc),
        )








