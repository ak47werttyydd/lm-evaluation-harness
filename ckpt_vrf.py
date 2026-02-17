import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
import torch.distributed.checkpoint as dcp

# 直接看 metadata
reader = FileSystemReader('/home/a84400789/flame/exp/gdn-340M-4K-20B/batch_search.lr3e-4.gbs50000.bs1.ga1.steps400000/checkpoint/step-610351')
md = reader.read_metadata()
for k in md.state_dict_metadata:
    if 'A' in k or 'decay' in k or 'alpha' in k:
        print(k)
