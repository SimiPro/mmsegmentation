from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class RobinDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('box', 'ice_pack'),
        palette=[[20, 20, 255], [255, 20, 20]]
    )


    def __init__(self, data_root, data_prefix, pipeline=[], 
                 img_suffix = '.png', 
                 ann_suffix = '.png', ann_file="",  **kwargs):
        super().__init__(data_root=data_root,
                          data_prefix=data_prefix, 
                          pipeline=pipeline, img_suffix=img_suffix,
                            seg_map_suffix=ann_suffix, ann_file=ann_file, **kwargs)