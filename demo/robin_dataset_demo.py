# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot

from mmengine.registry import init_default_scope
from mmseg.datasets import RobinDataset


def main():
    print("start")
    init_default_scope('mmseg')

    data_root = 'mmsegmentation/data/robin/'

    data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train')
    dataset = RobinDataset(data_root=data_root, data_prefix=data_prefix, 
                            pipeline=[], 
                            img_suffix = '.png',
                            ann_suffix = '.png'
                            )

    print(f"len(robin_dataset): {len(dataset)}")
    

if __name__ == '__main__':
    main()
