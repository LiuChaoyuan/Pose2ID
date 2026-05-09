# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    def __init__(
            self,
            root='',
            verbose=True,
            pid_begin=0,
            dataset_dir='',
            train_dir='',
            query_dir='',
            gallery_dir='',
            query_gen_dir='',
            gallery_gen_dir='',
            ipg_pose_num=8,
            **kwargs):
        super(Market1501, self).__init__()
        dataset_dir = dataset_dir or self.dataset_dir
        train_dir = train_dir or 'bounding_box_train'
        query_dir = query_dir or 'query'
        gallery_dir = gallery_dir or 'bounding_box_test'
        query_gen_dir = query_gen_dir or 'query_gen'
        gallery_gen_dir = gallery_gen_dir or 'bounding_box_test_gen'

        self.dataset_dir = self._resolve_dir(root, dataset_dir)
        self.train_dir = self._resolve_dir(self.dataset_dir, train_dir)
        self.query_dir = self._resolve_dir(self.dataset_dir, query_dir)
        self.gallery_dir = self._resolve_dir(self.dataset_dir, gallery_dir)
        self.query_gen_dir = self._resolve_dir(self.dataset_dir, query_gen_dir)
        self.gallery_gen_dir = self._resolve_dir(self.dataset_dir, gallery_gen_dir)
        self.ipg_pose_num = ipg_pose_num

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True, stage='train')
        query = self._process_dir(self.query_dir, relabel=False, stage='query')
        gallery = self._process_dir(self.gallery_dir, relabel=False, stage='gallery')

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    @staticmethod
    def _resolve_dir(root, directory):
        if osp.isabs(directory):
            return directory
        return osp.join(root, directory)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False, stage='train'):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            if stage == 'query':
                img_paths_ipg = self._build_ipg_paths(img_path, self.query_gen_dir, pid)
            elif stage == 'gallery':
                img_paths_ipg = self._build_ipg_paths(img_path, self.gallery_gen_dir, pid)
            else:
                img_paths_ipg = img_path

            dataset.append((img_path, self.pid_begin + pid, camid, 1, img_paths_ipg))

        return dataset

    def _build_ipg_paths(self, img_path, gen_dir, pid):
        if pid == 0:
            return [img_path for _ in range(self.ipg_pose_num)]

        img_name = osp.basename(img_path)
        return [
            osp.join(gen_dir, 'pose{}'.format(i + 1), img_name)
            for i in range(self.ipg_pose_num)
        ]
