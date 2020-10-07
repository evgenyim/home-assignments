#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _to_np_int8(arr):
    return (255 * arr).astype(np.uint8)


class CornerTracker:

    def __init__(self, frame):
        self.frame = frame
        self.max_corners = 10000
        self.quality = 0.008
        self.min_dist = 4
        self.corners, self.radiuses = self.find_corners(frame, max_corners=self.max_corners, quality=0.002)
        self.corner_ids = np.arange(self.corners.shape[0])
        self.last_id = self.corner_ids.shape[0]

    def process_frame(self, new_frame):
        self.corners, mask = self.apply_opt_flow(new_frame, self.corners)
        self.corners = self.corners[mask]
        self.radiuses = self.radiuses[mask]
        self.corner_ids = self.corner_ids[mask]
        filter_ = self.create_filter(new_frame, self.corners, self.min_dist)
        if self.corners.shape[0] < self.max_corners:
            new_corners, new_radiuses = self.find_corners(new_frame, filter_,
                                                          max_corners=self.max_corners - self.corners.shape[0],
                                                          quality=self.quality)
            new_ids = np.arange(self.last_id, self.last_id + new_corners.shape[0])
            self.last_id += new_ids.shape[0]
            self.corners = np.concatenate([self.corners, new_corners])
            self.radiuses = np.concatenate([self.radiuses, new_radiuses])
            self.corner_ids = np.concatenate([self.corner_ids, new_ids])

        mask = self.remove_close_corners(self.corners, self.min_dist)
        self.corners = self.corners[mask == 1]
        self.radiuses = self.radiuses[mask == 1]
        self.corner_ids = self.corner_ids[mask == 1]

        self.frame = new_frame

    def find_corners(self, frame, filter_=None, pyr_level=3, point_size=4, max_corners=10000, quality=0.008,
                     min_dist=4, block_size=7):
        corners = np.empty((0, 2)).astype(np.float32)
        radiuses = np.empty(0)
        coef = 1
        frame_ = frame.copy()
        for i in range(pyr_level):
            if filter_ is not None:
                filter_[filter_ != 255] = 0
            new_corners = cv2.goodFeaturesToTrack(_to_np_int8(frame_), max_corners, quality, min_dist * coef,
                                                  mask=filter_, blockSize=block_size)
            if new_corners is None:
                new_corners = np.empty((0, 2)).astype(np.float32)
            new_corners *= coef
            new_corners = new_corners.reshape(new_corners.shape[0], 2)
            new_radiuses = np.ones(new_corners.shape[0]) * point_size * coef
            corners = np.concatenate([corners, new_corners])
            radiuses = np.concatenate([radiuses, new_radiuses])
            new_filter = self.create_filter(frame_, new_corners / coef, area_size=min_dist)
            frame_ = cv2.pyrDown(frame_)
            if filter_ is not None:
                filter_mask = np.where(new_filter == 0)
                filter_[filter_mask] = 0
                filter_ = cv2.pyrDown(filter_)
            else:
                filter_ = new_filter
                filter_ = cv2.pyrDown(filter_)
            coef *= 2
            max_corners = max(0, max_corners - new_corners.shape[0])
            if max_corners == 0:
                return corners, radiuses
        return corners, radiuses

    def create_filter(self, image, corners, area_size=4):
        filter_ = np.full(image.shape, 255).astype(np.uint8)
        for p in corners:
            filter_ = cv2.circle(filter_, (p[0], p[1]), area_size, 0, -1)
        return filter_

    def apply_opt_flow(self, new_frame, corners, max_level=2):
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(_to_np_int8(self.frame), _to_np_int8(new_frame),
                                              corners, None, maxLevel=max_level)
        pb, stb, _ = cv2.calcOpticalFlowPyrLK(_to_np_int8(new_frame), _to_np_int8(self.frame),
                                              p1, None, maxLevel=max_level)
        dists = abs(corners - pb).squeeze().max(axis=1) < 0.3
        mask = dists & (st1 == 1).squeeze()
        return p1, mask

    def remove_close_corners(self, corners, min_dist):
        good_corners = np.array([corners[0]])
        st = np.ones(corners.shape[0])
        for i in range(corners.shape[0]):
            p = corners[i]
            c = np.min(np.sum(([p] - good_corners) ** 2, axis=1))
            if np.sqrt(c) > min_dist:
                good_corners = np.concatenate([good_corners, [p]])
            else:
                st[i] = 0
        return st

    def get_corners(self):
        return self.corners, self.corner_ids, self.radiuses


def get_pyr_level(image, level):
    if level == 0:
        return image
    image_0 = cv2.GaussianBlur(image, (3, 3), 0)
    image_0 = np.delete(image_0, list(range(0, image.shape[0], 2 ** level)), axis=0)
    image_0 = np.delete(image_0, list(range(0, image.shape[1], 2 ** level)), axis=1)
    return image_0


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:

    image_0 = frame_sequence[0]

    corner_tracker = CornerTracker(image_0)
    corners, corners_ids, radiuses = corner_tracker.get_corners()
    builder.set_corners_at_frame(0, FrameCorners(corners_ids, corners, radiuses))

    for frame, image_1 in enumerate(frame_sequence[1:], 1):

        corner_tracker.process_frame(image_1)
        corners, corners_ids, radiuses = corner_tracker.get_corners()
        builder.set_corners_at_frame(frame, FrameCorners(corners_ids, corners, radiuses))


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
