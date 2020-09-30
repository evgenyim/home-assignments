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


def get_pyr_level(image, level):
    if level == 0:
        return image
    image_0 = cv2.GaussianBlur(image, (3, 3), 0)
    image_0 = np.delete(image_0, list(range(0, image.shape[0], 2 ** level)), axis=0)
    image_0 = np.delete(image_0, list(range(0, image.shape[1], 2 ** level)), axis=1)
    return image_0


def calc_new_corners(image_0, image_1, corners1, corners_ids1, radiuses1, corners2, ids_number,
                     max_corners, quality, min_dist, point_size, pyr_level):

    image_1_2 = get_pyr_level(image_1, pyr_level)

    p1, st1, _ = cv2.calcOpticalFlowPyrLK(_to_np_int8(image_0), _to_np_int8(image_1),
                                          corners1, None, maxLevel=1)
    pb, stb, _ = cv2.calcOpticalFlowPyrLK(_to_np_int8(image_1), _to_np_int8(image_0),
                                          p1, None, maxLevel=1)
    dists = abs(corners1 - pb).squeeze().max(axis=1) < 0.5
    mask = dists & (st1 == 1).squeeze()
    corners_ids1 = corners_ids1[mask]
    corners1 = p1[mask]
    radiuses1 = radiuses1[mask]

    if len(corners1) < max_corners:
        new_possible_points = cv2.goodFeaturesToTrack(image_1_2, max_corners, quality, min_dist)
        new_possible_points2 = []
        for p in new_possible_points:
            p0 = np.array([p[0][0] * (2 ** pyr_level), p[0][1] * (2 ** pyr_level)]).reshape(1, 2)
            c = np.min(np.sum((p0[np.newaxis, :, :] - corners2) ** 2, axis=2))
            if c > 25 * (1 + pyr_level):
                new_possible_points2 += [p0]
        new_possible_points = np.array(new_possible_points2).astype(np.float32)
        if new_possible_points is not None:
            if len(corners1) == 0:
                corners1 = new_possible_points
                corners_ids1 = np.arange(ids_number, len(corners1))
                ids_number += len(corners1)
                radiuses1 = np.ones(len(corners1)) * point_size
            else:
                new_corners = []
                new_ids = []
                new_radiuses = []
                for p in new_possible_points:
                    if len(new_corners) + len(corners1) >= max_corners:
                        break
                    c = np.min(np.sum((p[np.newaxis, :, :] - corners1) ** 2, axis=2))
                    if c > 25 * (1 + pyr_level):
                        new_corners += [p]
                        new_ids += [ids_number]
                        ids_number += 1
                        new_radiuses += [point_size]
                if len(new_corners) > 0:
                    corners1 = np.concatenate([corners1, new_corners])
                    corners_ids1 = np.concatenate([corners_ids1, new_ids])
                    radiuses1 = np.concatenate([radiuses1, new_radiuses])
    return corners1, corners_ids1, radiuses1, ids_number


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    max_corners = 5000
    max_2_corners = 5000
    point_size = 7
    quality = 0.008
    min_dist = 5
    min_2_dist = 10

    image_0 = frame_sequence[0]
    # Creating image from 1 level of pyramid
    image_0_2 = get_pyr_level(image_0, 1)

    corners1 = cv2.goodFeaturesToTrack(image_0, max_corners, quality, min_dist)
    corners_ids1 = np.arange(len(corners1))
    radiuses1 = np.ones(len(corners1)) * point_size

    corners2 = cv2.goodFeaturesToTrack(image_0_2, max_2_corners, quality, min_2_dist)
    ids_number = len(corners_ids1)
    new_possible_points2 = []
    corners_ids2 = []
    radiuses2 = []
    # For found corners on pyramid level 1 restoring coordinates on image
    for p in corners2:
        p0 = np.array([p[0][0] * 2, p[0][1] * 2]).reshape(1, 2)
        c = np.min(np.sum((p0[np.newaxis, :, :] - corners1) ** 2, axis=2))
        if c > 10:
            new_possible_points2 += [p0]
            corners_ids2 += [ids_number]
            ids_number += 1
            radiuses2 += [point_size * 2]
    corners2 = np.array(new_possible_points2).astype(np.float32)
    corners_ids2 = np.array(corners_ids2)
    radiuses2 = np.array(radiuses2)

    corners = np.concatenate((corners1, corners2))
    corners_ids = np.concatenate((corners_ids1, corners_ids2))
    radiuses = np.concatenate((radiuses1, radiuses2))

    builder.set_corners_at_frame(0, FrameCorners(corners_ids, corners, radiuses))

    for frame, image_1 in enumerate(frame_sequence[1:], 1):

        corners1, corners_ids1, radiuses1, ids_number = calc_new_corners(image_0, image_1, corners1, corners_ids1,
                                                                         radiuses1, corners2, ids_number, max_corners,
                                                                         quality, min_dist, point_size, 0)

        corners2, corners_ids2, radiuses2, ids_number = calc_new_corners(image_0, image_1, corners2, corners_ids2,
                                                                         radiuses2, corners1, ids_number, max_2_corners,
                                                                         quality, min_2_dist, point_size * 2, 1)

        corners = np.concatenate((corners1, corners2))
        corners_ids = np.concatenate((corners_ids1, corners_ids2))
        radiuses = np.concatenate((radiuses1, radiuses2))

        builder.set_corners_at_frame(frame, FrameCorners(corners_ids, corners, radiuses))
        image_0 = image_1


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
