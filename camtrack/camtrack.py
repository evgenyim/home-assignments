#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2
import sortednp as snp

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    calc_inlier_indices,
    project_points
)


class CameraTracker:

    def __init__(self, corner_storage, intrinsic_mat, view_1, view_2):
        self.corner_storage = corner_storage
        self.frame_count = len(corner_storage)
        self.intrinsic_mat = intrinsic_mat
        corners_1 = self.corner_storage[view_1[0]]
        corners_2 = self.corner_storage[view_2[0]]

        corrs = build_correspondences(corners_1, corners_2)

        pose_1 = pose_to_view_mat3x4(view_1[1])
        pose_2 = pose_to_view_mat3x4(view_2[1])

        p, ids, med = triangulate_correspondences(corrs, pose_1, pose_2,
                                                  self.intrinsic_mat, TriangulationParameters(5, 1, .1))
        self.points2d = p
        self.ids = ids
        self.point_cloud_builder = PointCloudBuilder(ids, p)
        self.view_mats = {}
        self.view_nones = {i for i in range(self.frame_count)}
        self.view_mats[view_1[0]] = pose_1
        self.view_nones.remove(view_1[0])
        self.view_mats[view_2[0]] = pose_2
        self.view_nones.remove(view_2[0])
        self.last_added_idx = view_2[0]

    def track(self):
        while len(self.view_mats) < self.frame_count:
            print('Processing {0}/{1} frame'.format(len(self.view_mats) + 1, self.frame_count))
            for i, v_mat in self.view_mats.items():
                corners_1 = self.corner_storage[i]
                corners_2 = self.corner_storage[self.last_added_idx]

                corrs = build_correspondences(corners_1, corners_2)

                p, ids, med = triangulate_correspondences(corrs, v_mat, self.view_mats[self.last_added_idx],
                                                          self.intrinsic_mat, TriangulationParameters(5, 1, .1))
                self.point_cloud_builder.add_points(ids, p)
            print('Points cloud size: {0}'.format(len(self.point_cloud_builder.points)))
            view_mat, idx = self.solve_pnp_ransac()
            self.view_mats[idx] = view_mat[:3]
            self.view_nones.remove(idx)
            self.last_added_idx = idx
            self.remove_bad_points()

    def solve_pnp_ransac(self):
        best = -1
        best_ans = 0
        nones = list(self.view_nones)
        np.random.shuffle(nones)
        for i in nones:
            ids_3d = self.point_cloud_builder.ids
            ids_2d = self.corner_storage[i].ids
            ids_intersection, _ = snp.intersect(ids_3d.flatten(), ids_2d.flatten(), indices=True)
            idx3d = [i for i, j in enumerate(ids_3d) if j in ids_intersection]
            idx2d = [i for i, j in enumerate(ids_2d) if j in ids_intersection]
            points3d = self.point_cloud_builder.points[idx3d]
            points2d = self.corner_storage[i].points[idx2d]

            succeeded, r_vec, t_vec, inliers = cv2.solvePnPRansac(
                objectPoints=points3d,
                imagePoints=points2d,
                cameraMatrix=self.intrinsic_mat,
                distCoeffs=np.array([]),
                iterationsCount=250,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if len(inliers) > best_ans:
                best = (i, r_vec, t_vec)
                best_ans = len(inliers)
            break
        print('Used {} inliers'.format(best_ans))
        r_vec = best[1]
        t_vec = best[2]
        view_mat_3x4 = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)

        return view_mat_3x4, best[0]

    def remove_bad_points(self):
        to_remove = set()
        for i, v_mat in self.view_mats.items():
            ids_3d = self.point_cloud_builder.ids
            ids_2d = self.corner_storage[i].ids
            ids_intersection, _ = snp.intersect(ids_3d.flatten(), ids_2d.flatten(), indices=True)
            idx3d = [i for i, j in enumerate(ids_3d) if j in ids_intersection]
            idx2d = [i for i, j in enumerate(ids_2d) if j in ids_intersection]
            points3d = self.point_cloud_builder.points[idx3d]
            points2d = self.corner_storage[i].points[idx2d]
            projected_points = project_points(points3d, self.intrinsic_mat @ v_mat)

            mask = abs(projected_points - points2d).squeeze().max(axis=1) < 1
            bad_idx = np.array(np.where(mask == False))
            for j in bad_idx[0]:
                to_remove.add(idx3d[j])
        self.point_cloud_builder.remove_elements(to_remove)




def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    np.random.seed(1337)

    camera_tracker = CameraTracker(corner_storage, intrinsic_mat, known_view_1, known_view_2)
    camera_tracker.track()

    view_mats = [camera_tracker.view_mats[key] for key in sorted(camera_tracker.view_mats.keys())]

    point_cloud_builder = camera_tracker.point_cloud_builder

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
