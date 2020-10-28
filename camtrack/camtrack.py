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
    project_points,
    Correspondences,
    compute_reprojection_errors
)


class CameraTracker:

    def __init__(self, corner_storage, intrinsic_mat, view_1, view_2):
        self.corner_storage = corner_storage
        self.frame_count = len(corner_storage)
        self.intrinsic_mat = intrinsic_mat
        self.triangulation_parameters = TriangulationParameters(7, 0.5, .1)
        corners_1 = self.corner_storage[view_1[0]]
        corners_2 = self.corner_storage[view_2[0]]

        corrs = build_correspondences(corners_1, corners_2)

        pose_1 = pose_to_view_mat3x4(view_1[1])
        pose_2 = pose_to_view_mat3x4(view_2[1])

        p, ids, med = triangulate_correspondences(corrs, pose_1, pose_2,
                                                  self.intrinsic_mat, self.triangulation_parameters)
        self.points2d = p
        self.ids = ids
        self.point_cloud_builder = PointCloudBuilder(ids, p)
        self.view_mats = {}
        self.used_inliers = {}
        self.view_nones = {i for i in range(self.frame_count)}
        self.view_mats[view_1[0]] = pose_1
        self.used_inliers[view_1[0]] = len(ids)
        self.view_nones.remove(view_1[0])
        self.view_mats[view_2[0]] = pose_2
        self.used_inliers[view_2[0]] = len(ids)
        self.view_nones.remove(view_2[0])
        self.last_added_idx = view_2[0]

        self.retriangulate_frames = 3
        self.max_repr_error = 0.1
        self.last_retriangulated = {}
        self.used_inliers_point = {}
        self.step = 0

    def track(self):
        while len(self.view_mats) < self.frame_count:
            print('Processing {0}/{1} frame'.format(len(self.view_mats) + 1, self.frame_count))
            for i, v_mat in self.view_mats.items():
                corners_1 = self.corner_storage[i]
                corners_2 = self.corner_storage[self.last_added_idx]

                corrs = build_correspondences(corners_1, corners_2)

                p, ids, _ = triangulate_correspondences(corrs, v_mat, self.view_mats[self.last_added_idx],
                                                        self.intrinsic_mat, self.triangulation_parameters)
                self.point_cloud_builder.add_points(ids, p)
            print('Points cloud size: {0}'.format(len(self.point_cloud_builder.points)))
            view_mat, best_idx, inliers = self.solve_pnp_ransac()
            self.view_mats[best_idx] = view_mat[:3]
            self.used_inliers[best_idx] = inliers
            self.view_nones.remove(best_idx)
            self.last_added_idx = best_idx
            self.retriangulate_points(best_idx)
            if self.step % 10 == 0:
                self.update_tracks()
            self.step += 1

    def update_tracks(self):
        for i in self.view_mats.keys():
            v_mat, inliers = self.solve_pnp_ransac_one(i)
            if self.used_inliers[i] < len(inliers):
                self.view_mats[i] = v_mat
                self.used_inliers[i] = len(inliers)

    def solve_pnp_ransac(self):
        best = None
        best_ans = 0
        nones = list(self.view_nones)
        np.random.shuffle(nones)
        for i in nones:
            v_mat, inliers = self.solve_pnp_ransac_one(i)
            if len(inliers) > best_ans:
                best = v_mat, i
                best_ans = len(inliers)
            break
        print('Used {} inliers'.format(best_ans))

        return best[0], best[1], best_ans

    def solve_pnp_ransac_one(self, frame_id):
        ids_3d = self.point_cloud_builder.ids
        ids_2d = self.corner_storage[frame_id].ids
        ids_intersection, _ = snp.intersect(ids_3d.flatten(), ids_2d.flatten(), indices=True)
        idx3d = [i for i, j in enumerate(ids_3d) if j in ids_intersection]
        idx2d = [i for i, j in enumerate(ids_2d) if j in ids_intersection]
        points3d = self.point_cloud_builder.points[idx3d]
        points2d = self.corner_storage[frame_id].points[idx2d]

        _, r_vec, t_vec, inliers = cv2.solvePnPRansac(
            objectPoints=points3d,
            imagePoints=points2d,
            cameraMatrix=self.intrinsic_mat,
            distCoeffs=np.array([]),
            iterationsCount=250,
            flags=cv2.SOLVEPNP_EPNP
        )
        _, r_vec, t_vec = cv2.solvePnP(
            objectPoints=points3d[inliers],
            imagePoints=points2d[inliers],
            cameraMatrix=self.intrinsic_mat,
            distCoeffs=np.array([]),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
        return view_mat, inliers

    def retriangulate_points(self, frame_id):
        to_retriangulate = [i for i in self.corner_storage[frame_id].ids
                            if (i[0] not in self.last_retriangulated) or
                            (self.step - self.last_retriangulated[i[0]]) > 20]
        np.random.shuffle(to_retriangulate)
        to_retriangulate = to_retriangulate[:500]
        retriangulated_coords = []
        retriangulated_ids = []
        for idx in to_retriangulate:
            new_coords, cnt = self.retriangulate_point(idx)
            if new_coords is None:
                continue
            if idx[0] not in self.used_inliers_point or self.used_inliers_point[idx[0]] < cnt:
                self.used_inliers_point[idx[0]] = cnt
                retriangulated_ids += [idx[0]]
                retriangulated_coords += [new_coords[0]]
                self.last_retriangulated[idx[0]] = self.step
        retriangulated_ids = np.array(retriangulated_ids)
        retriangulated_coords = np.array(retriangulated_coords)
        self.point_cloud_builder.update_points(retriangulated_ids, retriangulated_coords)

    def retriangulate_point(self, point_id):
        coords = []
        frames = []
        v_mats = []
        for i, frame in enumerate(self.corner_storage):
            if point_id in frame.ids and i in self.view_mats.keys():
                idx = np.where(frame.ids == point_id)[0][0]
                coords += [frame.points[idx]]
                frames += [i]
                v_mats += [self.view_mats[i]]
        if len(coords) < 3:
            return None, None
        if len(coords) > self.retriangulate_frames:
            idxs = np.random.choice(len(coords), size=self.retriangulate_frames, replace=False)
            coords = np.array(coords)[idxs]
            frames = np.array(frames)[idxs]
            v_mats = np.array(v_mats)[idxs]
        best_coords = None
        best_cnt = 0
        for _ in range(5):
            i, j = np.random.choice(len(coords), 2, replace=True)
            corrs = Correspondences(np.array([point_id]), np.array([coords[i]]), np.array([coords[j]]))
            point3d, _, _ = triangulate_correspondences(corrs, v_mats[i], v_mats[j], self.intrinsic_mat,
                                                        self.triangulation_parameters)
            if len(point3d) == 0:
                continue
            errors = np.array([compute_reprojection_errors(point3d, np.array([p]), self.intrinsic_mat @ m)
                               for m, p in zip(v_mats, coords)])
            cnt = np.sum(errors < self.max_repr_error)
            if best_coords is None or best_cnt < cnt:
                best_cnt = cnt
                best_coords = point3d
        return best_coords, best_cnt


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
