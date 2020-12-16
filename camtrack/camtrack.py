#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2
import sortednp as snp

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


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
    compute_reprojection_errors,
    _remove_correspondences_with_ids,
    eye3x4
)

# --show ../videos/fox_head_short.mov ../data_examples/fox_camera_short.yml track.yml point_cloud.yml --camera-poses ../data_examples/ground_truth_short.yml --frame-1 0 --frame-2 10
#--show ../dataset/ironman_translation_fast/rgb/* ../dataset/ironman_translation_fast/camera.yml ../dataset/ironman_translation_fast/track.yml ../dataset/ironman_translation_fast/point_cloud.yml --camera-poses ../dataset/ironman_translation_fast/ground_truth.yml --frame-1 0 --frame-2 10
#--show ../dataset/bike_translation_slow/rgb/* ../dataset/bike_translation_slow/camera.yml ../dataset/bike_translation_slow/track.yml ../dataset/bike_translation_slow/point_cloud.yml --camera-poses ../dataset/bike_translation_slow/ground_truth.yml --frame-1 0 --frame-2 10
class CameraTracker:

    def __init__(self, corner_storage, intrinsic_mat, view_1, view_2):
        self.corner_storage = corner_storage
        self.frame_count = len(corner_storage)
        self.intrinsic_mat = intrinsic_mat
        self.triangulation_parameters = TriangulationParameters(1, 3, .1)
        self.view_mats = {}

        if view_1 is None or view_2 is None:
            print('Finding initial poses')
            pose1_idx, pose1, pose2_idx, pose2 = self.find_best_start_poses()
            print('Initial poses found in frames {0} and {1}'.format(pose1_idx, pose2_idx))
        else:
            pose1 = pose_to_view_mat3x4(view_1[1])
            pose2 = pose_to_view_mat3x4(view_2[1])
            pose1_idx = view_1[0]
            pose2_idx = view_2[0]

        corners_1 = self.corner_storage[pose1_idx]
        corners_2 = self.corner_storage[pose2_idx]

        corrs = build_correspondences(corners_1, corners_2)

        p, ids, med = triangulate_correspondences(corrs, pose1, pose2,
                                                  self.intrinsic_mat, self.triangulation_parameters)

        self.ids = ids
        self.point_cloud_builder = PointCloudBuilder(ids, p)
        self.point_frames = {}
        self.last_retriangulated = {}
        for i in ids:
            self.point_frames[i] = [pose1_idx, pose2_idx]
            self.last_retriangulated[i] = 2

        self.used_inliers = {}
        self.view_nones = {i for i in range(self.frame_count)}
        self.view_mats[pose1_idx] = pose1
        self.used_inliers[pose1_idx] = 0
        self.view_nones.remove(pose1_idx)
        self.view_mats[pose2_idx] = pose2
        self.used_inliers[pose2_idx] = 0
        self.view_nones.remove(pose2_idx)
        self.last_added_idx = pose2_idx
        self.last_inliers = []

        self.retriangulate_frames = 3
        self.max_repr_error = 1.5
        self.used_inliers_point = {}
        self.step = 0

    def track(self):
        while len(self.view_mats) < self.frame_count:
            print('Processing {0}/{1} frame'.format(len(self.view_mats) + 1, self.frame_count))
            for i, v_mat in self.view_mats.items():
                corners_1 = self.corner_storage[i]
                corners_2 = self.corner_storage[self.last_added_idx]

                corrs = build_correspondences(corners_1, corners_2)
                try:
                    p, ids, _ = triangulate_correspondences(corrs, v_mat, self.view_mats[self.last_added_idx],
                                                            self.intrinsic_mat, self.triangulation_parameters)
                    new_corners = []
                    new_ids = []
                    for j, pt in zip(ids, p):
                        if j in self.last_inliers or j not in self.point_frames:
                            new_ids += [j]
                            new_corners += [pt]
                    for i_ in ids:
                        if i_ not in self.point_frames:
                            self.point_frames[i_] = [i]
                        if self.last_added_idx not in self.point_frames[i_]:
                            self.point_frames[i_] += [self.last_added_idx]
                    if len(new_ids) > 0:
                        self.point_cloud_builder.add_points(np.array(new_ids), np.array(new_corners))
                except:
                    continue


            print('Points cloud size: {0}'.format(len(self.point_cloud_builder.points)))
            view_mat, best_idx, inliers = self.solve_pnp_ransac()
            self.view_mats[best_idx] = view_mat[:3]
            self.used_inliers[best_idx] = inliers
            self.view_nones.remove(best_idx)
            self.last_added_idx = best_idx
            self.retriangulate_points()
            if self.step % 10 == 0 and self.step > 0:
                self.update_tracks()
                k = list(self.view_mats.keys())
                k = k[self.step - 10:self.step]
                self.bundle_adjustment(k)
            self.step += 1

    def update_tracks(self):
        for i in self.view_mats.keys():
            v_mat, inliers = self.solve_pnp_ransac_one(i)
            if self.used_inliers[i] <= len(inliers):
                self.view_mats[i] = v_mat[:3]
                self.used_inliers[i] = len(inliers)

    def solve_pnp_ransac(self):
        best = None
        best_ans = []
        nones = list(self.view_nones)
        for i in nones:
            try:
                v_mat, inliers = self.solve_pnp_ransac_one(i)
            except:
                continue
            if len(inliers) > len(best_ans):
                best = v_mat, i
                best_ans = inliers
        print('Used {} inliers'.format(len(best_ans)))

        self.last_inliers = best_ans

        return best[0], best[1], len(best_ans)

    def solve_pnp_ransac_one(self, frame_id):
        ids_3d = self.point_cloud_builder.ids
        ids_2d = self.corner_storage[frame_id].ids
        ids_intersection, _ = snp.intersect(ids_3d.flatten(), ids_2d.flatten(), indices=True)
        idx3d = [i for i, j in enumerate(ids_3d) if j in ids_intersection]
        idx2d = [i for i, j in enumerate(ids_2d) if j in ids_intersection]
        points3d = self.point_cloud_builder.points[idx3d]
        points2d = self.corner_storage[frame_id].points[idx2d]

        if len(points3d) < 6:
            return None, []

        _, _, _, inliers = cv2.solvePnPRansac(
            objectPoints=points3d,
            imagePoints=points2d,
            cameraMatrix=self.intrinsic_mat,
            distCoeffs=np.array([]),
            iterationsCount=250,
            flags=cv2.SOLVEPNP_EPNP
        )

        if len(points3d[inliers]) < 6:
            return None, []

        _, r_vec, t_vec = cv2.solvePnP(
            objectPoints=points3d[inliers],
            imagePoints=points2d[inliers],
            cameraMatrix=self.intrinsic_mat,
            distCoeffs=np.array([]),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
        return view_mat, inliers

    def retriangulate_points(self):
        to_retriangulate = [i for i in self.point_cloud_builder.ids
                            if ((i[0] not in self.last_retriangulated) or
                                (len(self.point_frames[i[0]]) - self.last_retriangulated[i[0]]) > 5)
                            and i[0] in self.point_frames]
        np.random.shuffle(to_retriangulate)
        to_retriangulate = to_retriangulate[:300]
        self.retriangulate_points_arr(to_retriangulate)

    def retriangulate_points_arr(self, to_retriangulate):
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
                self.last_retriangulated[idx[0]] = len(self.point_frames[idx[0]])
        retriangulated_ids = np.array(retriangulated_ids)
        retriangulated_coords = np.array(retriangulated_coords)
        if len(retriangulated_coords) == 0:
            return
        print('Retriangulated {} points'.format(len(retriangulated_ids)))
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
        for _ in range(7):
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
        if best_cnt == 0:
            return None, None
        return best_coords, best_cnt

    def find_best_start_poses(self):
        best_i = 0
        best_j = 0
        best_j_pose = None
        best_pose_points = 0
        for i in range(self.frame_count):
            for j in range(i + 5, self.frame_count, 5):
                pose, pose_points = self.get_pose(i, j)
                if pose_points > best_pose_points:
                    best_i = i
                    best_j = j
                    best_j_pose = pose
                    best_pose_points = pose_points
        return best_i, eye3x4(), best_j, pose_to_view_mat3x4(best_j_pose)

    def get_pose(self, frame_1, frame_2):
        p1 = self.corner_storage[frame_1]
        p2 = self.corner_storage[frame_2]

        corresp = build_correspondences(p1, p2)

        if len(corresp.ids) < 6:
            return None, 0

        E, mask_essential = cv2.findEssentialMat(corresp[1], corresp[2], self.intrinsic_mat, method=cv2.RANSAC,
                                                 threshold=1.0)

        _, mask_homography = cv2.findHomography(corresp[1], corresp[2], method=cv2.RANSAC)

        if mask_essential is None or mask_homography is None:
            return None, 0

        essential_inliers = mask_essential.flatten().sum()
        homography_inliers = mask_homography.flatten().sum()

        if homography_inliers / essential_inliers > 0.5:
            return None, 0

        corresp = _remove_correspondences_with_ids(corresp, np.argwhere(mask_essential == 0))

        R1, R2, t = cv2.decomposeEssentialMat(E)

        possible_poses = [Pose(R1.T, R1.T @ t), Pose(R1.T, R1.T @ (-t)), Pose(R2.T, R2.T @ t), Pose(R2.T, R2.T @ (-t))]

        best_pose_points = 0
        best_pose = None

        for pose in possible_poses:
            p, ids, med = triangulate_correspondences(corresp, eye3x4(), pose_to_view_mat3x4(pose), self.intrinsic_mat, self.triangulation_parameters)
            if len(p) > best_pose_points:
                best_pose_points = len(p)
                best_pose = pose

        return best_pose, best_pose_points

    def bundle_adjustment(self, frames):
        def view_mat3x4_to_rodrigues_and_translation(vmat):
            r_mat = vmat[:, :3]
            _t_vec = vmat[:, 3:]
            _r_vec, _ = cv2.Rodrigues(r_mat)
            return _r_vec, _t_vec

        def bundle_adjustment_sparsity(n_cameras, n_points):
            m = camera_indices.size * 2
            n = n_cameras * 6 + n_points * 3
            A_ = lil_matrix((m, n), dtype=int)
            i = np.arange(len(camera_indices))
            for s in range(6):
                A_[2 * i, camera_indices * 6 + s] = 1
                A_[2 * i + 1, camera_indices * 6 + s] = 1

            for s in range(3):
                A_[2 * i, n_cameras * 6 + points_indices * 3 + s] = 1
                A_[2 * i + 1, n_cameras * 6 + points_indices * 3 + s] = 1
            return A_

        def rotate(points_, rot_vecs):
            theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
            with np.errstate(invalid='ignore'):
                v = rot_vecs / theta
                v = np.nan_to_num(v)
            dot = np.sum(points_ * v, axis=1)[:, np.newaxis]
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            return cos_theta * points_ + sin_theta * np.cross(v, points_) + dot * (1 - cos_theta) * v

        def project(points_, camera_params):
            points_proj = rotate(points_, camera_params[:, :3])
            points_proj += camera_params[:, 3:6]
            points_proj = np.dot(self.intrinsic_mat, points_proj.T)
            points_proj /= points_proj[[2]]
            return points_proj[:2].T

        def fun(params_, n_cameras, n_points):
            camera_params = params_[:n_cameras * 6].reshape((n_cameras, 6))
            points_3d = params_[n_cameras * 6:].reshape((n_points, 3))
            points_proj = project(points_3d[points_indices], camera_params[camera_indices])
            return (points_proj - corners_points).ravel()

        def residuals(params_):
            return fun(params_, len(frames), len(intersected))

        def apply_res(res_):
            for i in range(len(intersected)):
                p_ = res_[6 * len(frames) + i * 3:6 * len(frames) + i * 3 + 3]
                _idx = np.argwhere(self.point_cloud_builder.ids == intersected[i])[0][0]
                _p = self.point_cloud_builder.points[_idx]
                check_point(_p, p_, intersected[i])
            for i in range(len(frames)):
                r_vec_ = np.array([res_[i * 6], res_[i * 6 + 1], res_[i * 6 + 2]])
                t_vec_ = np.array([[res_[i * 6 + 3]], [res_[i * 6 + 4]], [res_[i * 6 + 5]]])
                v_mat = rodrigues_and_translation_to_view_mat3x4(r_vec_, t_vec_)
                self.view_mats[frames[i]] = v_mat

        def check_point(p_before, p_after, idx):
            projected_before = []
            projected_after = []
            point2d = []
            for frame in frames:
                projected_before += [project_points(np.array([p_before]), self.intrinsic_mat @ self.view_mats[frame])]
                projected_after += [project_points(np.array([p_after]), self.intrinsic_mat @ self.view_mats[frame])]
                idx_ = np.argwhere(self.corner_storage[frame].ids == idx)[0][0]
                p = self.corner_storage[frame].points[idx_]
                point2d += [p]
            projected_before = np.array(projected_before)
            projected_after = np.array(projected_after)
            point2d = np.array(point2d)
            err_before = np.linalg.norm(point2d - projected_before)
            err_before_amount = np.sum(err_before < 1.0)
            err_after = np.linalg.norm(point2d - projected_after)
            err_after_amount = np.sum(err_after < 1.0)
            if err_before_amount < err_after_amount:
                self.point_cloud_builder.points[idx] = p_after


        intersected = self.point_cloud_builder.ids.flatten()
        corners_idxs = {}
        for frame in frames:
            crnrs = self.corner_storage[frame].ids
            crnrs = crnrs.flatten()
            intersected, _ = snp.intersect(
                intersected, crnrs,
                indices=True)

        intersected = np.array(list(sorted(np.random.choice(intersected, min(500, len(intersected)), replace=False))))
        for frame in frames:
            corner_idx, _ = snp.intersect(
                intersected,
                self.corner_storage[frame].ids.flatten(),
                indices=True)
            corners_idxs[frame] = corner_idx

        points = []
        for idx in intersected:
            idx_ = np.argwhere(self.point_cloud_builder.ids == idx)[0][0]
            p = self.point_cloud_builder.points[idx_]
            points += [p[0], p[1], p[2]]
        corners_points = []
        for frame in frames:
            for idx in corners_idxs[frame]:
                idx_ = np.argwhere(self.corner_storage[frame].ids == idx)[0][0]
                p = self.corner_storage[frame].points[idx_]
                corners_points += [[p[0], p[1]]]
        camera_parameters = []
        for frame in frames:
            r_vec, t_vec = view_mat3x4_to_rodrigues_and_translation(self.view_mats[frame])
            camera_parameters += [r_vec[0], r_vec[1], r_vec[2], t_vec[0], t_vec[1], t_vec[2]]
        params = camera_parameters + points

        camera_indices = []
        for i in range(len(frames)):
            camera_indices += [i] * len(intersected)
        camera_indices = np.array(camera_indices)
        points_indices = np.tile(np.arange(len(intersected)), len(frames))

        self.retriangulate_points_arr([np.array([item]) for item in intersected])

        A = bundle_adjustment_sparsity(len(frames), len(intersected))
        res = least_squares(residuals, params, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, xtol=1e-4, method='trf')
        apply_res(res.x)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    np.random.seed(1337)

    camera_tracker = CameraTracker(corner_storage, intrinsic_mat, known_view_1, known_view_2)
    # try:
    camera_tracker.track()
    view_mats = [camera_tracker.view_mats[key] for key in sorted(camera_tracker.view_mats.keys())]
    # except Exception as e:
    #     print("Exception ocurred, can\'t restore more positions")
    #     view_mats_ = {key: camera_tracker.view_mats[key] for key in sorted(camera_tracker.view_mats.keys())}
    #     for key in sorted(camera_tracker.view_nones):
    #         view_mats_[key] = eye3x4()
    #     view_mats = [view_mats_[key] for key in sorted(view_mats_)]



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
