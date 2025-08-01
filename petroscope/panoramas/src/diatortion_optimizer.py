from torch.optim.lr_scheduler import StepLR
from kornia.geometry.calibration import undistort_points

import numpy as np
import torch
import torch.nn as nn
import random

from classes import StitchingData


class DistortionOptimizer:
    def __init__(self, device, data: StitchingData, f=10000.0, cx=0.0, cy=0.0, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0):

        self.device = device
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        self.data = data
        self.reper_idx = data.reper_idx

        homographies = []
        for i, id in enumerate(data.tile_set.order):
            img = data.tile_set.images[id]
            H = img.homography.reshape(-1)
            H = torch.tensor(H, device=self.device, dtype=torch.float32)
            homographies.append(H)
        homographies = torch.stack(homographies, dim=1)  # 9 * n_images

        self.n_images = homographies.shape[1]

        self.a_params = nn.Parameter(
            torch.zeros((4, self.n_images - 1), device=self.device, dtype=torch.float32, requires_grad=True)
        )
        self.b_params = nn.Parameter(
            torch.zeros((2, self.n_images - 1), device=self.device, dtype=torch.float32, requires_grad=True)
        )
        self.c_params = nn.Parameter(
            torch.zeros((2, self.n_images - 1), device=self.device, dtype=torch.float32, requires_grad=True)
        )

        a, b, c = self.homography_to_tens(homographies)
        with torch.no_grad():
            self.a_params.data = a
            self.b_params.data = b
            self.c_params.data = c

        self.f = nn.Parameter(torch.tensor([f], device=self.device, dtype=torch.float32, requires_grad=True))
        self.c_xy = nn.Parameter(torch.tensor([cx, cy], device=self.device, dtype=torch.float32, requires_grad=True))
        self.k1 = nn.Parameter(torch.tensor([k1], device=self.device, dtype=torch.float32, requires_grad=True))
        self.k2 = nn.Parameter(torch.tensor([k2], device=self.device, dtype=torch.float32, requires_grad=True))
        self.k3 = nn.Parameter(torch.tensor([k3], device=self.device, dtype=torch.float32, requires_grad=True))
        self.p = nn.Parameter(torch.tensor([p1, p2], device=self.device, dtype=torch.float32, requires_grad=True))

        self.inliers = []
        for inlier in data.matches:
            xy_i = torch.tensor(inlier.xy_i, device=self.device, dtype=torch.float32)
            xy_j = torch.tensor(inlier.xy_j, device=self.device, dtype=torch.float32)
            self.inliers.append({
                'i': inlier.i,
                'j': inlier.j,
                'xy_i': xy_i,
                'xy_j': xy_j
            })

    def get_camera_matrix(self) -> torch.Tensor:
        zero = torch.zeros(1, device=self.device, dtype=torch.float32)
        one = torch.ones(1, device=self.device, dtype=torch.float32)

        row1 = torch.cat([self.f, zero, self.c_xy[0].unsqueeze(0)])
        row2 = torch.cat([zero, self.f, self.c_xy[1].unsqueeze(0)])
        row3 = torch.cat([zero, zero, one])
        camera_matrix = torch.stack([row1, row2, row3], dim=0)
        return camera_matrix

    def get_inverse_camera_matrix(self) -> torch.Tensor:
        zero = torch.zeros(1, device=self.device, dtype=torch.float32)
        one = torch.ones(1, device=self.device, dtype=torch.float32)

        eps = 1e-18

        row1 = torch.cat([1.0 / self.f, zero, -self.c_xy[0].unsqueeze(0) / (self.f + eps)])
        row2 = torch.cat([zero, 1.0 / self.f, -self.c_xy[1].unsqueeze(0) / (self.f + eps)])
        row3 = torch.cat([zero, zero, one])
        inverse_camera_matrix = torch.stack([row1, row2, row3], dim=0)
        return inverse_camera_matrix

    def get_distortion_params(self) -> torch.Tensor:
        batch_size = 1
        dist_coeffs = torch.zeros((batch_size, 14), device=self.k1.device, dtype=self.k1.dtype)
        dist_coeffs[:, 0] = self.k1
        dist_coeffs[:, 1] = self.k2
        dist_coeffs[:, 2] = self.p[0].unsqueeze(0)
        dist_coeffs[:, 3] = self.p[1].unsqueeze(0)
        dist_coeffs[:, 4] = self.k3

        return dist_coeffs

    def get_homographies(self) -> torch.Tensor:

        homographies = torch.stack([
            self.a_params[0],
            self.a_params[1],
            self.b_params[0],
            self.a_params[2],
            self.a_params[3],
            self.b_params[1],
            self.c_params[0],
            self.c_params[1],
            torch.ones_like(self.a_params[1])
        ])

        homographies = torch.cat([
            homographies[:, :self.reper_idx],
            torch.eye(3,  device=self.device, dtype=torch.float32).view(-1, 1),
            homographies[:, self.reper_idx:]
        ], dim=1)

        return homographies

    def homography_to_tens(self, Hs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        homographies = torch.cat([
            Hs[:, :self.reper_idx],
            Hs[:, self.reper_idx + 1:]
        ], dim=1)
        a_params = torch.stack([
             homographies[0],
             homographies[1],
             homographies[3],
             homographies[4],
        ])

        b_params = torch.stack([
             homographies[2],
             homographies[5],
        ])

        c_params = torch.stack([
             homographies[6],
             homographies[7],
        ])

        return a_params, b_params, c_params

    def project(self, xy: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        n_matches = xy.shape[0]
        ones = torch.ones(n_matches, 1, device=self.device, dtype=torch.float32)
        project_points = torch.cat([xy, ones], dim=1)

        new_points = H @ project_points.T
        new_points = new_points / (new_points[2] + 1e-8)
        return new_points[:2].T

    def reprojection_mse(self) -> torch.Tensor:
        all_errors = []
        homographies = self.get_homographies()
        camera_matrix = self.get_camera_matrix()
        # inverse_camera_matrix = self.get_inverse_camera_matrix()

        dist_coeffs = self.get_distortion_params()

        for inlier in self.inliers:
            i = inlier['i']
            j = inlier['j']
            xy_i = inlier['xy_i']
            xy_j = inlier['xy_j']

            Hi = homographies[:, i].reshape(3, 3)
            Hj = homographies[:, j].reshape(3, 3)

            xy_i_corrected = undistort_points(
                xy_i.unsqueeze(0),
                camera_matrix.unsqueeze(0),
                dist_coeffs,
                num_iters=5
            ).squeeze(0)
            xy_j_corrected = undistort_points(
                xy_j.unsqueeze(0),
                camera_matrix.unsqueeze(0),
                dist_coeffs,
                num_iters=5
            ).squeeze(0)

            xy_i_warped = self.project(xy_i_corrected, Hi)
            xy_j_warped = self.project(xy_j_corrected, Hj)

            diff = (xy_i_warped - xy_j_warped)
            reproj_error = torch.sum(diff ** 2, dim=1)
            all_errors.append(reproj_error)

        return torch.cat(all_errors).mean()

    def bundle_adjustment(self, lr_f=1e3, lr_c=1e1, lr_k1=1e-2, lr_k2=1e-4, lr_k3=1e-6, lr_p=1e-3,
                          h_gamma=0.9, d_gamma=0.1,
                          max_iter=10000, plot=False, verbose='full') -> StitchingData:

        homographies_optimizer = torch.optim.Adam([
            {'params': [self.a_params], 'lr': 1e-3},
            {'params': [self.b_params], 'lr': 1e-0},
            {'params': [self.c_params], 'lr': 1e-6}
        ], betas=(0.9, 0.999), eps=1e-8)

        distortion_optimizer = torch.optim.Adam([
            {'params': [self.f], 'lr': lr_f},
            {'params': [self.c_xy], 'lr': lr_c},
            {'params': [self.k1], 'lr': lr_k1},
            {'params': [self.k2], 'lr': lr_k2},
            {'params': [self.k3], 'lr': lr_k3},
            {'params': [self.p], 'lr': lr_p},
        ], betas=(0.9, 0.999), eps=1e-8)

        homographies_scheduler = StepLR(homographies_optimizer, step_size=1000, gamma=h_gamma)
        distortion_scheduler = StepLR(distortion_optimizer, step_size=1000, gamma=d_gamma)

        with torch.no_grad():
            initial_loss = torch.sqrt(self.reprojection_mse()).item()
            print(f"Initial error: {initial_loss}")

        iteration_history = [0]
        loss_history = [initial_loss]
        for iteration in range(max_iter):

            homographies_optimizer.zero_grad()
            distortion_optimizer.zero_grad()

            loss = self.reprojection_mse()
            loss.backward()

            homographies_optimizer.step()
            distortion_optimizer.step()

            homographies_scheduler.step()
            distortion_scheduler.step()

            # current_loss = loss.item()

            if (iteration + 1) % 100 == 0:
                current_error = torch.sqrt(loss).item()
                iteration_history.append(iteration + 1)
                loss_history.append(current_error)
                if verbose == 'full':
                    print(f"Iteration: {iteration + 1} Loss: {current_error}")

        with torch.no_grad():
            final_loss = torch.sqrt(self.reprojection_mse()).item()
            print(f"Optimized error: {final_loss}")

        print(
            f"Final params:\n"
            f"f={self.f.item():.2f}\n"
            f"cx={self.c_xy[0].item():.2f}\n"
            f"cy={self.c_xy[1].item():.2f}\n"
            f"k1={self.k1.item():.6f}\n"
            f"k2={self.k2.item():.6f}"
        )

        with torch.no_grad():
            homographies = self.get_homographies()

        for i, id in enumerate(self.data.tile_set.order):
            img = self.data.tile_set.images[id]
            img.homography = homographies[:, i].view(3, 3).cpu().numpy()

        return final_loss  # self.data
