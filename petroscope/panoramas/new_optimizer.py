from classes import StitchingData
import torch
import torch.nn as nn
import torch.optim as optim

from logger import logger, log_time

device = torch.device('cuda:6')
# device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')


class PanoramaOptimizer(nn.Module):
    def __init__(self, homographies, fixed_idx, initial_cam_params):
        super().__init__()
        self.num_images = len(homographies)
        self.fixed_idx = fixed_idx

        # Преобразуем гомографии в параметры (8 параметров на гомографию)
        homos = []
        for i in range(self.num_images):
            H = homographies[i]
            H = H / H[2, 2]
            params = torch.tensor([
                H[0, 0], H[0, 1], H[0, 2],
                H[1, 0], H[1, 1], H[1, 2],
                H[2, 0], H[2, 1]
            ], dtype=torch.float64, device=device)
            homos.append(params)
        self.homographies = nn.Parameter(torch.stack(homos))  # [num_images, 8]

        # Параметры камеры: f (для fx и fy), cx, cy, k1, k2
        # Разделяем на группы с разными learning rates
        self.cam_params_group1 = nn.Parameter(
            torch.tensor(initial_cam_params[0:1], dtype=torch.float64, device=device)
        )  # f (для fx и fy)
        self.cam_params_group2 = nn.Parameter(
            torch.tensor(initial_cam_params[2:4], dtype=torch.float64, device=device)
        )  # cx, cy
        self.cam_params_group3 = nn.Parameter(
            torch.tensor(initial_cam_params[5:6], dtype=torch.float64, device=device)
        )  # k1
        self.cam_params_group4 = nn.Parameter(
            torch.tensor(initial_cam_params[6:7], dtype=torch.float64, device=device)
        )  # k2

    def forward(self, matches):
        loss = 0.0
        for m in matches:
            H_i = self.get_H(m.i)
            H_j = self.get_H(m.j)

            pts_i = torch.tensor(m.xy_i, dtype=torch.float64, device=device).view(-1, 2)
            pts_j = torch.tensor(m.xy_j, dtype=torch.float64, device=device).view(-1, 2)

            pts_i_proj = self.apply_homography(H_i, pts_i)
            pts_j_proj = self.apply_homography(H_j, pts_j)

            pts_i_dist = self.distort_points(
                pts_i_proj, self.cam_params_group1, self.cam_params_group2, 
                self.cam_params_group3, self.cam_params_group4
            )
            pts_j_dist = self.distort_points(
                pts_j_proj, self.cam_params_group1, self.cam_params_group2, 
                self.cam_params_group3, self.cam_params_group4
            )

            diff = pts_i_dist - pts_j_dist

            conf = torch.tensor(m.conf, dtype=torch.float64, device=device)
            # Ensure conf can be broadcasted to match diff's shape (N, 2)
            if conf.dim() == 1:
                conf = conf.unsqueeze(1)  # Shape (N, 1)
            elif conf.dim() > 2:
                conf = conf.view(-1, 1)  # Flatten to (N, 1) if needed
            weighted_diff = diff * conf  # учитываем confidence with broadcasting
            loss += (weighted_diff**2).sum()
        return loss

    def get_H(self, idx):
        params = self.homographies[idx]
        H = torch.eye(3, dtype=torch.float64, device=device)
        H[0, 0] = params[0]
        H[0, 1] = params[1]
        H[0, 2] = params[2]
        H[1, 0] = params[3]
        H[1, 1] = params[4]
        H[1, 2] = params[5]
        H[2, 0] = params[6]
        H[2, 1] = params[7]
        H[2, 2] = 1.0
        return H

    def apply_homography(self, H, pts):
        N = pts.shape[0]
        pts_h = torch.cat([pts, torch.ones(N, 1, dtype=torch.float64, device=device)], dim=1)
        pts_trans = (H @ pts_h.t()).t()
        return pts_trans[:, :2] / pts_trans[:, 2:3]

    def distort_points(self, pts, cam_params_group1, cam_params_group2, cam_params_group3, cam_params_group4):
        f = cam_params_group1[0]  # f используется для fx и fy
        cx, cy = cam_params_group2
        k1 = cam_params_group3[0]
        k2 = cam_params_group4[0]
        skew = 0.0  # зафиксировано на 0
        k3 = 0.0    # зафиксировано на 0
        p1 = 0.0    # зафиксировано на 0
        p2 = 0.0    # зафиксировано на 0
        x = (pts[:, 0] - cx - skew * (pts[:, 1] - cy)) / f
        y = (pts[:, 1] - cy) / f
        r2 = x**2 + y**2
        radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        x_dist = x * radial + 2*p1*x*y + p2*(r2 + 2*x**2)
        y_dist = y * radial + p1*(r2 + 2*y**2) + 2*p2*x*y
        x_pix = f * x_dist + cx + skew * (f * y_dist)
        y_pix = f * y_dist + cy
        return torch.stack([x_pix, y_pix], dim=1)

    def fix_reference_homography(self):
        with torch.no_grad():
            self.homographies[self.fixed_idx].copy_(
                torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float64, device=device)
            )


def optimize(homographies, matches, fixed_idx, initial_cam_params, n_iters=100, lr=1e-6):
    model = PanoramaOptimizer(homographies, fixed_idx, initial_cam_params).to(device)

    # Define parameter groups with different learning rates
    param_groups = [
        {'params': model.cam_params_group1, 'lr': 1e2},  # f (для fx и fy)
        {'params': model.cam_params_group2, 'lr': 1e1},  # cx, cy
        {'params': model.cam_params_group3, 'lr': 1e-2},  # k1
        {'params': model.cam_params_group4, 'lr': 1e-2},  # k2
        {'params': model.homographies, 'lr': 1e-5}       # homography parameters
    ]
    optimizer = optim.Adam(param_groups)

    for it in range(n_iters):
        optimizer.zero_grad()
        loss = model(matches)
        loss.backward()
        optimizer.step()
        model.fix_reference_homography()
        if it % 10 == 0:
            print(f"Iteration {it}, loss: {loss.item()}")

    optimized_homos = [model.get_H(i).detach().cpu().numpy() for i in range(model.num_images)]
    optimized_cam = torch.cat([
        model.cam_params_group1, model.cam_params_group1, model.cam_params_group2, 
        torch.tensor([0.0], dtype=torch.float64, device=device),  # skew
        model.cam_params_group3, model.cam_params_group4,
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device=device)  # k3, p1, p2
    ]).detach().cpu().numpy()
    return optimized_homos, optimized_cam


class Optimizer:
    def __init__(self, data: StitchingData):
        self.data = data
        self.inliers = data.matches
        self.reper_idx = data.reper_idx
        self.homographies = []
        for id in data.tile_set.order:
            img = data.tile_set.images[id]
            self.homographies.append(img.homography)

    @log_time("Bundle adjustment done for", logger)
    def bundle_adjustment(self) -> StitchingData:
        id = self.data.tile_set.order[0]
        w, h = self.data.tile_set.images[id].orig_size

        # Параметры камеры: f (для fx и fy), cx, cy, skew, k1, k2, k3, p1, p2
        # Начальные значения: f, cx, cy, skew=0, k1, k2, k3=0, p1=0, p2=0
        initial_cam_params = [40_000, 40_000, w / 2, h / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        model = PanoramaOptimizer(self.homographies, self.reper_idx, initial_cam_params)
        model.eval()

        with torch.no_grad():
            init_loss = model(self.inliers).item()
        logger.debug(f"Initial reprojection error (sum of squared residuals): {init_loss}")

        homographies, cam_params = optimize(self.homographies, self.inliers, self.reper_idx, initial_cam_params)

        model = PanoramaOptimizer(homographies, self.reper_idx, cam_params)
        model.eval()
        with torch.no_grad():
            final_loss = model(self.inliers).item()
        logger.debug(f"Final reprojection error (sum of squared residuals): {final_loss}")

        for i, id in enumerate(self.data.tile_set.order):
            img = self.data.tile_set.images[id]
            img.homography = homographies[i]

        print(cam_params)
        return self.data
