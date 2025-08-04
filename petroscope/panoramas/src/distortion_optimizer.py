# супер-пупер-быстрая SOTA оптимизация

from torch.optim.lr_scheduler import StepLR
from kornia.geometry.calibration import undistort_points

import numpy as np
import torch
import torch.nn as nn
import random

from classes import StitchingData


class DistortionOptimizer:
    def __init__(self, device, data: StitchingData, f=10000.0, cx=0.0, cy=0.0, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
                 freeze_principal_point=False, freeze_tangential=False):

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

        # Сохраняем флаги заморозки
        self.freeze_principal_point = freeze_principal_point
        self.freeze_tangential = freeze_tangential

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

        # Параметры главной точки - заморозка контролируется флагом
        self.c_xy = nn.Parameter(
            torch.tensor(
                [cx, cy], device=self.device, dtype=torch.float32,
                requires_grad=not freeze_principal_point
            )
        )

        self.k1 = nn.Parameter(torch.tensor([k1], device=self.device, dtype=torch.float32, requires_grad=True))
        self.k2 = nn.Parameter(torch.tensor([k2], device=self.device, dtype=torch.float32, requires_grad=True))
        self.k3 = nn.Parameter(torch.tensor([k3], device=self.device, dtype=torch.float32, requires_grad=True))

        # Тангенциальные параметры дисторсии - заморозка контролируется флагом
        self.p = nn.Parameter(
            torch.tensor(
                [p1, p2], device=self.device, dtype=torch.float32,
                requires_grad=not freeze_tangential
            )
        )

        # Векторизованная подготовка данных для матчей
        self._prepare_vectorized_matches(data.matches)

    def _prepare_vectorized_matches(self, matches):
        """Предварительная векторизация всех матчей для ускорения вычислений"""
        i_indices = []
        j_indices = []
        xy_i_all = []
        xy_j_all = []

        for match in matches:
            n_points = len(match.xy_i)
            i_indices.extend([match.i] * n_points)
            j_indices.extend([match.j] * n_points)
            xy_i_all.extend(match.xy_i)
            xy_j_all.extend(match.xy_j)

        # Конвертируем в numpy массивы сначала, затем в тензоры
        i_indices_np = np.array(i_indices, dtype=np.int64)
        j_indices_np = np.array(j_indices, dtype=np.int64)
        xy_i_all_np = np.array(xy_i_all, dtype=np.float32)
        xy_j_all_np = np.array(xy_j_all, dtype=np.float32)

        # Создаем тензоры из numpy массивов (намного быстрее)
        self.i_indices = torch.from_numpy(i_indices_np).to(self.device)
        self.j_indices = torch.from_numpy(j_indices_np).to(self.device)
        self.xy_i_all = torch.from_numpy(xy_i_all_np).to(self.device)
        self.xy_j_all = torch.from_numpy(xy_j_all_np).to(self.device)
        self.total_matches = len(i_indices)

    def get_camera_matrix_batch(self, batch_size=1) -> torch.Tensor:
        """Батчевая версия camera matrix для kornia"""
        # zero = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        one = torch.ones(batch_size, device=self.device, dtype=torch.float32)
        f_batch = self.f.expand(batch_size)
        cx_batch = self.c_xy[0].expand(batch_size)
        cy_batch = self.c_xy[1].expand(batch_size)

        camera_matrix = torch.zeros(batch_size, 3, 3, device=self.device, dtype=torch.float32)
        camera_matrix[:, 0, 0] = f_batch
        camera_matrix[:, 1, 1] = f_batch
        camera_matrix[:, 0, 2] = cx_batch
        camera_matrix[:, 1, 2] = cy_batch
        camera_matrix[:, 2, 2] = one

        return camera_matrix

    def get_distortion_params_batch(self, batch_size=1) -> torch.Tensor:
        """Батчевая версия distortion parameters"""
        dist_coeffs = torch.zeros((batch_size, 14), device=self.device, dtype=torch.float32)
        dist_coeffs[:, 0] = self.k1.expand(batch_size)
        dist_coeffs[:, 1] = self.k2.expand(batch_size)
        dist_coeffs[:, 2] = self.p[0].expand(batch_size)
        dist_coeffs[:, 3] = self.p[1].expand(batch_size)
        dist_coeffs[:, 4] = self.k3.expand(batch_size)
        return dist_coeffs

    def get_homographies(self) -> torch.Tensor:
        """Возвращает все гомографии как тензор [3, 3, n_images]"""
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

        # Вставляем единичную матрицу для референсного изображения
        identity = torch.eye(3, device=self.device, dtype=torch.float32).view(-1, 1)
        homographies = torch.cat([
            homographies[:, :self.reper_idx],
            identity,
            homographies[:, self.reper_idx:]
        ], dim=1)

        return homographies.view(3, 3, self.n_images)

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

    def project_vectorized(self, xy: torch.Tensor, H_indices: torch.Tensor, homographies: torch.Tensor) -> torch.Tensor:
        """Упрощенная векторизованная проекция точек"""
        n_points = xy.shape[0]

        # Добавляем координату z=1
        ones = torch.ones(n_points, 1, device=self.device, dtype=torch.float32)
        xy_homo = torch.cat([xy, ones], dim=1)  # [n_points, 3]

        # Вместо bmm используем более простой подход с advanced indexing
        projected_points = []

        # Группируем точки по индексам гомографий для эффективности
        unique_indices = torch.unique(H_indices)

        for idx in unique_indices:
            mask = H_indices == idx
            points_subset = xy_homo[mask]  # [n_subset, 3]
            H = homographies[:, :, idx]    # [3, 3]

            # Простое матричное умножение
            projected_subset = (H @ points_subset.T).T  # [n_subset, 3]

            # Нормализация
            z = projected_subset[:, 2:3] + 1e-8
            projected_2d = projected_subset[:, :2] / z

            projected_points.append((mask, projected_2d))

        # Собираем результаты в правильном порядке
        result = torch.zeros(n_points, 2, device=self.device, dtype=torch.float32)
        for mask, projected_2d in projected_points:
            result[mask] = projected_2d

        return result

    def reprojection_mse_vectorized(self) -> torch.Tensor:
        """Полностью векторизованная версия репроекционной ошибки"""
        homographies = self.get_homographies()  # [3, 3, n_images]

        # Батчевые параметры камеры и дисторсии
        camera_matrix = self.get_camera_matrix_batch(1)
        dist_coeffs = self.get_distortion_params_batch(1)

        # Исправление дисторсии для всех точек сразу
        xy_i_corrected = undistort_points(
            self.xy_i_all.unsqueeze(0),  # [1, n_points, 2]
            camera_matrix,  # [1, 3, 3]
            dist_coeffs,    # [1, 14]
            num_iters=5     # Уменьшено для скорости
        ).squeeze(0)  # [n_points, 2]

        xy_j_corrected = undistort_points(
            self.xy_j_all.unsqueeze(0),
            camera_matrix,
            dist_coeffs,
            num_iters=5
        ).squeeze(0)

        # Векторизованная проекция всех точек
        xy_i_warped = self.project_vectorized(xy_i_corrected, self.i_indices, homographies)
        xy_j_warped = self.project_vectorized(xy_j_corrected, self.j_indices, homographies)

        # Вычисление ошибки для всех точек сразу
        diff = xy_i_warped - xy_j_warped  # [n_points, 2]
        reproj_errors = torch.sum(diff ** 2, dim=1)  # [n_points]

        return reproj_errors.mean()

    def bundle_adjustment(self, lr_f=1e3, lr_c=1e1, lr_k1=1e-2, lr_k2=1e-4, lr_k3=1e-6, lr_p=1e-3,
                          h_gamma=0.95, d_gamma=0.3,
                          max_iter=5000, plot=False, verbose='full') -> float:

        homographies_optimizer = torch.optim.Adam([
            {'params': [self.a_params], 'lr': 1e-3},
            {'params': [self.b_params], 'lr': 1e-0},
            {'params': [self.c_params], 'lr': 1e-6}
        ], betas=(0.9, 0.999), eps=1e-8)

        # Создаем параметры для дисторсионного оптимизатора только для незамороженных параметров
        distortion_params = [
            {'params': [self.f], 'lr': lr_f},
            {'params': [self.k1], 'lr': lr_k1},
            {'params': [self.k2], 'lr': lr_k2},
            {'params': [self.k3], 'lr': lr_k3},
        ]

        # Добавляем параметры главной точки только если они не заморожены
        if not self.freeze_principal_point:
            distortion_params.append({'params': [self.c_xy], 'lr': lr_c})

        # Добавляем тангенциальные параметры только если они не заморожены
        if not self.freeze_tangential:
            distortion_params.append({'params': [self.p], 'lr': lr_p})

        distortion_optimizer = torch.optim.Adam(distortion_params, betas=(0.9, 0.999), eps=1e-8)

        homographies_scheduler = StepLR(homographies_optimizer, step_size=1000, gamma=h_gamma)
        distortion_scheduler = StepLR(distortion_optimizer, step_size=1000, gamma=d_gamma)

        with torch.no_grad():
            initial_loss = torch.sqrt(self.reprojection_mse_vectorized()).item()
            print(f"Initial error: {initial_loss}")

        # Для early stopping
        best_loss = float('inf')
        patience = 500
        patience_counter = 0

        iteration_history = [0]
        loss_history = [initial_loss]

        for iteration in range(max_iter):
            homographies_optimizer.zero_grad()
            distortion_optimizer.zero_grad()

            loss = self.reprojection_mse_vectorized()
            loss.backward()

            homographies_optimizer.step()
            distortion_optimizer.step()

            homographies_scheduler.step()
            distortion_scheduler.step()

            current_loss = loss.item()

            # Early stopping
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience:
                    print(f"Early stopping at iteration {iteration}")
                    break

            if (iteration + 1) % 200 == 0:
                current_error = torch.sqrt(loss).item()
                iteration_history.append(iteration + 1)
                loss_history.append(current_error)
                if verbose == 'full':
                    print(f"Iteration: {iteration + 1}, Loss: {current_error:.6f}")

        with torch.no_grad():
            final_loss = torch.sqrt(self.reprojection_mse_vectorized()).item()
            print(f"Optimized error: {final_loss}")

        # Выводим финальные параметры с пометками о заморозке
        cx_str = f"cx={self.c_xy[0].item():.2f}" + (" (frozen)" if self.freeze_principal_point else "")
        cy_str = f"cy={self.c_xy[1].item():.2f}" + (" (frozen)" if self.freeze_principal_point else "")
        p1_str = f"p1={self.p[0].item():.6f}" + (" (frozen)" if self.freeze_tangential else "")
        p2_str = f"p2={self.p[1].item():.6f}" + (" (frozen)" if self.freeze_tangential else "")

        print(
            f"Final params:\n"
            f"f={self.f.item():.2f}\n"
            f"{cx_str}\n"
            f"{cy_str}\n"
            f"k1={self.k1.item():.6f}\n"
            f"k2={self.k2.item():.6f}\n"
            f"k3={self.k3.item():.6f}\n"
            f"{p1_str}\n"
            f"{p2_str}"
        )

        # Обновляем данные
        with torch.no_grad():
            homographies = self.get_homographies()

        for i, id in enumerate(self.data.tile_set.order):
            img = self.data.tile_set.images[id]
            img.homography = homographies[:, :, i].cpu().numpy()

        return final_loss


class AffineDistortionOptimizer:
    def __init__(self, device, data: StitchingData, f=10000.0, cx=0.0, cy=0.0, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
                 freeze_principal_point=False, freeze_tangential=False):

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

        # Сохраняем флаги заморозки
        self.freeze_principal_point = freeze_principal_point
        self.freeze_tangential = freeze_tangential

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

        a, b = self.homography_to_tens(homographies)
        with torch.no_grad():
            self.a_params.data = a
            self.b_params.data = b

        self.f = nn.Parameter(torch.tensor([f], device=self.device, dtype=torch.float32, requires_grad=True))

        self.c_xy = nn.Parameter(
            torch.tensor(
                [cx, cy], device=self.device, dtype=torch.float32,
                requires_grad=not freeze_principal_point
            )
        )

        self.k1 = nn.Parameter(torch.tensor([k1], device=self.device, dtype=torch.float32, requires_grad=True))
        self.k2 = nn.Parameter(torch.tensor([k2], device=self.device, dtype=torch.float32, requires_grad=True))
        self.k3 = nn.Parameter(torch.tensor([k3], device=self.device, dtype=torch.float32, requires_grad=True))

        self.p = nn.Parameter(
            torch.tensor(
                [p1, p2], device=self.device, dtype=torch.float32,
                requires_grad=not freeze_tangential
            )
        )

        self._prepare_vectorized_matches(data.matches)

    def _prepare_vectorized_matches(self, matches):
        """Предварительная векторизация всех матчей для ускорения вычислений"""
        i_indices = []
        j_indices = []
        xy_i_all = []
        xy_j_all = []

        for match in matches:
            n_points = len(match.xy_i)
            i_indices.extend([match.i] * n_points)
            j_indices.extend([match.j] * n_points)
            xy_i_all.extend(match.xy_i)
            xy_j_all.extend(match.xy_j)

        # Конвертируем в numpy массивы сначала, затем в тензоры
        i_indices_np = np.array(i_indices, dtype=np.int64)
        j_indices_np = np.array(j_indices, dtype=np.int64)
        xy_i_all_np = np.array(xy_i_all, dtype=np.float32)
        xy_j_all_np = np.array(xy_j_all, dtype=np.float32)

        # Создаем тензоры из numpy массивов (намного быстрее)
        self.i_indices = torch.from_numpy(i_indices_np).to(self.device)
        self.j_indices = torch.from_numpy(j_indices_np).to(self.device)
        self.xy_i_all = torch.from_numpy(xy_i_all_np).to(self.device)
        self.xy_j_all = torch.from_numpy(xy_j_all_np).to(self.device)
        self.total_matches = len(i_indices)

    def get_camera_matrix_batch(self, batch_size=1) -> torch.Tensor:
        """Батчевая версия camera matrix для kornia"""
        # zero = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        one = torch.ones(batch_size, device=self.device, dtype=torch.float32)
        f_batch = self.f.expand(batch_size)
        cx_batch = self.c_xy[0].expand(batch_size)
        cy_batch = self.c_xy[1].expand(batch_size)

        camera_matrix = torch.zeros(batch_size, 3, 3, device=self.device, dtype=torch.float32)
        camera_matrix[:, 0, 0] = f_batch
        camera_matrix[:, 1, 1] = f_batch
        camera_matrix[:, 0, 2] = cx_batch
        camera_matrix[:, 1, 2] = cy_batch
        camera_matrix[:, 2, 2] = one

        return camera_matrix

    def get_distortion_params_batch(self, batch_size=1) -> torch.Tensor:
        """Батчевая версия distortion parameters"""
        dist_coeffs = torch.zeros((batch_size, 14), device=self.device, dtype=torch.float32)
        dist_coeffs[:, 0] = self.k1.expand(batch_size)
        dist_coeffs[:, 1] = self.k2.expand(batch_size)
        dist_coeffs[:, 2] = self.p[0].expand(batch_size)
        dist_coeffs[:, 3] = self.p[1].expand(batch_size)
        dist_coeffs[:, 4] = self.k3.expand(batch_size)
        return dist_coeffs

    def get_homographies(self) -> torch.Tensor:
        """Возвращает все гомографии как тензор [3, 3, n_images]"""
        homographies = torch.stack([
            self.a_params[0],
            self.a_params[1],
            self.b_params[0],
            self.a_params[2],
            self.a_params[3],
            self.b_params[1],
            torch.zeros_like(self.a_params[1]),
            torch.zeros_like(self.a_params[1]),
            torch.ones_like(self.a_params[1])
        ])

        # Вставляем единичную матрицу для референсного изображения
        identity = torch.eye(3, device=self.device, dtype=torch.float32).view(-1, 1)
        homographies = torch.cat([
            homographies[:, :self.reper_idx],
            identity,
            homographies[:, self.reper_idx:]
        ], dim=1)

        return homographies.view(3, 3, self.n_images)

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

        return a_params, b_params

    def project_vectorized(self, xy: torch.Tensor, H_indices: torch.Tensor, homographies: torch.Tensor) -> torch.Tensor:
        """Упрощенная векторизованная проекция точек"""
        n_points = xy.shape[0]

        ones = torch.ones(n_points, 1, device=self.device, dtype=torch.float32)
        xy_homo = torch.cat([xy, ones], dim=1)  # [n_points, 3]

        projected_points = []

        unique_indices = torch.unique(H_indices)

        for idx in unique_indices:
            mask = H_indices == idx
            points_subset = xy_homo[mask]  # [n_subset, 3]
            H = homographies[:, :, idx]    # [3, 3]

            projected_subset = (H @ points_subset.T).T  # [n_subset, 3]

            # не надо скорее всего
            z = projected_subset[:, 2:3] + 1e-8
            projected_2d = projected_subset[:, :2] / z

            projected_points.append((mask, projected_2d))

        result = torch.zeros(n_points, 2, device=self.device, dtype=torch.float32)
        for mask, projected_2d in projected_points:
            result[mask] = projected_2d

        return result

    def reprojection_mse_vectorized(self) -> torch.Tensor:
        """Полностью векторизованная версия репроекционной ошибки"""
        homographies = self.get_homographies()  # [3, 3, n_images]

        # Батчевые параметры камеры и дисторсии
        camera_matrix = self.get_camera_matrix_batch(1)
        dist_coeffs = self.get_distortion_params_batch(1)

        # Исправление дисторсии для всех точек сразу
        xy_i_corrected = undistort_points(
            self.xy_i_all.unsqueeze(0),  # [1, n_points, 2]
            camera_matrix,  # [1, 3, 3]
            dist_coeffs,    # [1, 14]
            num_iters=5     # Уменьшено для скорости
        ).squeeze(0)  # [n_points, 2]

        xy_j_corrected = undistort_points(
            self.xy_j_all.unsqueeze(0),
            camera_matrix,
            dist_coeffs,
            num_iters=5
        ).squeeze(0)

        # Векторизованная проекция всех точек
        xy_i_warped = self.project_vectorized(xy_i_corrected, self.i_indices, homographies)
        xy_j_warped = self.project_vectorized(xy_j_corrected, self.j_indices, homographies)

        # Вычисление ошибки для всех точек сразу
        diff = xy_i_warped - xy_j_warped  # [n_points, 2]
        reproj_errors = torch.sum(diff ** 2, dim=1)  # [n_points]

        return reproj_errors.mean()

    def bundle_adjustment(self, lr_f=1e3, lr_c=1e1, lr_k1=1e-2, lr_k2=1e-4, lr_k3=1e-6, lr_p=1e-3,
                          h_gamma=0.95, d_gamma=0.3,
                          max_iter=5000, plot=False, verbose='full') -> float:

        homographies_optimizer = torch.optim.Adam([
            {'params': [self.a_params], 'lr': 1e-3},
            {'params': [self.b_params], 'lr': 1e-0},
        ], betas=(0.9, 0.999), eps=1e-8)

        distortion_params = [
            {'params': [self.f], 'lr': lr_f},
            {'params': [self.k1], 'lr': lr_k1},
            {'params': [self.k2], 'lr': lr_k2},
            {'params': [self.k3], 'lr': lr_k3},
        ]

        if not self.freeze_principal_point:
            distortion_params.append({'params': [self.c_xy], 'lr': lr_c})

        if not self.freeze_tangential:
            distortion_params.append({'params': [self.p], 'lr': lr_p})

        distortion_optimizer = torch.optim.Adam(distortion_params, betas=(0.9, 0.999), eps=1e-8)

        homographies_scheduler = StepLR(homographies_optimizer, step_size=1000, gamma=h_gamma)
        distortion_scheduler = StepLR(distortion_optimizer, step_size=1000, gamma=d_gamma)

        with torch.no_grad():
            initial_loss = torch.sqrt(self.reprojection_mse_vectorized()).item()
            print(f"Initial error: {initial_loss}")

        # Для early stopping
        best_loss = float('inf')
        patience = 500
        patience_counter = 0

        iteration_history = [0]
        loss_history = [initial_loss]

        for iteration in range(max_iter):
            homographies_optimizer.zero_grad()
            distortion_optimizer.zero_grad()

            loss = self.reprojection_mse_vectorized()
            loss.backward()

            homographies_optimizer.step()
            distortion_optimizer.step()

            homographies_scheduler.step()
            distortion_scheduler.step()

            current_loss = loss.item()

            # Early stopping
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience:
                    print(f"Early stopping at iteration {iteration}")
                    break

            if (iteration + 1) % 200 == 0:
                current_error = torch.sqrt(loss).item()
                iteration_history.append(iteration + 1)
                loss_history.append(current_error)
                if verbose == 'full':
                    print(f"Iteration: {iteration + 1}, Loss: {current_error:.6f}")

        with torch.no_grad():
            final_loss = torch.sqrt(self.reprojection_mse_vectorized()).item()
            print(f"Optimized error: {final_loss}")

        # Выводим финальные параметры с пометками о заморозке
        cx_str = f"cx={self.c_xy[0].item():.2f}" + (" (frozen)" if self.freeze_principal_point else "")
        cy_str = f"cy={self.c_xy[1].item():.2f}" + (" (frozen)" if self.freeze_principal_point else "")
        p1_str = f"p1={self.p[0].item():.6f}" + (" (frozen)" if self.freeze_tangential else "")
        p2_str = f"p2={self.p[1].item():.6f}" + (" (frozen)" if self.freeze_tangential else "")

        print(
            f"Final params:\n"
            f"f={self.f.item():.2f}\n"
            f"{cx_str}\n"
            f"{cy_str}\n"
            f"k1={self.k1.item():.6f}\n"
            f"k2={self.k2.item():.6f}\n"
            f"k3={self.k3.item():.6f}\n"
            f"{p1_str}\n"
            f"{p2_str}"
        )

        # Обновляем данные
        with torch.no_grad():
            homographies = self.get_homographies()

        for i, id in enumerate(self.data.tile_set.order):
            img = self.data.tile_set.images[id]
            img.homography = homographies[:, :, i].cpu().numpy()

        return final_loss
