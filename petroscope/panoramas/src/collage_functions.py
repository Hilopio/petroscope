import cv2
import numpy as np
from classes import StitchingData, Panorama

borderValue = 0.0


def make_collage(
        data: StitchingData,
        use_gains=False,
        draw_inliers=False,
        draw_connections=False,
) -> Panorama:

    images = [
        data.tile_set.images[id].image_compensated
        if use_gains
        else data.tile_set.images[id].image
        for id in data.tile_set.order
    ]
    homographies = [data.tile_set.images[id].homography for id in data.tile_set.order]

    c = 1 if images[0].ndim == 2 else images[0].shape[2]
    w, h = data.panorama_size
    panorama = np.full(shape=(h, w, c), fill_value=borderValue, dtype='float32')

    for image, H in zip(images, homographies):
        cv2.warpPerspective(
            image,
            H,
            (w, h),
            dst=panorama,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

    if draw_inliers or draw_connections:
        # Цвета в BGR
        inlier_line_color = (0, 255, 0)  # Зеленый для линий инлайеров
        inlier_point_color = (255, 0, 0)  # Синий для точек инлайеров
        connection_line_color = (0, 0, 255)  # Красный для соединений
        inlier_line_thickness = 10
        inlier_point_radius = 5
        inlier_point_thickness = -1  # Заливка
        connection_thickness = 5  # Умеренная толщина

        # Если grayscale, конвертируем в RGB
        if panorama.ndim == 2 or panorama.shape[2] == 1:
            panorama = cv2.cvtColor(panorama, cv2.COLOR_GRAY2RGB)

        # Конвертируем в uint8 для рисования
        panorama_vis = (panorama.clip(0, 1) * 255).astype(np.uint8)

        if draw_inliers:
            # Рисуем инлайеры
            for match in data.matches:
                i = match.i
                j = match.j

                id_i = data.tile_set.order[i]
                id_j = data.tile_set.order[j]

                H_i = data.tile_set.images[id_i].homography
                H_j = data.tile_set.images[id_j].homography

                # Трансформируем точки
                points_i_homo = np.column_stack([match.xy_i, np.ones(len(match.xy_i))])
                points_j_homo = np.column_stack([match.xy_j, np.ones(len(match.xy_j))])

                points_i_transformed = (H_i @ points_i_homo.T).T
                points_j_transformed = (H_j @ points_j_homo.T).T

                points_i_transformed = points_i_transformed[:, :2] / points_i_transformed[:, 2:3]
                points_j_transformed = points_j_transformed[:, :2] / points_j_transformed[:, 2:3]

                for pt_i, pt_j in zip(points_i_transformed, points_j_transformed):
                    pt_i = tuple(map(int, pt_i))
                    pt_j = tuple(map(int, pt_j))

                    if (0 <= pt_i[0] < w and 0 <= pt_i[1] < h and 0 <= pt_j[0] < w and 0 <= pt_j[1] < h):
                        cv2.line(panorama_vis, pt_i, pt_j, inlier_line_color, inlier_line_thickness)

                    if 0 <= pt_i[0] < w and 0 <= pt_i[1] < h:
                        cv2.circle(panorama_vis, pt_i, inlier_point_radius, inlier_point_color, inlier_point_thickness)
                    if 0 <= pt_j[0] < w and 0 <= pt_j[1] < h:
                        cv2.circle(panorama_vis, pt_j, inlier_point_radius, inlier_point_color, inlier_point_thickness)

        if draw_connections:
            # Уникальные пары
            connections = set()
            for match in data.matches:
                i = match.i
                j = match.j
                id_i = data.tile_set.order[i]
                id_j = data.tile_set.order[j]
                if id_i != id_j:
                    connections.add(frozenset({id_i, id_j}))

            for pair in connections:
                id_i, id_j = list(pair)
                # Используем shape от оригинального image (assuming compensated имеет то же)
                img_i = data.tile_set.images[id_i].image
                img_j = data.tile_set.images[id_j].image
                H_i = data.tile_set.images[id_i].homography
                H_j = data.tile_set.images[id_j].homography

                center_i = np.array([[img_i.shape[1] / 2, img_i.shape[0] / 2, 1]])
                center_j = np.array([[img_j.shape[1] / 2, img_j.shape[0] / 2, 1]])

                center_i_transformed = (H_i @ center_i.T).T[:, :2] / (H_i @ center_i.T).T[:, 2:3]
                center_j_transformed = (H_j @ center_j.T).T[:, :2] / (H_j @ center_j.T).T[:, 2:3]

                pt_i = tuple(map(int, center_i_transformed[0]))
                pt_j = tuple(map(int, center_j_transformed[0]))

                if (0 <= pt_i[0] < w and 0 <= pt_i[1] < h and 0 <= pt_j[0] < w and 0 <= pt_j[1] < h):
                    cv2.line(panorama_vis, pt_i, pt_j, connection_line_color, connection_thickness)

        # Конвертируем обратно один раз
        panorama = panorama_vis.astype(np.float32) / 255.0

    return Panorama(panorama, None)


def make_mosaic(data: StitchingData, use_gains=False) -> Panorama:
    """
    Create a collage-style panorama from optimized image data.

    Args:
        optimize_data: OptimizeData object containing image paths and optimized transformations.

    Returns:
        PanoramaData: Data object representing the stitched collage panorama.
    """
    canvas = data.canvas
    images = []
    homographies = []
    for id in data.tile_set.order:
        img = data.tile_set.images[id]
        if use_gains:
            images.append(img.image_compensated)
        else:
            images.append(img.image)
        homographies.append(img.homography)
    w, h = data.panorama_size
    panorama = np.full(shape=(h, w, 3), fill_value=borderValue, dtype='float32')

    for i, (image, H) in enumerate(zip(images, homographies)):
        warped = cv2.warpPerspective(
            image,
            H,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_TRANSPARENT,
            borderValue=borderValue
        )
        panorama = np.where((canvas == i)[..., np.newaxis], warped, panorama)

    return Panorama(panorama, canvas)
