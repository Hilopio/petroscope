import cv2
import numpy as np
from classes import StitchingData, Panorama

borderValue = 0.0


def make_collage(data: StitchingData, use_gains=False) -> Panorama:
    """
    Create a collage-style panorama from optimized image data.

    Args:
        optimize_data: OptimizeData object containing image paths and optimized transformations.

    Returns:
        PanoramaData: Data object representing the stitched collage panorama.
    """
    images = []
    homographies = []
    for id in data.tile_set.order:
        img = data.tile_set.images[id]
        if use_gains:
            images.append(img.image_compensated)
        else:
            images.append(img.image)
        homographies.append(img.homography)

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

    return Panorama(panorama, None)


def make_collage_with_inliers(
    data: StitchingData,
    use_gains=False,
    borderValue=0,
    line_color=(0, 255, 0),  # BGR format for OpenCV (зеленый по умолчанию)
    line_thickness=10,
    point_color=(255, 0, 0),  # BGR format for OpenCV (синий по умолчанию)
    point_radius=5,
    point_thickness=-1  # -1 для заливки, положительное число для контура
) -> Panorama:
    """
    Создает коллаж панорамы с визуализацией инлайеров.

    Args:
        data (StitchingData): Данные для сшивки
        use_gains (bool): Использовать ли коррекцию усиления
        borderValue: Значение границы для заполнения
        line_color (tuple): Цвет линий в формате BGR (B, G, R)
        line_thickness (int): Толщина линий
        point_color (tuple): Цвет точек в формате BGR (B, G, R)
        point_radius (int): Радиус точек
        point_thickness (int): Толщина контура точек (-1 для заливки)

    Returns:
        Panorama: Объект панорамы с визуализированными инлайерами
    """
    images = []
    homographies = []

    # Собираем изображения и гомографии
    for id in data.tile_set.order:
        img = data.tile_set.images[id]
        if use_gains:
            images.append(img.image_compensated)
        else:
            images.append(img.image)
        homographies.append(img.homography)

    w, h = data.panorama_size
    panorama = np.full(shape=(h, w, 3), fill_value=borderValue, dtype='float32')

    # Создаем панораму
    for image, H in zip(images, homographies):
        cv2.warpPerspective(
            image,
            H,
            (w, h),
            dst=panorama,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

    # Конвертируем в uint8 для рисования (OpenCV работает с uint8 для рисования)
    panorama_vis = (panorama.clip(0, 1) * 255).astype(np.uint8)

    # Рисуем инлайеры
    for match in data.matches:
        i = match.i
        j = match.j

        id_i = data.tile_set.order[i]
        id_j = data.tile_set.order[j]

        # Получаем гомографии для соответствующих изображений
        H_i = data.tile_set.images[id_i].homography
        H_j = data.tile_set.images[id_j].homography

        # Трансформируем точки в координаты панорамы
        # Добавляем единичные координаты для гомогенной трансформации
        points_i_homo = np.column_stack([match.xy_i, np.ones(len(match.xy_i))])
        points_j_homo = np.column_stack([match.xy_j, np.ones(len(match.xy_j))])

        # Применяем гомографии
        points_i_transformed = (H_i @ points_i_homo.T).T
        points_j_transformed = (H_j @ points_j_homo.T).T

        # Нормализуем гомогенные координаты
        points_i_transformed = points_i_transformed[:, :2] / points_i_transformed[:, 2:3]
        points_j_transformed = points_j_transformed[:, :2] / points_j_transformed[:, 2:3]

        # Рисуем линии между соответствующими точками
        for pt_i, pt_j in zip(points_i_transformed, points_j_transformed):
            pt_i = tuple(map(int, pt_i))
            pt_j = tuple(map(int, pt_j))

            # Проверяем, что точки находятся в пределах изображения
            if (
                0 <= pt_i[0] < w and 0 <= pt_i[1] < h and
                0 <= pt_j[0] < w and 0 <= pt_j[1] < h
            ):
                cv2.line(panorama_vis, pt_i, pt_j, line_color, line_thickness)

        # Рисуем точки
        for pt_i, pt_j in zip(points_i_transformed, points_j_transformed):
            pt_i = tuple(map(int, pt_i))
            pt_j = tuple(map(int, pt_j))

            # Рисуем точки только если они в пределах изображения
            if 0 <= pt_i[0] < w and 0 <= pt_i[1] < h:
                cv2.circle(panorama_vis, pt_i, point_radius, point_color, point_thickness)
            if 0 <= pt_j[0] < w and 0 <= pt_j[1] < h:
                cv2.circle(panorama_vis, pt_j, point_radius, point_color, point_thickness)

    # Конвертируем обратно в float32 в диапазоне [0, 1]
    panorama_final = panorama_vis.astype(np.float32) / 255.0

    return Panorama(panorama_final, data.canvas)


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
