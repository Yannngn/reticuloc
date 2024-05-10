import os

import cv2
import numpy as np


class ReticulAI:
    def __init__(self):
        pass

    def find(self, image_path: str) -> list[np.ndarray] | None:
        image = self._open_image(image_path)
        gray = self._process_image(image)
        cells = self._detect_cells(gray)

        if cells is None:
            return cells

        return self._cut_cells(image, cells)

    def detect(self, image_path: str):
        cells = self.find(image_path)

        if cells is None:
            return None

        cells = self._isolate_cells(cells)

        return cells

    def display(self, images: list[np.ndarray], side: int = 5, size: int = 50):
        images = self._resize_cells(images, size)

        grids = []
        while images:
            chunk = images[: side**2]
            images = images[side**2 :]
            grids.append(chunk)

        for idx, grid in enumerate(grids):
            grid = self._grid_images(grid)
            cv2.imshow(f"{idx}", grid)

        cv2.waitKey(0)

    def save(
        self,
        images: list[np.ndarray],
        output_path: str = ".",
        prefix: str = "",
        number_of_images: int = 5,
        size: int = 50,
    ):
        os.makedirs(output_path, exist_ok=True)
        images = self._resize_cells(images, size)

        grids = []
        while images:
            chunk = images[:number_of_images]
            images = images[number_of_images:]
            grids.append(chunk)

        for idx, chunk in enumerate(grids):
            grid = self._color_grid(chunk)
            cv2.imwrite(os.path.join(output_path, f"{prefix}_{idx}.png"), grid)

    def _open_image(self, image_path: str) -> np.ndarray:
        # Read the image
        image = cv2.imread(image_path)

        return image

    # PIPELINE PARA DETECÇÃO

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale and apply blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.medianBlur(gray, 3)

    def _detect_cells(self, gray: np.ndarray) -> np.ndarray | None:
        # Detect circles using Hough Transform
        rows = gray.shape[0]
        cells: np.ndarray = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=10,
            param1=60,
            param2=20,
            minRadius=1,
            maxRadius=rows // 6,
        )

        if cells is None:
            return cells

        return np.around(cells).astype(np.int64)

    def _cut_cells(self, image: np.ndarray, cells: np.ndarray) -> list[np.ndarray]:
        cut_cells = []
        for cell in cells[0, :]:
            x, y, r = cell

            x0, y0 = max(0, x - r), max(0, y - r)
            x1, y1 = min(image.shape[1], x + r), min(image.shape[0], y + r)

            cut = image[y0:y1, x0:x1]
            centered = cv2.copyMakeBorder(
                cut,
                abs(min(0, y - r)),
                max(image.shape[0], y + r) - image.shape[0],
                abs(min(0, x - r)),
                max(image.shape[1], x + r) - image.shape[1],
                0,
            )

            cut_cells.append(centered)

        return cut_cells

    def _isolate_cells(self, cut_cells: list[np.ndarray]):
        cells = []
        for cell in cut_cells:
            r = cell.shape[0] // 2

            bool_sq = np.zeros((2 * r, 2 * r, 3), np.uint8)

            bool_sq = cv2.circle(bool_sq, (r, r), r, (255, 255, 255), -1)

            isolated = np.where(bool_sq, cell, 0)

            cells.append(isolated)

        return cells

    def _resize_cells(self, cut_cells: list[np.ndarray], size: int = 50):
        return [cv2.resize(cell, (size, size)) for cell in cut_cells]

    def _grid_images(self, images: list[np.ndarray], size: int = 50):
        num_images = len(images)

        rows = int(np.sqrt(num_images))
        cols = (num_images + rows - 1) // rows

        grid_width = size * cols
        grid_height = size * rows
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        index = 0
        for i in range(rows):
            for j in range(cols):
                if index >= num_images:
                    break
                start_x, start_y = j * size, i * size
                end_x, end_y = start_x + size, start_y + size

                grid_image[start_y:end_y, start_x:end_x] = images[index]
                index += 1

        return grid_image

    def _color_grid(self, images: list[np.ndarray], size: int = 50):
        num_images = len(images)
        color_fn = [
            None,
            self._to_gray,
            self._red_channel,
            self._green_channel,
            self._blue_channel,
        ]

        rows = num_images
        cols = len(color_fn)

        grid_width = size * cols
        grid_height = size * rows
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        for i in range(rows):
            for j, fn in enumerate(color_fn):
                start_x, start_y = j * size, i * size
                end_x, end_y = start_x + size, start_y + size
                if fn is None:
                    grid_image[start_y:end_y, start_x:end_x] = images[i]
                    continue

                grid_image[start_y:end_y, start_x:end_x] = fn(images[i])

        return grid_image

    # PIPELINE PARA CLASSIFICAÇÃO

    def _blue_channel(self, image: np.ndarray) -> np.ndarray:
        b, _, _ = cv2.split(image)

        image = np.array((b,) * 3).transpose((1, 2, 0))

        return image

    def _red_channel(self, image: np.ndarray) -> np.ndarray:
        _, _, r = cv2.split(image)

        image = np.array((r,) * 3).transpose((1, 2, 0))

        return image

    def _green_channel(self, image: np.ndarray) -> np.ndarray:
        _, g, _ = cv2.split(image)

        image = np.array((g,) * 3).transpose((1, 2, 0))

        return image

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = np.array((gray,) * 3).transpose((1, 2, 0))

        return image
