import cv2
import pytesseract
from pdf_to_img_converter import convert
import os
import numpy as np
from tqdm import tqdm
from pdf_sudoku_downloader import get_puzzle, get_pdf_name
import matplotlib.pyplot as plt
import json
import time

main_res_folder = 'result_txts\\'
main_image_folder = 'images\\'
crop_image_subfolder = 'cropped\\'
product_image_subfolder = 'product_images\\'


def measure_black_width(binary_image: np.ndarray) -> int:
    """
    Принимает бинарное изображение (0 = чёрный, 255 = белый).
    Возвращает ширину области, содержащей чёрные пиксели (в пикселях).
    """
    assert len(binary_image.shape) == 2, "Изображение должно быть чёрно-белым (градации серого)"

    # Получаем проекцию по X: для каждого столбца — есть ли там хотя бы один чёрный пиксель
    projection = np.any(binary_image < 128, axis=0)  # логический массив

    # Получаем координаты "чёрных" столбцов
    black_columns = np.where(projection)[0]
    if black_columns.size == 0:
        return 0  # чёрных пикселей нет

    width = black_columns[-1] - black_columns[0] + 1
    return width


def get_config(precision: int):
    return fr'--psm {precision} -c tessedit_char_whitelist=0123456789'


def center_pad(image, target_size=(128, 128), pad_value=255):
    """
    Центрирует изображение на белом фоне заданного размера.

    :param image: исходное изображение (grayscale или цветное)
    :param target_size: (ширина, высота) целевого полотна
    :param pad_value: цвет фона (255 для белого)
    :return: центрированное изображение
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    top = (target_h - h) // 2
    bottom = target_h - h - top
    left = (target_w - w) // 2
    right = target_w - w - left

    if len(image.shape) == 2:  # grayscale
        padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
    else:  # color
        padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3)

    return padded


def is_digit_present(cell_img, pixel_threshold=30, black_pixel_ratio=0.05):
    """
    Проверяет, есть ли в изображении (ячейке) потенциальный текст.
    :param cell_img: Изображение ячейки
    :param pixel_threshold: Порог бинаризации
    :param black_pixel_ratio: Минимальная доля тёмных пикселей для запуска OCR
    :return: True если текст вероятен, иначе False
    """

    if len(cell_img.shape) == 3:
        resimg = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        resimg = cell_img.copy()

    _, binary = cv2.threshold(resimg, pixel_threshold, 255, cv2.THRESH_BINARY_INV)

    black_pixels = np.sum(binary == 255)
    total_pixels = binary.size
    ratio = black_pixels / total_pixels
    return ratio > black_pixel_ratio


def images_fully_match(local_file: np.ndarray, file_to_be_compared: np.ndarray):
    res = cv2.bitwise_xor(local_file, file_to_be_compared)
    _, binary = cv2.threshold(res, 30, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('test.png', binary)
    black_pixels = np.sum(binary == 255)
    total_pixels = binary.size

    return black_pixels / total_pixels <= 140 / 6400


def main_parser(puzzle_id: int, show_graphs: bool):
    # ---------------- 1. Загрузка PDF и обрезка поля ----------------
    res = True
    result_name = f"{main_res_folder}result_{puzzle_id}.txt"
    pdf_name = get_pdf_name(puzzle_id)
    if not os.path.exists(pdf_name):
        pdf_name = get_puzzle(puzzle_id)
        print(f"Файл {pdf_name} успешно загружен")
    else:
        print(f"Файл {pdf_name} уже существует - нет необходимости в скачивании")
    path = pdf_name

    # Конвертируем PDF → PNG
    imgpath, imgname = convert(path, puzzle_id)

    # Загружаем «полное» изображение и сразу переводим в оттенки серого
    img_full = cv2.imread(imgpath)
    gray_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY)

    # GaussianBlur → Canny → findContours, чтобы найти четырехугольник «поле»
    blur = cv2.GaussianBlur(gray_full, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_rect = None
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        if len(approx) == 4 and area > max_area:
            max_area = area
            # сдвигаем внутрь на 6 пикселей, чтобы «обрезать ровно»
            best_rect = (x + 6, y + 6, w - 12, h - 12)

    # name_cropped = imgpath.replace("sudoku", "sudoku_cropped")
    name_cropped = f"{main_image_folder}{crop_image_subfolder}{imgname}"

    if not os.path.exists(name_cropped):
        if best_rect:
            x, y, w, h = best_rect
            cropped = img_full[y: y + h, x: x + w]
            cv2.imwrite(name_cropped, cropped)
            print(f"Игровое поле обрезано:", name_cropped)
        else:
            raise RuntimeError("Граница судоку не найдена.")
    else:
        print(f"Файл {name_cropped} уже существует - нет необходимости в конвертировании в картинку")

    if not os.path.exists(result_name):
        # ---------------- 2. Подготовка обрезанного поля ----------------

        img = cv2.imread(name_cropped)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Заранее «жёсткие» размеры клетки
        cell_h, cell_w = 135, 135

        # Настройка Tesseract: читаем только цифры

        # CLAHE для локального контраста (будет далее)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

        # ---------------- 3. OCR: извлечение (i, j, value) ----------------

        start_x = [1, 139, 276, 414, 552, 690, 828, 965, 1103]
        start_y = [1, 139, 276, 414, 552, 690, 828, 965, 1103]

        cage_sums = []  # здесь будет (row, col, sum_value)
        total_detections, ocr_detections = 0, 0
        total_time_taken, ocr_time_taken = 0, 0

        for idx in tqdm(range(81), desc="OCR-сумм по клеткам"):
            i, j = divmod(idx, 9)

            x1 = start_x[j] + 1
            y1 = start_y[i] + 1
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            cell = gray[y1:y2, x1:x2]
            # cv2.imwrite(f"{main_image_folder}{product_image_subfolder}{i}{j}_1_cell_{imgname}", cell)

            corner_blur_dotted = cv2.medianBlur(cell, 5)[8: 39, 9: 55]
            # cv2.imwrite(f"{main_image_folder}{product_image_subfolder}{i}{j}_2_blur_{imgname}", corner_blur_dotted)
            num_width = measure_black_width(corner_blur_dotted)
            corner = cell[8: 43, 9: 34 + 21 * (num_width > 25)]

            # Проверяем наличие цифры ДО OCR

            if not is_digit_present(corner_blur_dotted):
                continue  # пропускаем, если "угол" пуст

            # corner_inversed = cv2.bitwise_not(corner)
            # cv2.imwrite(f"{main_image_folder}{product_image_subfolder}{i}{j}_4_corner_{imgname}", corner_inversed)

            # centered = cv2.medianBlur(center_pad(cleaned_corner, (90, 90)), 1)
            centered = center_pad(corner, (80, 80))

            text = str()

            # OCR
            current_precision = 6

            while text == '' and current_precision < 12:
                text = pytesseract.image_to_string(centered, config=get_config(current_precision)).strip()
                current_precision += 1
            try:
                value = int(text)
                ocr_detections += 1
                if 1 <= value <= 45:
                    cage_sums.append((i, j, value))
            except ValueError:
                pass

        sum_dict = {(i, j): val for (i, j, val) in cage_sums}

        # --------------- 4. Построение «mask» с пунктирными границами ---------------

        # 4.1. Бинаризация (чёрные линии + цифры → белое)
        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        h, w = bw.shape

        # 4.2. Извлекаем «сплошные» горизонтальные (9×9 / 3×3) линии
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 2, 1))
        horizontals = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_kernel)

        # 4.3. Извлекаем «сплошные» вертикальные линии
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 2))
        verticals = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel)

        # 4.4. Объединяем все эти «толстые» линии в единую маску thick_lines
        thick_lines = cv2.bitwise_or(horizontals, verticals)

        # 4.5. Удаляем thick_lines из bw → остаются только пунктирные штрихи + цифры
        dotted_plus_digits = cv2.bitwise_and(bw, cv2.bitwise_not(thick_lines))

        # 4.6. «Затираем» цифры по ROI: для каждой распознанной «суммы» в sum_dict
        mask_wo_digits = dotted_plus_digits.copy()
        for (i, j), v in sum_dict.items():
            # Точно такие же смещения, что и для OCR:
            x1 = start_x[j] + 10
            y1 = start_y[i] + 10
            x2 = x1 + 54
            y2 = y1 + 50
            # Затираем ВСЕ пиксели в этой области (погашаем цифры)
            mask_wo_digits[y1:y2, x1:x2] = 0

        # 4.7. Усиливаем контраст пунктирных (CLAHE)
        dotted_clahe = clahe.apply(mask_wo_digits)

        # 4.8. «Сшиваем» оставшиеся пунктирные штрихи (Close 5×5, 2 ит.)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        stitched = cv2.morphologyEx(dotted_clahe, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        # 4.9. Утолщаем линии (Dilate 5×5, 2 ит.) → получаем окончательную mask
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.dilate(stitched, dilate_kernel, iterations=2)

        # ---------------- 5. Визуализация всех шагов формирования mask ----------------
        if show_graphs:
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 3, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Обрезанное поле")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(bw, cmap="gray")
            plt.title("Бинаризация (все линии+цифры)")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(dotted_plus_digits, cmap="gray")
            plt.title("Пунктир + цифры (после удаления толстых)")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            plt.imshow(mask_wo_digits, cmap="gray")
            plt.title("Пунктир без цифр (после zатирания ROI)")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(dotted_clahe, cmap="gray")
            plt.title("Пунктир после CLAHE")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(mask, cmap="gray")
            plt.title("Итоговая mask (Close+Dilate)")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        # ------------ 6. Построение графа смежности 9×9 (по mask) ------------

        def has_border(i1, j1, i2, j2, mask):
            """
            Проверяем, есть ли пунктирная граница между центрами клеток
              (i1, j1) и (i2, j2) **с использованием start_x и start_y**.
            Если хоть один пиксель >128, значит – пунктир (###) между ними.
            """
            # Центр клетки (i1, j1):
            x1 = start_x[j1] + cell_w // 2
            y1 = start_y[i1] + cell_h // 2
            # Центр клетки (i2, j2):
            x2 = start_x[j2] + cell_w // 2
            y2 = start_y[i2] + cell_h // 2

            # Линейная интерполяция 10 точек между двумя центрами:
            for step in range(1, 10):
                xx = int(x1 + (x2 - x1) * step / 10)
                yy = int(y1 + (y2 - y1) * step / 10)
                if mask[yy, xx] > 128:
                    return True  # встретили >128 → есть пунктир
            return False  # ни разу не встретили → границы (пути) нет

        # Построим adjacency для всех 9×9 клеток:
        adjacency = {(i, j): [] for i in range(9) for j in range(9)}
        for i in range(9):
            for j in range(9):
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 9 and 0 <= nj < 9:
                        if not has_border(i, j, ni, nj, mask):
                            adjacency[(i, j)].append((ni, nj))

        # ------------ 7. Группировка ячеек в cage’и (DFS) --------------

        visited = set()
        cages = []

        def dfs(cell, group):
            visited.add(cell)
            group.append(cell)
            for neigh in adjacency[cell]:
                if neigh not in visited:
                    dfs(neigh, group)

        for i in range(9):
            for j in range(9):
                if (i, j) not in visited:
                    group = []
                    dfs((i, j), group)
                    cages.append(sorted(group))

        # --------- 8. Привязка OCR-сумм (только к первой клетке группы) ---------

        final_cages = []
        for group in cages:
            first_cell = group[0]
            group_sum = sum_dict.get(first_cell, 0)
            final_cages.append((group, group_sum))

        # --------- 9. Безопасное объединение одиночных «пустых» клеток ---------

        merged = []
        occupied = set()

        for group, s in final_cages:
            if s == 0 and len(group) == 1:
                (i, j) = group[0]
                # Ищем соседнюю клетку, в которой есть ненулевая сумма
                for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if 0 <= ni < 9 and 0 <= nj < 9:
                        for k, (g2, s2) in enumerate(final_cages):
                            if (ni, nj) in g2 and s2 > 0:
                                final_cages[k][0].append((i, j))
                                occupied.add((i, j))
                                break
            else:
                merged.append((group, s))

        for group, s in final_cages:
            # Добавляем каждую клетку, если она не была «объединена» выше
            if any(cell not in occupied for cell in group) and (group, s) not in merged:
                merged.append((group, s))

        # Переформатируем в список словарей с уникальными ячейками
        final_cages = []
        for group, s in merged:
            unique_cells = sorted(set(group))
            final_cages.append({"sum": s, "cells": unique_cells})

        # -------- 9. Визуализация результата (итоговые cage'и) --------

        if show_graphs:
            canvas = img.copy()
            colors = plt.get_cmap("tab20", len(final_cages))

            for idx, cage_info in enumerate(final_cages):
                color = tuple((np.array(colors(idx)[:3]) * 255).astype(int).tolist())
                for (ci, cj) in cage_info["cells"]:
                    x1 = start_x[cj]  # вместо cj*cell_w
                    y1 = start_y[ci]  # вместо ci*cell_h
                    x2 = x1 + cell_w
                    y2 = y1 + cell_h
                    overlay = canvas.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
                i0, j0 = cage_info["cells"][0]
                text_pos = (j0 * cell_w + 12, i0 * cell_h + 28)
                cv2.putText(canvas, str(cage_info["sum"]), text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("Финальные cage'и и их суммы")
            plt.show()

        total_sum = sum(cage_info["sum"] for cage_info in final_cages)
        if total_sum != 405:
            print(f"Ошибка парсинга: сумма всех cage'ей = {total_sum}, а ожидается 405.")
            # with open(result_name, "w", encoding="utf-8") as f:
            #    json.dump({}, f, ensure_ascii=False, indent=2)
            return False
        else:
            print(f"Проверка пройдена: сумма всех cage'ей = {total_sum} = 405.")

        # -------- 10. Текстовый вывод --------

        cell_dict = {}

        # Перебираем все найденные cage
        # idx начинаем с 1, чтобы совпадать с нумерацией «Cage #1, Cage #2, …»
        for idx, cage_info in enumerate(final_cages, start=1):
            cage_sum = cage_info["sum"]
            for (i, j) in cage_info["cells"]:
                # Ключ — координаты клетки, значение — словарь с idx и суммой
                cell_dict[(i, j)] = {"idx": idx, "sum": cage_sum}

        json_ready = {}
        for (i, j), info in cell_dict.items():
            key_str = f"{i},{j}"  # например, (0,1) → "0,1"
            json_ready[key_str] = info

        # 2) Сохраняем в файл в формате JSON:
        with open(result_name, "w", encoding="utf-8") as f:
            json.dump(json_ready, f, ensure_ascii=False, indent=2)

        print(f"Словарь успешно сохранён в {result_name}")
    else:
        print(f"Файл {result_name} уже существует - нет необходимости в парсинге")
    return res
