import os
from img_sudoku_parser import main_parser
import json
import time
import pyautogui
import random as r
import webbrowser
from solvers.killer_dlx import solve_killer_exact_cover
import pickle

BOARD_X = 428  # пикселей от левого края экрана до левого края поля (ячейка (0,0))
BOARD_Y = 177  # пикселей от верха экрана до верхнего края поля (ячейка (0,0))
CELL_SIZE = 75  # например, 50 px ширина и высота каждой «ячейки» в поле


def find_partitions(n: int, k: int) -> list[list[int]]:
    result = []

    if n < sum(range(1, k + 1)) or n > sum(range(10 - k, 10)):
        return []

    def backtrack(start: int, path: list[int], current_sum: int):
        if len(path) == k:
            if current_sum == n:
                result.append(path[:])
            return
        for i in range(start, 10):
            if current_sum + i > n:
                continue
            path.append(i)
            backtrack(i + 1, path, current_sum + i)
            path.pop()

    backtrack(1, [], 0)
    return result


def find_option_intersection(options: list[list[int]]):
    res = set(list(range(1, 10)))
    for option in options:
        res &= set(option)
    return list(res)


def click_cell_and_type(i: int, j: int, digit: int):
    """
    Перемещаем указатель мыши в центр клетки (i,j) и печатаем digit (1–9).
    i, j: индексы строки и столбца (от 0 до 8).
    digit: число 1..9.
    """

    # Вычисляем экранные координаты центра клетки:

    x = BOARD_X + j * CELL_SIZE + CELL_SIZE // 2 + r.randrange(-10, 11)
    y = BOARD_Y + i * CELL_SIZE + CELL_SIZE // 2 + r.randrange(-10, 11)

    # Перемещаем мышь и кликаем (левая кнопка):
    pyautogui.moveTo(x, y, duration=0.02 + 0.1 * r.random())  # duration небольшая (0.1с), чтобы было плавнее
    pyautogui.click()

    # Небольшая пауза, чтобы успел сработать фокус на клетке
    time.sleep(0.01 + 0.01 * r.random())

    # Печатаем цифру (она попадёт в поле ввода)
    pyautogui.press(str(digit))

    # Ещё одна микро-пауза (опционально), чтобы ввод успел «приняться» на сайте
    time.sleep(0.01 + 0.1 * r.random())


def fill_board_smart(id: int, solution: list[list[int]], cage_contents: dict[str, dict]):
    """
    Заполняет доску по решению, начиная с самых простых клеток — те, которые входят
    в наименьшие cage. Использует cage_data вида: {(i,j): {"idx": cage_id, "sum": s}}

    Аргументы:
    - solution: готовая решённая доска 9×9
    - cage_data: информация о клетках и cage, как в исходном JSON
    """
    from collections import defaultdict

    # 1. Построим: cage_id → размер (число клеток)
    cage_size = {}
    for key in cage_contents:
        cage_id = cage_contents[key]["idx"]
        cage_size[cage_id] = cage_size.get(cage_id, 0) + 1

    # 2. Соберём все заполненные клетки вместе с их приоритетом
    cells = []
    for i in range(9):
        for j in range(9):
            digit = solution[i][j]
            if 1 <= digit <= 9:
                cage_key = ','.join(map(str, [i, j]))
                cage_info = cage_contents.get(cage_key)
                if cage_info:
                    cage_id = cage_info["idx"]
                    cage_sum = cage_info["sum"]
                    abs_sum_delta = abs(cage_sum - 27)
                    # priority = 1 / (abs(cage_size[cage_id] - 6.5) * (1 if abs_sum_delta == 0 else abs_sum_delta))
                    priority = len(find_partitions(cage_sum, cage_size[cage_id]))
                else:
                    cage_id = 0
                    cage_sum = 0
                    priority = 4  # если клетка не входит ни в один cage
                cells.append(((i, j), digit, priority, cage_id, cage_sum))

    # 3. Сортируем по приоритету (меньше — проще)
    cells.sort(key=lambda x: (x[2], x[4], r.random()))

    # 4. Вводим на доску
    for (i, j), digit, *_ in cells:
        click_cell_and_type(i, j, digit)

    print(f"#{id}: Ввод завершён.")


def procede_with_autocompletion(id: int):
    webbrowser.open(f"https://www.dailykillersudoku.com/puzzle/{id}")
    pyautogui.hotkey('alt', 'tab')
    time.sleep(4)
    pyautogui.click(900, 800)
    pyautogui.press('down', 14, interval=0.1)
    pyautogui.click(770, 600)  # click continue
    pyautogui.click(113, 244)  # restart
    time.sleep(0.5)
    pyautogui.click(1113, 552)  # confirm restart
    time.sleep(1)
    fill_board_smart(id, res, cage_data)
    time.sleep(1)
    pyautogui.click(1160, 550)
    time.sleep(1)
    pyautogui.hotkey('ctrl', 'w')
    time.sleep(1)
    pyautogui.hotkey('alt', 'tab')
    time.sleep(1)
    pyautogui.click(980, 610)
    time.sleep(1)


with open("solved_on_site.txt", 'r') as res_storage:
    data = json.load(res_storage)

for i in range(26566, 26568):
    strkey = str(i)
    if strkey in data:
        is_solved = data[strkey] == 'True'
        if is_solved:
            print(f"#{i}: Уже решен ранее")
            continue

    filename = fr"result_txts\result_{i}.txt"
    res = True
    if not os.path.exists(filename):
        res = main_parser(i, False)
    if not res:
        data[strkey] = 'False'
        continue

    cage_data = json.load(open(filename, "r", encoding="utf-8"))

    solution_name = fr"solutions\solution_{i}.txt"

    if not os.path.exists(solution_name):
        print(f"#{i}: Начинаю решение...")
        res = solve_killer_exact_cover(filename)

        if res is None:
            print(f"#{i}: Не решено :(")
            data[strkey] = 'False'
            continue
        print(f"#{i}: Успешно решено - сохраняю...")
        pickle.dump(res, open(solution_name, 'wb'))

    else:
        print(f"#{i}: Решение найдено на диске, загружаю...")
        res = pickle.load(open(solution_name, 'rb'))

    procede_with_autocompletion(i)
    data[strkey] = 'True'

with open("solved_on_site.txt", 'w') as res_storage:
    res_storage.write(json.dumps(data))
