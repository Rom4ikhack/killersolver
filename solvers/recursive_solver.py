import time
import logging
import itertools

# Настройка логгера (можно вынести в “main” или в самый верх модуля):
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # или DEBUG, если нужны более подробные сообщения

# Пример базовой конфигурации: выводить логи формата “[уровень][время] сообщение”
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)


class Cell:
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        self.value = 0  # 0 означает «пока не заполнена»
        self.candidates = set(range(1, 10))  # возможные значения 1–9

    def is_solved(self) -> bool:
        return self.value != 0

    def assign(self, val: int):
        """
        Устанавливаем значение value = val и очищаем список кандидатов.
        """
        self.value = val
        self.candidates.clear()
        self.candidates.add(val)

    def eliminate_candidate(self, val: int) -> bool:
        """
        Убираем val из кандидатов.
        Возвращает True, если набор кандидатов изменился (и всё ещё непуст),
        False если val не было в candidates.
        """
        if val in self.candidates:
            self.candidates.remove(val)
            return True
        return False


class Cage:
    def __init__(self, cells: list[Cell], target_sum: int):
        self.cells = cells
        self.target_sum = target_sum

        # Набираем все комбинации (один раз) при инициализации
        self.all_valid_sets = [
            combo for combo in itertools.combinations(range(1, 10), len(cells))
            if sum(combo) == target_sum
        ]
        # Сохраняем текущее число комбинаций, чтобы пропускать не нужные prune
        self._last_combo_count = len(self.all_valid_sets)

    def current_sum(self) -> int:
        return sum(cell.value for cell in self.cells if cell.is_solved())

    def unfilled_cells(self) -> list[Cell]:
        return [cell for cell in self.cells if not cell.is_solved()]

    def possible_combinations(self) -> list[tuple]:
        filled = [cell.value for cell in self.cells if cell.is_solved()]
        remaining_sum = self.target_sum - sum(filled)
        unfilled = len(self.unfilled_cells())

        used = set(filled)
        combos = []
        for combo in self.all_valid_sets:
            # проверяем, что combo содержит все уже установленные цифры:
            if not used.issubset(combo):
                continue
            # убедимся, что остаётся ровно нужное число клеток
            if len(set(combo) - used) == unfilled:
                combos.append(combo)
        return combos

    def prune_candidates(self) -> bool:
        combos = self.possible_combinations()
        if len(combos) == self._last_combo_count:
            return False  # ничего не поменялось
        self._last_combo_count = len(combos)

        allowed_digits = set()
        for combo in combos:
            allowed_digits.update(combo)

        progress = False
        for cell in self.unfilled_cells():
            before = set(cell.candidates)
            cell.candidates.intersection_update(allowed_digits)
            if cell.candidates != before:
                progress = True

        return progress


class Board:
    def __init__(self, cage_data: dict[str, dict]):
        """
        :param cage_data: словарь вида { "i,j": {"idx": cage_index, "sum": cage_sum}, ... }
        """
        # 1) Создаём пустые 9×9 клетки
        self.cells = [[Cell(r, c) for c in range(9)] for r in range(9)]

        # 2) Группируем по cage_index
        from collections import defaultdict
        temp = defaultdict(list)
        sums = {}  # cage_index -> target_sum
        for key, info in cage_data.items():
            i, j = map(int, key.split(','))
            idx = info["idx"]
            s = info["sum"]
            temp[idx].append((i, j))
            sums[idx] = s

        # 3) Создаём объекты Cage
        self.cages: list[Cage] = []
        for idx, coords in temp.items():
            cells_in_cage = [self.cells[i][j] for (i, j) in coords]
            target_sum = sums[idx]
            self.cages.append(Cage(cells_in_cage, target_sum))

        # 4) Подготовка вспомогательных структур:
        #     – для строк, столбцов и блоков (3×3)
        self.rows = [set() for _ in range(9)]  # занятые цифры в каждой строке
        self.cols = [set() for _ in range(9)]  # занятые цифры в каждом столбце
        self.boxes = [set() for _ in range(9)]  # занятые цифры в каждом 3×3‐блоке

    def get_box_index(self, row: int, col: int) -> int:
        """Номер 3×3‐блока: (row//3)*3 + (col//3)"""
        return (row // 3) * 3 + (col // 3)

    def place_number(self, cell: Cell, val: int) -> bool:
        """
        Пытаемся записать val в данную cell.
        Если в строке/столбце/блоке и cage это допустимо,
        фиксируем и возвращаем True. Иначе – False (невозможно).
        """
        r, c = cell.row, cell.col
        b = self.get_box_index(r, c)

        # проверяем классические ограничения:
        if val in self.rows[r] or val in self.cols[c] or val in self.boxes[b]:
            return False

        # проверяем cage:
        #   – сумма текущих заполненных + val не превосходит target_sum
        #   – val не дублирует уже стоящие в cage
        for cage in self.cages:
            if cell in cage.cells:
                curr_sum = cage.current_sum()
                if curr_sum + val > cage.target_sum:
                    return False
                # проверяем, нет ли уже val среди заполненных
                for other in cage.cells:
                    if other is not cell and other.value == val:
                        return False
                break

        # если всё ок, проставляем:
        cell.assign(val)
        self.rows[r].add(val)
        self.cols[c].add(val)
        self.boxes[b].add(val)
        return True

    def remove_number(self, cell: Cell):
        """
        Убираем число из cell (возвращаем клетку в состояние 0 → пусто),
        корректируем множества rows/cols/boxes.
        """
        val = cell.value
        r, c = cell.row, cell.col
        b = self.get_box_index(r, c)
        if val != 0:
            self.rows[r].remove(val)
            self.cols[c].remove(val)
            self.boxes[b].remove(val)
            cell.value = 0
            # Восстанавливаем candidates = {1..9}, но лучше потом делать
            # повторную пересадку кандидатов через constraint propagation.

    def is_solved(self) -> bool:
        """Правило: все 81 клетка заполнены."""
        return all(self.cells[r][c].is_solved() for r in range(9) for c in range(9))

    def find_empty_cell(self) -> Cell | None:
        """
        Находит ещё не заполненную клетку с минимальным количеством candidates
        (меньше всего вариантов) – прием «Minimum Remaining Values» для ускорения backtracking.
        Если все заполнены, возвращает None.
        """
        best = None
        best_len = 10
        for r in range(9):
            for c in range(9):
                cell = self.cells[r][c]
                if not cell.is_solved():
                    l = len(cell.candidates)
                    if l < best_len:
                        best_len = l
                        best = cell
                        if l == 1:
                            return best
        return best

    def propagate_all(self) -> bool:
        progress = False

        # 1) Cage‐based pruning (для всех cage’ей)
        for cage in self.cages:
            if cage.prune_candidates():
                progress = True

        # 2) Row/Col/Box pruning (как раньше)
        for r in range(9):
            used = self.rows[r]
            for c in range(9):
                cell = self.cells[r][c]
                if not cell.is_solved():
                    before = set(cell.candidates)
                    cell.candidates.difference_update(used)
                    if cell.candidates != before:
                        progress = True

        for c in range(9):
            used = self.cols[c]
            for r in range(9):
                cell = self.cells[r][c]
                if not cell.is_solved():
                    before = set(cell.candidates)
                    cell.candidates.difference_update(used)
                    if cell.candidates != before:
                        progress = True

        for b in range(9):
            used = self.boxes[b]
            br = (b // 3) * 3
            bc = (b % 3) * 3
            for dr in range(3):
                for dc in range(3):
                    r = br + dr
                    c = bc + dc
                    cell = self.cells[r][c]
                    if not cell.is_solved():
                        before = set(cell.candidates)
                        cell.candidates.difference_update(used)
                        if cell.candidates != before:
                            progress = True

        return progress


class Solver:
    def __init__(self, board: Board):
        self.board = board

        # Счётчик всех рекурсивных вызовов solve()
        self._calls = 0
        # Текущая глубина (сколько клеток уже поставлено по текущему пути)
        self._depth = 0
        # Время старта, чтобы можно было печатать прогресс по времени
        self._t_start = time.time()
        # Последнее время, когда мы печатали лог
        self._last_log_time = self._t_start
        # Интервал (в секундах) между логами
        self._LOG_INTERVAL = 2.0

    def solve(self) -> bool:
        self._calls += 1
        now = time.time()

        # Если прошло больше чем LOG_INTERVAL секунд с последнего лога
        # или если _calls кратен 10000, выводим статус через logger.info
        if (now - self._last_log_time > self._LOG_INTERVAL) or (self._calls % 10000 == 0):
            filled = sum(
                1 for r in range(9) for c in range(9) if self.board.cells[r][c].is_solved()
            )
            elapsed = now - self._t_start
            logger.info(
                "[%d вызовов, глубина=%d, заполнено=%d/81, время=%.1fс]",
                self._calls, self._depth, filled, elapsed
            )
            self._last_log_time = now

        # 1) Constraint propagation
        while True:
            progress = self.board.propagate_all()
            if not progress:
                break

        # 2) Если доска полностью заполнена, решение найдено
        if self.board.is_solved():
            logger.info("Решение найдено! Всего вызовов solve(): %d, всего времени: %.1fс",
                        self._calls, time.time() - self._t_start)
            return True

        # 3) Ищем пустую клетку с минимальным количеством кандидатов (MRV)
        cell = self.board.find_empty_cell()
        if cell is None:
            # нет пустых, но board.is_solved() всё ещё False → конфликт
            return False

        # Сохраним текущее состояние кандидатов всех незаполненных клеток
        saved_candidates = {
            (other.row, other.col): set(other.candidates)
            for row in self.board.cells for other in row if not other.is_solved()
        }

        # 4) Перебираем кандидатов для этой клетки
        for val in list(cell.candidates):
            # Увеличиваем глубину (мы идём на уровень ниже, ставя val)
            self._depth += 1

            # Пытаемся поставить val в cell
            ok = self.board.place_number(cell, val)
            if ok:
                # Если успешно — рекурсивно вызываем solve()
                if self.solve():
                    return True
                # Иначе: откатываем это значение
                self.board.remove_number(cell)

            # После попытки (успешной или нет) уменьшаем глубину
            self._depth -= 1

            # Восстанавливаем candidates всех пустых клеток
            for (rr, cc), cand in saved_candidates.items():
                self.board.cells[rr][cc].candidates = set(cand)

        # 5) Ни один из кандидатов не дал решения → возвращаем False
        return False
