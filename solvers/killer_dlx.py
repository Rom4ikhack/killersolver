import json
import itertools
import copy
import time
import sys


def load_cages(json_path: str):
    """
    Считываем JSON вида {"r,c": {"idx":i, "sum":s}, …}
    Возвращаем:
      cage_of[(r,c)] = idx,
      cage_cells[idx] = [(r1,c1), …],
      cage_sum[idx] = s,
      N_cages = максимум idx.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cage_of = {}
    cage_cells = {}
    cage_sum = {}
    for key, info in data.items():
        r, c = map(int, key.split(","))
        idx, s = info["idx"], info["sum"]
        cage_of[(r, c)] = idx
        cage_sum[idx] = s
        cage_cells.setdefault(idx, []).append((r, c))
    n_cages = max(cage_cells.keys())
    return cage_of, cage_cells, cage_sum, n_cages


def generate_valid_sets(cage_cells: dict[int, list[tuple]], cage_sum: dict[int, int]):
    """
    Для каждого cage_idx генерируем все k-комбинации цифр (1..9),
    сумма которых = target_sum. Возвращаем valid_sets[idx] = [tuple,...].
    """
    valid_sets = {}
    for idx, cells in cage_cells.items():
        k = len(cells)
        s = cage_sum[idx]
        combos = [combo for combo in itertools.combinations(range(1, 10), k) if sum(combo) == s]
        valid_sets[idx] = combos
        if not combos:
            print(f"⚠ У cage {idx} (размер={k}, сумма={s}) нет допустимых наборов", file=sys.stderr)
    return valid_sets


def build_exact_cover_structs(cage_of, cage_cells, valid_sets):
    """
    Строим структуры для Exact Cover:
      - row_to_cols[row_id] = set(col_id),
      - col_to_rows[col_id] = set(row_id).
    И список row_entries[row_id] = list of ((r,c), digit).
    Также возвращаем total_cols (число столбцов).
    """

    # --- 1) Назначаем col_id для каждого constraint-ключа ---
    col_index = {}
    next_col = 0

    # A) CellConstraint: "cell_r_c"
    for r in range(9):
        for c in range(9):
            col_index[("cell", r, c)] = next_col
            next_col += 1

    # B) RowConstraint: "row_r_d"
    for r in range(9):
        for d in range(1, 10):
            col_index[("row", r, d)] = next_col
            next_col += 1

    # C) ColConstraint: "col_c_d"
    for c in range(9):
        for d in range(1, 10):
            col_index[("col", c, d)] = next_col
            next_col += 1

    # D) BlockConstraint: "block_b_d"
    for b in range(9):
        for d in range(1, 10):
            col_index[("block", b, d)] = next_col
            next_col += 1

    # E) CageConstraint: "cage_idx"
    all_cage_idxs = sorted(cage_cells.keys())
    for idx in all_cage_idxs:
        col_index[("cage", idx)] = next_col
        next_col += 1

    total_cols = next_col

    # --- 2) Инициализируем пустые множества для col_to_rows ---
    col_to_rows = {col_id: set() for col_id in range(total_cols)}

    # --- 3) Будем заполнять row_to_cols и row_entries ---
    row_to_cols = {}
    row_entries = {}  # row_id -> [ ((r,c), d), ... ]

    next_row = 0

    # Перебираем каждую cage
    for idx, cells in cage_cells.items():
        k = len(cells)
        tgt = valid_sets[idx]  # список кортежей длины k, которые суммируют в cage_sum[idx]

        # Если для этой cage нет ни одного набора — мы уже предупредили выше
        for combo in tgt:
            # Генерируем все перестановки combo → различное распределение цифр по клеткам
            for perm in itertools.permutations(combo):
                # row_cols — множество col_id, где в этой строке стоят „1“
                cols = set()
                assignment = []  # будем заполнять row_entries[next_row]
                for (r, c), d in zip(cells, perm):
                    b = (r // 3) * 3 + (c // 3)
                    # 1) cell constraint
                    cols.add(col_index[("cell", r, c)])
                    # 2) row constraint
                    cols.add(col_index[("row", r, d)])
                    # 3) col constraint
                    cols.add(col_index[("col", c, d)])
                    # 4) block constraint
                    cols.add(col_index[("block", b, d)])
                    assignment.append(((r, c), d))
                # 5) cage constraint
                cols.add(col_index[("cage", idx)])

                # Запоминаем
                row_to_cols[next_row] = cols
                row_entries[next_row] = assignment
                for col_id in cols:
                    col_to_rows[col_id].add(next_row)

                next_row += 1

    return row_to_cols, col_to_rows, row_entries, total_cols


def solve_exact_cover(row_to_cols, col_to_rows):
    """
    Реализация Algorithm X (Exact Cover) на Python.
    row_to_cols: dict[row_id] = set(col_id)
    col_to_rows: dict[col_id] = set(row_id)
    Yields: список row_id для каждого полного покрытия.
    """

    def search(solution, row_to_cols, col_to_rows, active_cols):
        # Если нет больше активных столбцов — мы покрыли всё!
        if not active_cols:
            yield list(solution)
            return

        # MRV: выбираем столбец с наименьшим числом строк
        c = min(active_cols, key=lambda col: len(col_to_rows[col]))
        if not col_to_rows[c]:
            return  # конфликт — нет строк для покрытия этого столбца

        for r in list(col_to_rows[c]):
            solution.append(r)

            removed_cols = []
            removed_rows = {}

            # «Cover»: убираем все столбцы, которые входят в row_to_cols[r]
            for j in row_to_cols[r]:
                if j in active_cols:
                    active_cols.remove(j)
                    removed_cols.append(j)
                    removed_rows[j] = col_to_rows[j].copy()
                    # Удаляем строки, которые содержат j, из всех пересекающих столбцов
                    for rr in removed_rows[j]:
                        for k in row_to_cols[rr]:
                            if k != j:
                                col_to_rows[k].discard(rr)
                    col_to_rows[j].clear()

            # Рекурсивный вызов
            yield from search(solution, row_to_cols, col_to_rows, active_cols)

            # Откат (uncover)
            for j in reversed(removed_cols):
                for rr in removed_rows[j]:
                    for k in row_to_cols[rr]:
                        if k != j:
                            col_to_rows[k].add(rr)
                col_to_rows[j] = removed_rows[j].copy()
                active_cols.add(j)

            solution.pop()

    active_cols = set(col_to_rows.keys())
    yield from search([], copy.deepcopy(row_to_cols), copy.deepcopy(col_to_rows), active_cols)


def solve_killer_exact_cover(json_path: str, verbose=False):
    """
    Основная функция: решаем Killer Sudoku с помощью Exact Cover без внешних библиотек.
    Возвращает 9×9 матрицу решения или None, если решения нет.
    Если verbose=True, печатаем время на этапы.
    """
    t0 = time.time()
    cage_of, cage_cells, cage_sum, n_cages = load_cages(json_path)
    valid_sets = generate_valid_sets(cage_cells, cage_sum)
    if verbose:
        print(f"Loaded cages, generated valid sets in {time.time() - t0:.3f}s")

    row_to_cols, col_to_rows, row_entries, total_cols = build_exact_cover_structs(
        cage_of, cage_cells, valid_sets
    )
    if verbose:
        print(f"Built Exact Cover structures: {len(row_to_cols)} rows, {total_cols} cols in {time.time() - t0:.3f}s")

    solution = None
    for sol in solve_exact_cover(row_to_cols, col_to_rows):
        solution = sol
        break
    t1 = time.time()
    if verbose:
        print(f"Solved Exact Cover, time {t1 - t0:.3f}s")

    if solution is None:
        print("‼ No solution found")
        return None

    # Восстанавливаем 9×9
    grid = [[0] * 9 for _ in range(9)]
    for r_id in solution:
        for (r, c), d in row_entries[r_id]:
            grid[r][c] = d
    if verbose:
        print(f"Reconstructed grid in {time.time() - t1:.3f}s")
    return grid
