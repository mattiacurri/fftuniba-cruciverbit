import json
import csv
import ast
import unicodedata
from collections import defaultdict
import time
import pickle
import os

PRINT_PARTIALS = False
PRINT_TO_SCREEN = False

ITALIAN_DICT_PATH = "scripts\\task_2\\italian_words_v2.txt"

_out = None
_grid_out = None

def tprint(*args, **kwargs):
    if PRINT_TO_SCREEN:
        print(*args, **kwargs)
    print(*args, **kwargs, file=_out)

def write_grid_repr(grid):
    print(str(grid), file=_grid_out)

def read_grids_empty(path):
    with open(path, encoding="utf-8") as f:
        return [ast.literal_eval(line.strip()) for line in f if line.strip()]

def read_clues_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def read_candidates_csv(path):
    by_xw = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            cid = int(row["num_cruciverba"])
            idx = int(row["num"])
            cand_list = [c.strip() for c in row["candidates"].split(";") if c.strip()]
            conf_list = [x for x in row["confidence_scores"].strip("[]").split(";")]
            conf_f = []
            for s in conf_list:
                s = s.strip()
                try:
                    conf_f.append(float(s))
                except:
                    conf_f.append(float("-inf"))
            if len(conf_f) < len(cand_list):
                conf_f += [float("-inf")] * (len(cand_list) - len(conf_f))
            elif len(conf_f) > len(cand_list):
                conf_f = conf_f[:len(cand_list)]
            cand_conf = list(zip(cand_list, conf_f))
            by_xw[cid].append((idx, cand_conf))

    out = {}
    for cid, items in by_xw.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        out[cid] = [cand for (_i, cand) in items_sorted]
    return out

def _strip_accents(s: str) -> str:
    nkfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nkfd if not unicodedata.combining(ch))

def load_italian_dictionary(path=ITALIAN_DICT_PATH):
    pickle_path = path + ".pickle"
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    
    by_len = defaultdict(list)
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                w = raw.strip()
                if not w:
                    continue
                w_no_acc = _strip_accents(w)
                w_no_acc = w_no_acc.replace("'", "")
                w_clean = w_no_acc.strip()
                if not w_clean:
                    continue
                if all(ch.isalpha() for ch in w_clean):
                    by_len[len(w_clean)].append(w_clean.upper())
    except FileNotFoundError:
        tprint(f"Dictionary file '{path}' not found.")
        return by_len
    
    with open(pickle_path, "wb") as f:
        pickle.dump(by_len, f)
    return by_len

def print_grid(grid):
    for row in grid:
        tprint("".join(row))
    tprint("-" * 20)

def apply_assignment_to_grid(base_grid, assignment, clues):
    grid = [row[:] for row in base_grid]
    for var, word in assignment.items():
        cl = clues[var]
        r, c = cl["row"], cl["col"]
        d = cl["direction"]
        for ch in word:
            grid[r][c] = ch.upper()
            if d == "A":
                c += 1
            else:
                r += 1
    return grid

def clue_cells(cl):
    r, c = cl["row"], cl["col"]
    d, L = cl["direction"], cl["length"]
    out = []
    for _ in range(L):
        out.append((r, c))
        if d == "A":
            c += 1
        else:
            r += 1
    return out

def build_intersections(clues):
    cells_by_var = [clue_cells(cl) for cl in clues]
    cell_to_vars = defaultdict(list)
    for i, cells in enumerate(cells_by_var):
        for pos, rc in enumerate(cells):
            cell_to_vars[rc].append((i, pos))
    inter = defaultdict(list)
    neigh = defaultdict(set)
    for rc, lst in cell_to_vars.items():
        if len(lst) > 1:
            for a in range(len(lst)):
                for b in range(a+1, len(lst)):
                    i, pi = lst[a]
                    j, pj = lst[b]
                    inter[(i,j)].append((pi,pj))
                    inter[(j,i)].append((pj,pi))
                    neigh[i].add(j)
                    neigh[j].add(i)
    return inter, neigh

def can_place_word_on_grid(grid, word, r, c, d):
    R = len(grid); C = len(grid[0])
    for ch in word:
        if not (0 <= r < R and 0 <= c < C):
            return False
        cell = grid[r][c]
        if cell == ".":
            return False
        if cell.isalpha() and cell != ch:
            return False
        if not cell.isalpha() and cell != " ":
            return False
        if d == "A":
            c += 1
        else:
            r += 1
    return True

def compatible(w1, w2, crosses):
    for (p1, p2) in crosses:
        if w1[p1] != w2[p2]:
            return False
    return True

class BacktrackSolver:
    def __init__(self, base_grid, clues, domains, intersections, neighbors,
                 max_nodes_per_start, max_nodes_per_crossword,
                 print_partials,
                 italian_dict=None,
                 max_candidates_csv=100,
                 max_candidates_dict=3):

        self.base_grid = base_grid
        self.clues = clues
        
        self.domains = {v: [(w.upper(), conf) for (w, conf) in ws] for v, ws in domains.items()}
        self.intersections = intersections
        self.neighbors = neighbors

        self.max_nodes_per_start = max_nodes_per_start
        self.max_nodes_per_crossword = max_nodes_per_crossword

        self.print_partials = print_partials

        self.italian_dict = italian_dict or {}
        self.max_candidates_csv = max_candidates_csv
        self.max_candidates_dict = max_candidates_dict

        self.nodes_global = 0
        self.nodes_start = 0
        self.nodes_crossword = 0

        self.stop_due_to_crossword_limit = False

        self.best_assignment = {}
        self.best_filled_words = 0
        self.best_filled_letters = 0
        self.best_csv_words = 0

        self.current_start = None
        self.local_best_assignment = {}
        self.local_best_words = 0
        self.local_best_letters = 0
        self.local_best_csv_words = 0

        R = len(base_grid); C = len(base_grid[0])
        total = 0
        for r in range(R):
            for c in range(C):
                if base_grid[r][c] != ".":
                    total += 1
        self.total_letters = total
        self.total_words = len(clues)

        self.cells_by_var = [clue_cells(cl) for cl in clues]
        self.dict_cache = {}

    def update_best_from_grid(self, assignment, grid, csv_vars=None):
        fw = len(assignment)
        fl = sum(1 for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c].isalpha())
        csv_count = len(csv_vars) if csv_vars is not None else 0

        improve = False
        if csv_count > self.best_csv_words:
            improve = True
        elif csv_count == self.best_csv_words:
            if fw > self.best_filled_words:
                improve = True
            elif fw == self.best_filled_words:
                if fl > self.best_filled_letters:
                    improve = True

        if improve:
            self.best_assignment = assignment.copy()
            self.best_filled_words = fw
            self.best_filled_letters = fl
            self.best_csv_words = csv_count
            tprint(f"Best global partial: csv_words={csv_count}, {fw}/{self.total_words} words, {fl}/{self.total_letters} letters (crossword_nodes: {self.nodes_crossword})")
            if self.print_partials:
                print_grid(grid)

    def update_local_from_grid(self, assignment, grid, csv_vars=None):
        fw = len(assignment)
        fl = sum(1 for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c].isalpha())
        csv_count = len(csv_vars) if csv_vars is not None else 0

        improve = False
        if fw > self.local_best_words:
            improve = True
        elif fw == self.local_best_words:
            if fl > self.local_best_letters:
                improve = True
            elif fl == self.local_best_letters:
                if csv_count > self.local_best_csv_words:
                    improve = True

        if improve:
            self.local_best_assignment = assignment.copy()
            self.local_best_words = fw
            self.local_best_letters = fl
            self.local_best_csv_words = csv_count

    def placeable_words_from_csv(self, v, grid, assignment):
        cl = self.clues[v]
        r, c = cl["row"], cl["col"]
        d = cl["direction"]
        L = cl["length"]

        out = []
        dom = self.domains.get(v, [])
        for w, conf in dom:
            if len(w) != L:
                continue
            ok = True
            for a_var, a_word in assignment.items():
                key = (v, a_var)
                if key in self.intersections and not compatible(w, a_word, self.intersections[key]):
                    ok = False
                    break
            if ok and can_place_word_on_grid(grid, w, r, c, d):
                out.append((w, conf))
            if self.max_candidates_csv is not None and len(out) >= self.max_candidates_csv:
                break
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def _pattern_for_slot(self, v, grid):
        cells = self.cells_by_var[v]
        return "".join(grid[r][c] for (r, c) in cells)

    def _make_cache_key(self, v, pattern, assignment):
        relevant_neighbors = []
        for neigh in self.neighbors.get(v, []):
            if neigh in assignment:
                relevant_neighbors.append((neigh, assignment[neigh]))
        
        relevant_neighbors.sort()
        return (v, pattern, tuple(relevant_neighbors))

    def placeable_words_from_dict(self, v, grid, assignment):
        cl = self.clues[v]
        r0, c0 = cl["row"], cl["col"]
        d = cl["direction"]
        L = cl["length"]

        pattern = self._pattern_for_slot(v, grid)
        
        cache_key = self._make_cache_key(v, pattern, assignment)
        
        if cache_key in self.dict_cache:
            candidate_list = self.dict_cache[cache_key]
        else:
            dict_words = self.italian_dict.get(L, [])
            cand_acc = []
            for w in dict_words:
                ok = True
                for wc, pc in zip(w, pattern):
                    if pc.isalpha() and pc != wc:
                        ok = False
                        break
                if not ok:
                    continue
                
                ok2 = True
                for a_var, a_word in assignment.items():
                    key = (v, a_var)
                    if key in self.intersections and not compatible(w, a_word, self.intersections[key]):
                        ok2 = False
                        break
                if not ok2:
                    continue
                
                cand_acc.append(w)
                if len(cand_acc) > 5000:
                    break
            
            self.dict_cache[cache_key] = cand_acc
            candidate_list = cand_acc

        out = []
        limit = self.max_candidates_dict
        
        for w in candidate_list:
            out.append(w)
            if limit and len(out) >= limit:
                break
        
        return out

    def _place_word_mutating(self, grid, v, w):
        cl = self.clues[v]
        r, c = cl["row"], cl["col"]
        d = cl["direction"]
        rr, cc = r, c
        modified = []
        ok = True
        for ch in w:
            cell = grid[rr][cc]
            if cell == ".":
                ok = False
                break
            if cell.isalpha() and cell != ch:
                ok = False
                break
            if not cell.isalpha() and cell != " ":
                ok = False
                break
            if cell == " ":
                modified.append((rr, cc, " "))
                grid[rr][cc] = ch
            if d == "A":
                cc += 1
            else:
                rr += 1
        if not ok:
            for (r0, c0, old) in modified:
                grid[r0][c0] = old
            return None
        return modified

    def dfs(self, assignment, grid, csv_vars):
        
        if self.stop_due_to_crossword_limit:
            return False

        self.nodes_global += 1
        self.nodes_start += 1
        self.nodes_crossword += 1

        if self.max_nodes_per_crossword is not None and self.nodes_crossword > self.max_nodes_per_crossword:
            self.stop_due_to_crossword_limit = True
            tprint(f"Crossword nodes limit reached: ({self.nodes_crossword}).")
            return False

        if self.max_nodes_per_start is not None and self.nodes_start > self.max_nodes_per_start:
            return False

        self.update_local_from_grid(assignment, grid, csv_vars)
        self.update_best_from_grid(assignment, grid, csv_vars)

        unassigned = [v for v in range(len(self.clues)) if v not in assignment]
        if not unassigned:
            return False

        global_candidates = []
        append_gc = global_candidates.append

        for v in unassigned:
            csv_cands = self.placeable_words_from_csv(v, grid, assignment)
            if csv_cands:
                for w, conf in csv_cands:
                    append_gc(((0, -conf if conf is not None else 0), v, w, "csv"))
            else:
                dict_cands = self.placeable_words_from_dict(v, grid, assignment)
                for w in dict_cands:
                    append_gc(((1, 0), v, w, "dict"))

        if not global_candidates:
            return False

        global_candidates.sort(key=lambda x: x[0])

        for (_prio, v, w, source) in global_candidates:
            if self.stop_due_to_crossword_limit:
                return False

            if self.max_nodes_per_start is not None and self.nodes_start >= self.max_nodes_per_start:
                return False

            if self.max_nodes_per_crossword is not None and self.nodes_crossword >= self.max_nodes_per_crossword:
                self.stop_due_to_crossword_limit = True
                tprint(f"Crossword nodes limit reached: ({self.nodes_crossword}).")
                return False

            modified = self._place_word_mutating(grid, v, w)
            if modified is None:
                continue

            assignment[v] = w
            added_csv = False
            if source == "csv":
                if v not in csv_vars:
                    csv_vars.add(v)
                    added_csv = True

            found_full = self.dfs(assignment, grid, csv_vars)

            if added_csv:
                csv_vars.remove(v)
            if v in assignment:
                del assignment[v]
            for (r0, c0, old) in reversed(modified):
                grid[r0][c0] = old

            if self.stop_due_to_crossword_limit:
                return False

        return False

    def enumerate_starts(self):
        starts = []
        for v in range(len(self.clues)):
            grid0 = [row[:] for row in self.base_grid]
            csv_cands = self.placeable_words_from_csv(v, grid0, {})
            try:
                w, conf = csv_cands[0]
            except IndexError:
                continue
            starts.append((v, w, "csv", conf))
        return starts

    def solve(self):
        
        starts = self.enumerate_starts()

        def start_key(item):
            v, w, src, conf = item
            if src == "csv":
                return (0, -conf if conf is not None else 0)
            else:
                return (1, 0)
        starts.sort(key=start_key)

        seen = set()
        for idx, (v_start, w_start, src, conf) in enumerate(starts):
            key = (v_start, w_start)
            if key in seen:
                continue
            seen.add(key)

            if self.max_nodes_per_crossword is not None and self.nodes_crossword >= self.max_nodes_per_crossword:
                tprint(f"Crossword nodes limit ({self.max_nodes_per_crossword}) reached. Stopping all starts.")
                break
            if self.stop_due_to_crossword_limit:
                break

            self.nodes_start = 0

            self.current_start = f"v{v_start}='{w_start[:30]}'"

            self.local_best_assignment = {}
            self.local_best_words = 0
            self.local_best_letters = 0
            self.local_best_csv_words = 0

            base_grid_copy = [row[:] for row in self.base_grid]
            assignment = {}

            modified = self._place_word_mutating(base_grid_copy, v_start, w_start)
            if modified is None:
                continue

            assignment[v_start] = w_start
            csv_vars = set()
            if src == "csv":
                csv_vars.add(v_start)

            self.update_local_from_grid(assignment, base_grid_copy, csv_vars)
            self.update_best_from_grid(assignment, base_grid_copy, csv_vars)

            found_full = self.dfs(assignment, base_grid_copy, csv_vars)

            if v_start in assignment:
                del assignment[v_start]
            for (r0, c0, old) in reversed(modified):
                base_grid_copy[r0][c0] = old

            if self.stop_due_to_crossword_limit:
                tprint("Not exploring other starts, finished crossword nodes.")
                break

        return self.best_assignment


# change limits based on grid size
# to not search too much in small grids or too little in large grids

def get_dynamic_candidate_limits(n_rows, n_cols):
    n = n_rows

    if n <= 6:
        return {
            "max_nodes_per_start": 100,
            "max_candidates_csv": 100,
            "max_candidates_dict": 3
        }
    elif n <= 8:
        return {
            "max_nodes_per_start": 500,
            "max_candidates_csv": 500,
            "max_candidates_dict": 5
        }
    elif n <= 10:
        return {
            "max_nodes_per_start": 1000,
            "max_candidates_csv": 600,
            "max_candidates_dict": 10
        }
    else:
        return {
            "max_nodes_per_start": 2000,
            "max_candidates_csv": 600,
            "max_candidates_dict": 10
        }



def run_all():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--grids_empty", type=str, default="scripts\\task_2\\test_grids_empty.txt",
                        help="Path to the file containing empty grids.")
    parser.add_argument("--clues_jsonl", type=str, default="scripts\\task_2\\test_cross_clues.jsonl",
                        help="Path to the JSONL file containing clues.")
    parser.add_argument("--candidates", type=str, required=True,
                        help="Path to the CSV file containing candidates.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output debug file.")
    parser.add_argument("--output_grid_file", type=str, required=True,
                        help="Path to the output grid file.")
    parser.add_argument("-n", type=int, default=50,
                        help="Number of crosswords to process (default: 50).")
    parser.add_argument("--no_dict", action="store_true",
                        help="Disable the use of the Italian dictionary.")
    
    args = parser.parse_args()

    OUTPUT_FILE = args.output_file
    OUTPUT_GRID_FILE = args.output_grid_file

    PROCESS_FIRST = args.n

    global _out, _grid_out

    _out = open(OUTPUT_FILE, "w", encoding="utf-8")
    _grid_out = open(OUTPUT_GRID_FILE, "w", encoding="utf-8")


    empty = read_grids_empty(args.grids_empty)
    clues_all = read_clues_jsonl(args.clues_jsonl)
    cand_all = read_candidates_csv(args.candidates)
    if args.no_dict:
        italian_dict = {}
        tprint("Italian dictionary disabled (--no_dict).")
    else:
        italian_dict = load_italian_dictionary(ITALIAN_DICT_PATH)
        tprint("Italian dictionary enabled.")

    N = len(empty)
    Np = min(N, PROCESS_FIRST) if PROCESS_FIRST is not None else N

    total_start = time.time()

    import tqdm
    for i in tqdm.tqdm(range(Np)):
        tprint("\n" + "="*40)
        tprint(f"CRUCIVERBA {i+1}")
        tprint("="*40 + "\n")

        base = [row[:] for row in empty[i]]
        clues = clues_all[i]

        R = len(base)
        C = len(base[0])

        limits = get_dynamic_candidate_limits(R, C)

        domains = {}
        cand_lists = cand_all[i]
        for v in range(len(clues)):
            L = clues[v]["length"]
            cand = [(w, conf) for (w, conf) in cand_lists[v] if len(w) == L]
            cand = cand[:limits["max_candidates_csv"]]
            cand.sort(key=lambda x: x[1], reverse=True)
            domains[v] = cand

        inter, neigh = build_intersections(clues)

        max_nodes_start = limits["max_nodes_per_start"]
        max_nodes_crossword = max_nodes_start * len(clues)

        solver = BacktrackSolver(
            base, clues, domains, inter, neigh,
            max_nodes_start, max_nodes_crossword,
            PRINT_PARTIALS,
            italian_dict=italian_dict,
            max_candidates_csv=limits["max_candidates_csv"],
            max_candidates_dict=limits["max_candidates_dict"]
        )

        start_time = time.time()
        best = solver.solve()
        final = apply_assignment_to_grid(base, best, clues)
        end_time = time.time()
        
        write_grid_repr(final)

        tprint(f"Execution time for this crossword: {end_time - start_time:.2f} seconds")

        tprint("\nFinal grid:")
        print_grid(final)

    total_end = time.time()
    tprint(f"\nTotal execution time: {total_end - total_start:.2f} seconds")

    _out.close()
    _grid_out.close()

if __name__ == "__main__":
    run_all()