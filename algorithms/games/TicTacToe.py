import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Iterator, Dict
from functools import reduce


@dataclass
class Cell:
    marker: str
    position: np.array

@dataclass
class Move:
    marker: str
    position: np.array

@dataclass
class Result:
    marker: str
    game_over: bool

    def score(self) -> int:
        if (not self.game_over) | (self.marker == " "):
            return 0
        else:
            return -1 if self.marker == "x" else 1


class GameState:

    def __init__(self, state: np.array = np.repeat(" ", 9).reshape(3,3)) -> None:
        self.state = state

    def show(self) -> str:
        show_str: str = f"""
            {self.state[0][0]} | {self.state[0][1]} | {self.state[0][2]}
            ---------
            {self.state[1][0]} | {self.state[1][1]} | {self.state[1][2]}
            ---------
            {self.state[2][0]} | {self.state[2][1]} | {self.state[2][2]}

        """
        return show_str

    def udpate(self, move: Move) -> None:
        i, j = move.position
        self.state[i][j] = move.marker

    @staticmethod
    def get_neighbors(cell: Cell, arr: np.array) -> List[Cell]:
        """Collect all of the valid neighbor cells around a primary cell"""
        nrows, ncols = arr.shape
        i: int = cell.position[0]
        j:int = cell.position[1]
        out: List[Cell] = []

        # TODO: Very tailored to the 3 x 3, but maybe make this more robust later
        for ridx in range(-1, 2):
            for cidx in range(-1, 2):
                r_nb: int = i + ridx
                c_nb: int = j + cidx
                # only collect valid neighbors
                r_nb_in_bounds: bool = (r_nb >= 0) & (r_nb < nrows)
                c_nb_in_bounds: bool = (c_nb >= 0) & (c_nb < ncols)
                not_self: bool = not ((ridx == 0) & (cidx == 0))
                if r_nb_in_bounds & c_nb_in_bounds & not_self:
                    nb: Cell = Cell(arr[r_nb, c_nb], np.array([r_nb, c_nb]))
                    out.append(nb)
        return out

    @staticmethod
    def victory_from_cell(cell: Cell, arr: np.array) -> bool:
        """For a given cell, determines whether or not the game is concluded in a run that starts at the cell"""

        # unmarked cells cannot win
        if cell.marker == " ":
            return False

        nrows, ncols = arr.shape
        neighbors: Iterator[Cell] = filter(lambda c: c.marker == cell.marker, GameState.get_neighbors(cell, arr))
    
        def check_neighbors_neighbor(nb_cell: Cell) -> bool:
            """
            Leverages that victory falls along a common vector direction to restrict next neighbor
            comparisons. 

            TODO: This is also very tailored to the need to only get three in a row
            """
            far_neighbor_idx: np.array = (nb_cell.position - cell.position) + nb_cell.position
            i, j = far_neighbor_idx
            # only consider valid neighbors (once removed)
            i_in_range: bool = (i >= 0) & (i < nrows)
            j_in_range: bool = (j >= 0) & (j < ncols)
            if not (i_in_range & j_in_range):
                return False
            far_neighbor_marker: str = arr[i,j]
            return cell.marker == far_neighbor_marker
    
        # TODO: a more efficient solution would stop upon encountering the first victory (maybe ok to not worry in practice)
        victories_by_neighbor: List[bool] = list(map(check_neighbors_neighbor, neighbors))
        if len(victories_by_neighbor) > 0:
            result: bool = reduce(lambda f, s: f | s, victories_by_neighbor)
            return result
        else: 
            return False


    @staticmethod
    def toCell(i: int, j: int, arr: np.array) -> Cell:
        return Cell(arr[i,j], np.array([i,j]))

    @staticmethod
    def evaluate_game(arr: np.array) -> Result:
        """
        Evaluates the game by considering whether victory has occurred on a cell by cell basis.
        The order of consideration is across the first row (i.e. (0,0) -> (0, 1) -> (0,2)), and
        then down the first column (i.e. (1,0) -> (2,0)). The approach leverages the fact that
        victory is not possible without one of these five cells.
        """

        for i in range(3):
            icell: Cell = GameState.toCell(0, i, arr)
            ivictory: bool = GameState.victory_from_cell(icell, arr)
            if ivictory:
                return Result(icell.marker, True)

        for j in range(1,3):
            jcell: Cell = GameState.toCell(j, 0, arr)
            jvictory: bool = GameState.victory_from_cell(jcell, arr)
            if jvictory:
                return Result(jcell.marker, True)

        # if all cells have markers, the game ends in a draw
        if len(np.argwhere(arr == " ")) == 0:
            return Result(" ", True) # game ends in a draw
        else:
            return Result(" ", False) # game incomplete



class Heuristic:

    def __init__(self):
        super().__init__()

    @staticmethod
    def max_value(node: T3Node) -> int:



class T3Node:

    def __init__(self, game: GameState):
        self.game = game
        self.depth, self.next_marker = self.__depth_and_next_marker()
        self.children = self.get_children()

    def __depth_and_next_marker(self) -> Tuple[int, str]:
        """
        Calculates depth and the next marker. This is done simultaneuously
        to avoid needing to count markers twice. 

        TODO: I am doing this here to make this knowledge dependent only
        upon the GameState, but there is some cost, insofar as I could just
        carry over information at instantiation.
        """
        xs: int = len(np.argwhere(self.game.state == "x"))
        os: int = len(np.argwhere(self.game.state == "o"))
        depth: int = xs + os
        next_m: str = "o" if xs > os else "x"
        return (depth, next_m)


    def get_children(self) -> List["T3Node"]:
        remaining_positions: np.array = np.argwhere(self.game.state == " ")
        if len(remaining_positions) == 0:
            return []
        
        def produce_child(position: np.array) -> T3Node:
            current_state: np.array = np.copy(self.game.state)
            i, j = position
            current_state[i,j] = self.next_marker
            return T3Node(GameState(current_state))

        children: List[T3Node] = list(map(produce_child, remaining_positions))
        return children

    @staticmethod
    def get_node_results(node: "T3Node") -> List[str]:
        root_result: Result = GameState.evaluate_game(node.game.state)
        if root_result.game_over == True:
            return list(root_result.marker)
        else:
            children_results: Iterator[List[str]] = map(T3Node.get_node_results, node.children)
            return reduce(lambda c_i, c_j: c_i + c_j, children_results)

    @staticmethod
    def get_node_score(node: "T3Node") -> int:
        results: List[str] = T3Node.get_node_results(node)
        score_map: Dict[str, int] = {
            "x": -1,
            " ": 0,
            "o": 1
        }
        scores: Iterator[int] = map(lambda result: score_map[result], results)
        node_score: int = reduce(lambda score1, score2: score1 + score2, scores)
        return node_score


    # def evaluate_children(self, node: "T3Node") -> List[List[str]]:
    #     root_result: Result = GameState.evaluate_game(node.game)


# class TicTacToe:
# 
#     def __init__(self, root: T3Node = T3Node(GameState())) -> None:
#         self.tree = self.build_tree(root)
# 
#     def build_tree(self, root: T3Node = T3Node(GameState())) -> T3Node:
#         children: List[T3Node] = root.get_children()
#         if len(children) == 0:
#             return root
#         else:
#             for child in children:
#                 return self.build_tree(child)
