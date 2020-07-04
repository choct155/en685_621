from typing import Optional, List
from numbers import Rational
from algorithms.data_structures.Node import Node
from algorithms.utils.CollectionOps import foldSeq

class EmptyNode(Node):
    
    def __init__(self) -> None:
        self.value = None
        
    def __str__(self) -> str:
        return "EMPTY"
    
    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: Node) -> bool:
        return True if isinstance(other, EmptyNode) else False
    
    def append(self, value: Rational) -> Node:
        return NonEmptyNode(value)
    
    def sorted_values(self) -> List[Rational]:
        return []
    
        
class NonEmptyNode(Node):
    
    def __init__(self, value: Rational, left: Node = EmptyNode(), right: Node = EmptyNode()) -> None:
        self.value = value
        self.left = left
        self.right = right
        
    def __str__(self) -> str:
        return f"(value: {self.value},  left: {self.left}, right: {self.right})"
    
    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: Node) -> bool:
        same_value: bool = self.value == other.value
        same_left: bool = self.left.__eq__(other.left)
        same_right: bool = self.right.__eq__(other.right)
        return same_value & same_left & same_right

    
    def append(self, value: Rational) -> Node:
        if value <= self.value:
            return NonEmptyNode(self.value, self.left.append(value), self.right)
        else:
            return NonEmptyNode(self.value, self.left, self.right.append(value))
        
    def sorted_values(self) -> List[Rational]:
        sorted_left: List[Rational] = self.left.sorted_values()
        sorted_right: List[Rational] = self.right.sorted_values()
        return sorted_left + [self.value] + sorted_right

    # @staticmethod
    # def init_tree(seq: List[Rational]) -> Node:
    #     def loop(loop_seq: List[Rational], out_tree: Node = EmptyNode()):
    #         if len(loop_seq) == 0:
    #             return out_tree
    #         head, *tail = loop_seq
    #         new_out: NonEmptyNode = out_tree.append(head)
    #         return loop(tail, new_out)
    #     return loop(seq)

    @staticmethod
    def init_tree(seq: List[Rational]) -> Node:
        return foldSeq(seq, EmptyNode(), lambda out, next: out.append(next))
