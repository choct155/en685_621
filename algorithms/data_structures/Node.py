from typing import Optional
from numbers import Rational

class Node:
    
    def __init__(self, value: Optional[Rational]):
        self.value = value