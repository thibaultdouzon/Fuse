from typing import List
import numpy as np
from pathlib2 import Path


class Pokemon:
    """Stores a Pokemon"""

    def __init__(self, num: int):
        data_path = Path(f"data/{num}.txt")
        if not data_path.is_file():
            self.id = -1
        else:
            with data_path.open() as file:
                info = file.readline().split()
                id, name = info[0], " ".join(info[1:])
                assert(num == int(id))
                self.id = num
                self.name = name
                self.type_l = file.readline().split()
                self.sprite = f"sprites/{num}.png"
        pass

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Pokemon(num={self.id})"


def main():
    p = Pokemon(1)
    print(p)
    pass


if __name__ == '__main__':
    main()
