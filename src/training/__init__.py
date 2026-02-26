from .eval import main as eval_main
from .mtl import main as mtl_main, train as train_mtl
from .task1 import main as task1_main, train as train_task1
from .task2 import main as task2_main, train as train_task2

__all__ = [
    "train_task1",
    "train_task2",
    "train_mtl",
    "task1_main",
    "task2_main",
    "mtl_main",
    "eval_main",
]
