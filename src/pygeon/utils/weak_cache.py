"""Cache of the methods of the discretizations that depend only on the grid."""

from functools import wraps
from typing import Any, Callable, TypeVar
from weakref import WeakKeyDictionary

T = TypeVar("T")

# the dictionaries of all the decorated methods, to drop the results of a grid
_caches: list[WeakKeyDictionary] = []


def weak_cache(method: Callable[[Any, Any], T]) -> Callable[[Any, Any], T]:
    """
    Cache the result of a method of a discretization for each grid and class of the
    discretization, in place of functools.cache.

    The grid is a weak key, so that its results are released together with it, and
    the class, not the instance, is the inner key, so that the discretizations created
    on the fly share the results while the subclasses inheriting the method keep them
    apart. The result must therefore depend only on the grid and on the class, and must
    not refer to the grid.

    Args:
        method (Callable): A method with the grid as only argument.

    Returns:
        Callable: The method with the cache.
    """
    cache: WeakKeyDictionary = WeakKeyDictionary()
    _caches.append(cache)

    @wraps(method)
    def wrapped(self, sd):
        results = cache.setdefault(sd, {})
        cls = type(self)
        if cls not in results:
            results[cls] = method(self, sd)
        return results[cls]

    return wrapped


def clear_weak_cache(sd: Any) -> None:
    """
    Drop the cached results of a grid from all the methods decorated with
    :func:`pygeon.utils.weak_cache.weak_cache`, e.g. when its geometry is recomputed.

    Args:
        sd (pg.Grid): The grid.
    """
    for cache in _caches:
        cache.pop(sd, None)
