from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, Iterator, Tuple, TypeVar, overload
import warnings
from tabulate import tabulate

T = TypeVar("T", bound=Callable[..., Any])


class Registry(Iterable[Tuple[str, Any]]):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """Create an empty registry with the given display name.

        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    # def _do_register(self, name: str, obj: Any) -> None:
    #     assert (
    #         name not in self._obj_map
    #     ), "An object named '{}' was already registered in '{}' registry!".format(
    #         name, self._name
    #     )
    #     self._obj_map[name] = obj

    def _do_register(self, name: str, obj: Any) -> None:
        """Register an object by name; warn and preserve the existing object on duplicates."""
        if name in self._obj_map:
            warnings.warn(
                f"[{self._name}] An object named '{name}' is already registered. "
                "Keep the existing object and ignore the new one.",
                UserWarning,
                stacklevel=2,
            )
            return
        self._obj_map[name] = obj

    @overload
    def register(self, obj: T) -> T:
        """Register an object directly and preserve its type."""
        ...

    @overload
    def register(self, obj: None = None) -> Callable[[T], T]:
        """Return a decorator that preserves the registered object type."""
        ...

    def register(self, obj: T | None = None) -> T | Callable[[T], T]:
        """
        Register the given object under the name `obj.__name__`.
        Can be used as either a decorator or a direct function call.
        """
        if obj is None:
            # Used as a decorator: @registry.register()
            def deco(func_or_class: T) -> T:
                """Register and return the decorated function or class."""
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # Used as a function call: registry.register(MyClass)
        name = obj.__name__
        self._do_register(name, obj)
        return obj

    def get(self, name: str) -> Any:
        """Return the registered object by name; raise KeyError when the name is absent."""
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))
        return ret

    def __contains__(self, name: str) -> bool:
        """Return whether the registry contains the requested name."""
        return name in self._obj_map

    def __repr__(self) -> str:
        """Return a readable representation of the stored configuration."""
        table_headers = ["Names", "Objects"]
        table = tabulate(self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid")
        return "Registry of {}:\n".format(self._name) + table

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Iterate over registered name/object pairs."""
        return iter(self._obj_map.items())

    __str__ = __repr__
