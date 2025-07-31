from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar, final, overload
from typing_extensions import Self

_C = TypeVar("_C", bound=Callable[..., Any])
_T = TypeVar("_T")

class ReferenceType(Generic[_T]):  # "weakref"
    __callback__: Callable[[Self], Any]
    def __new__(cls, o: _T, callback: Callable[[Self], Any] | None = ..., /) -> Self: ...

ref = ReferenceType

@final
class CallableProxyType(Generic[_C]):  # "weakcallableproxy"
    def __eq__(self, value: object, /) -> bool: ...
    def __getattr__(self, attr: str) -> Any: ...
    __call__: _C
    __hash__: ClassVar[None]  # type: ignore[assignment]

# Return CallableProxyType if object is callable, ProxyType otherwise
@overload
def proxy(object: _C, callback: Callable[[_C], Any] | None = None, /) -> CallableProxyType[_C]: ...
@overload
def proxy(object: _T, callback: Callable[[_T], Any] | None = None, /) -> Any: ...
