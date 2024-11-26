from typing import Union, Callable, Tuple, Set

class Value:
    """Stores a single scalar value and its gradient for automatic differentiation."""

    def __init__(self, data: float, _children: Tuple['Value', ...] = (), _op: str = ''):
        """
        Initialize a Value node.

        Args:
            data (float): The scalar value stored in the node.
            _children (Tuple[Value, ...]): The parent nodes that contributed to this node.
            _op (str): The operation that produced this node.
        """
        self.data: float = data
        self.grad: float = 0.0

        # Internal variables used for autograd graph construction
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set['Value'] = set(_children)
        self._op: str = _op  # Operation for debugging or graph visualization.

    def __add__(self, other: Union['Value', float]) -> 'Value':
        """
        Add two Value nodes or a Value node and a float.

        Args:
            other (Union[Value, float]): The other operand for addition.

        Returns:
            Value: A new Value node representing the sum.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other: Union['Value', float]) -> 'Value':
        """
        Multiply two Value nodes or a Value node and a float.

        Args:
            other (Union[Value, float]): The other operand for multiplication.

        Returns:
            Value: A new Value node representing the product.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other: Union[int, float]) -> 'Value':
        """
        Raise a Value node to a power.

        Args:
            other (Union[int, float]): The exponent.

        Returns:
            Value: A new Value node representing the result.
        """
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward() -> None:
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self) -> 'Value':
        """
        Apply the ReLU activation function.

        Returns:
            Value: A new Value node representing ReLU(self.data).
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self) -> None:
        """
        Perform backpropagation to compute gradients for all nodes.
        """
        # Topological ordering of nodes in the computation graph
        topo: list['Value'] = []
        visited: Set['Value'] = set()

        def build_topo(v: 'Value') -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Apply the chain rule in reverse topological order
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self) -> 'Value':
        """
        Negate the Value node.

        Returns:
            Value: A new Value node representing -self.
        """
        return self * -1

    def __radd__(self, other: Union['Value', float]) -> 'Value':
        """
        Right-hand addition (other + self).

        Args:
            other (Union[Value, float]): The other operand.

        Returns:
            Value: A new Value node representing the sum.
        """
        return self + other

    def __sub__(self, other: Union['Value', float]) -> 'Value':
        """
        Subtract a Value node or float from self.

        Args:
            other (Union[Value, float]): The other operand.

        Returns:
            Value: A new Value node representing the difference.
        """
        return self + (-other)

    def __rsub__(self, other: Union['Value', float]) -> 'Value':
        """
        Subtract self from another Value node or float (other - self).

        Args:
            other (Union[Value, float]): The other operand.

        Returns:
            Value: A new Value node representing the difference.
        """
        return other + (-self)

    def __rmul__(self, other: Union['Value', float]) -> 'Value':
        """
        Right-hand multiplication (other * self).

        Args:
            other (Union[Value, float]): The other operand.

        Returns:
            Value: A new Value node representing the product.
        """
        return self * other

    def __truediv__(self, other: Union['Value', float]) -> 'Value':
        """
        Divide self by another Value node or float.

        Args:
            other (Union[Value, float]): The other operand.

        Returns:
            Value: A new Value node representing the quotient.
        """
        return self * other**-1

    def __rtruediv__(self, other: Union['Value', float]) -> 'Value':
        """
        Divide another Value node or float by self (other / self).

        Args:
            other (Union[Value, float]): The other operand.

        Returns:
            Value: A new Value node representing the quotient.
        """
        return other * self**-1

    def __repr__(self) -> str:
        """
        String representation of the Value node.

        Returns:
            str: A string showing the data and gradient of the Value node.
        """
        return f"Value(data={self.data}, grad={self.grad})"
