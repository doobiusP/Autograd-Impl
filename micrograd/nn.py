import random
from typing import Union

from micrograd.engine import Value

class Module:
    """
    Base class for all components (e.g., Neurons, Layers, Networks) in the model.

    Provides common functionality like zeroing gradients and retrieving parameters.
    """

    def zero_grad(self):
        """
        Sets the gradients of all parameters in the module to zero.
        This is necessary before performing backpropagation to avoid accumulation of gradients.
        """
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> list['Value']:
        """
        Returns the parameters of the module.

        Returns:
            list[Value]: A list of parameters (as `Value` objects) belonging to this module.
        """
        return []


class Neuron(Module):
    """
    Represents a single artificial neuron.

    Attributes:
        w (list[Value]): The weights of the neuron in the form of a row vector
        b (Value): The bias term.
        nonlin (bool): Indicates whether the neuron applies a non-linear activation function (ReLU)
    """

    def __init__(self, nin: int, nonlin: bool = True):
        """
        Initializes the neuron with random weights and a bias term.

        Args:
            nin (int): Number of input features to the neuron.
            nonlin (bool): If True, applies a ReLU activation function; otherwise, linear activation.
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x: list[Value]) -> Value:
        """
        Computes the output of the neuron for a given input.

        Args:
            x (list[Value]): A list of inputs to the neuron.

        Returns:
            Value: The computed output after applying the activation function.
        """
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> list[Value]:
        """
        Retrieves all parameters (weights and bias) of the neuron.

        Returns:
            list[Value]: A list containing the weights and bias of the neuron.
        """
        return self.w + [self.b]

    def __repr__(self) -> str:
        """
        String representation of the neuron.

        Returns:
            str: The type of neuron (ReLU or Linear) and the number of input features.
        """
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """
    Represents a layer of neurons.

    Attributes:
        neurons (list[Neuron]): A list of neurons in the layer.
    """

    def __init__(self, nin: int, nout: int, **kwargs):
        """
        Initializes a layer with a specified number of neurons.

        Args:
            nin (int): Number of input features to each neuron in the layer.
            nout (int): Number of neurons in the layer.
            **kwargs: Additional arguments passed to each neuron (e.g., nonlin).
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> Union[Value, list[Value]]:
        """
        Computes the outputs of the layer for a given input.

        Args:
            x (list[Value]): A list of inputs to the layer.

        Returns:
            Union[Value, list[Value]]: The outputs of the layer. If the layer contains a single neuron,
                                       returns a single Value; otherwise, returns a list of Values.
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Value]:
        """
        Retrieves all parameters (weights and biases) of the neurons in the layer.

        Returns:
            list[Value]: A flattened list of parameters for all neurons in the layer.
        """
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        """
        String representation of the layer.

        Returns:
            str: A list of neurons in the layer.
        """
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """
    Represents a multi-layer perceptron (MLP) neural network.

    Attributes:
        layers (list[Layer]): A list of layers in the network.
    """

    def __init__(self, nin: int, nouts: list[int]):
        """
        Initializes the MLP with a sequence of layers.

        Args:
            nin (int): Number of input features to the first layer.
            nouts (list[int]): List specifying the number of neurons in each layer.
                               The length of the list determines the number of layers.
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i != len(nouts)-1) for i in range(len(nouts))] # Every layer except the last has a ReLU activation

    def __call__(self, x: list[Value]) -> Union[Value, list[Value]]:
        """
        Computes the output of the MLP for a given input.

        Args:
            x (list[Value]): A list of inputs to the network.

        Returns:
            Union[Value, list[Value]]: The output of the network.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        """
        Retrieves all parameters (weights and biases) of the network.

        Returns:
            list[Value]: A flattened list of parameters for all layers in the network.
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        """
        String representation of the MLP.

        Returns:
            str: A description of all layers in the MLP.
        """
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
