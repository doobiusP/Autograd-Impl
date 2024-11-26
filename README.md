
# **Autograd Implementation: An Educational Deep Learning Framework**

## **Overview**
This project provides a minimalistic implementation of an automatic differentiation (autograd) engine for educational purposes. It supports reverse-mode autodifferentiation over a dynamically constructed Directed Acyclic Graph (DAG) and includes a lightweight neural networks library with a PyTorch-like API.

### Key Features
- Approximately 100 lines of code for the autograd engine.
- A small neural network library (50 lines of code) built on top of the autograd engine.
- Operates over scalar values by decomposing neural network computations into fundamental operations like addition and multiplication.
- Enables the construction of fully functional deep neural networks for tasks such as binary classification.

Despite its simplicity, this implementation is powerful enough to demonstrate the principles of backpropagation and neural network training.


## **Installation and Setup**

### **1. Setting Up a Local Python Environment**
To ensure a clean and controlled environment for running the project, follow these steps:

1. **Create a virtual environment**:
   ```bash
   python -m venv autograd_env
   ```
2. **Activate the environment**:
   - On Windows:
     ```bash
     autograd_env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source autograd_env/bin/activate
     ```

3. **Upgrade pip**:
   ```bash
   python -m pip install --upgrade pip
   ```

### **2. Installing Dependencies**
- Install the required libraries using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Example Usage**

Below is a demonstration of supported operations using the `Value` class from the autograd engine. This example illustrates the construction of a computational graph, performing a forward pass, and computing gradients via backpropagation:

```python
from micrograd.engine import Value

# Define variables
a = Value(-4.0)
b = Value(2.0)

# Perform computations
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f

# Output the forward pass result
print(f'{g.data:.4f}')  # Prints the result of the computation

# Perform backpropagation
g.backward()

# Output the gradients
print(f'{a.grad:.4f}')  # Gradient of g with respect to a
print(f'{b.grad:.4f}')  # Gradient of g with respect to b
```

---

## **Training a Neural Network**

The project includes a demonstration notebook (`demo.ipynb`) that walks through the process of training a 2-layer neural network for binary classification. 

### **Training Steps**
1. **Initialize the Neural Network**:
   - Construct the network using the `micrograd.nn` module, which provides a simple API for defining models.

2. **Define the Loss Function**:
   - Implement a simple SVM "max-margin" binary classification loss function.

3. **Train the Model**:
   - Use Stochastic Gradient Descent (SGD) for optimization.

### **Example Output**
Using a 2-layer neural network with two 16-node hidden layers, we can achieve the following decision boundary on the moon dataset:


---
