# coding: utf-8

# # Hello, pytorch
# 
# ![img](https://pytorch.org/tutorials/_static/pytorch-logo-dark.svg)
# 
# [PyTrorch](http://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/) are two of the most commonly used deep learning frameworks. Both of these tools are notable for their ability to compute gradients automatically and do operations on GPU, which can be by orders of magnitude faster than running on CPU. Both libraries serve the same purpose, choosing between them is a matter of preference.
# 
# In this school, we'll use PyTorch for our practical examples.
# 
# We'll start from using the low-level core of PyTorch, and then try out some high-level features.

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import torch
print(torch.__version__)

"""
# In[3]:


# numpy world

x = np.arange(16).reshape(4, 4)

print(f"X :\n{             x}")
print(f"X.shape : {        x.shape}")
print(f"add 5 :\n{         x + 5}")
print(f"X*X^T :\n{         x @ x.T}")
print(f"mean over rows :\n{x.mean(axis=-1)}")
print(f"cumsum of cols :\n{x.cumsum(axis=0)}")


# In[4]:


# pytorch world

x = np.arange(16).reshape(4, 4)

x = torch.tensor(x, dtype=torch.float32)  # or torch.arange(0,16).reshape(4,4).to(torch.float32)

print(f"X :\n{             x}")
print(f"X.shape : {        x.shape}")
print(f"add 5 :\n{         x + 5}")
print(f"X*X^T :\n{         x @ x.T}")
print(f"mean over rows :\n{x.mean(axis=-1)}")
print(f"cumsum of cols :\n{x.cumsum(axis=0)}")


# ## NumPy and Pytorch
# 
# As you can notice, pytorch allows you to hack stuff much the same way you did with numpy. The syntax is in some ways compatible with numpy (as in the example above), [to some extent](https://github.com/pytorch/pytorch/issues/38349).
# 
# Though original naming conventions in PyTorch are a bit different, so if you look up documentation to various functions you may notice for example patameter name `dim` instead of `axis`.
# 
# Also type conversions need be done explicitly. So operations that cannot be done on current type will raise an error rather than converting the tensor:

# In[5]:


x = torch.arange(5)
#x.mean() # instead try: 
x.to(float).mean()


# In[6]:


x = torch.tensor([1, 2])
#x /= 2.0 # instead try: `
x /= 2
x


# Converting tensors back to numpy:

# In[7]:


x = torch.ones(size=(3, 5))
x.numpy()


# ## Warmup: trigonometric knotwork
# _inspired by [this post](https://www.quora.com/What-are-the-most-interesting-equation-plots)_
# 
# There are some simple mathematical functions with cool plots. For one, consider this:
# 
# $$ x(t) = t - 1.5 * cos( 15 t) $$
# $$ y(t) = t - 1.5 * sin( 16 t) $$
# 

# In[8]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

t = torch.linspace(-10, 10, steps=10000)

# compute x=x(t) and y=y(t) as defined above
# YOUR CODE HERE
raise NotImplementedError()

plt.plot(x.numpy(), y.numpy());


# In[ ]:


assert np.isclose(x[42].item(), -9.2157)
assert np.isclose(y[1990].item(), -4.6997)


# --------

# ## Automatic gradients
# 
# Any self-respecting DL framework must do your backprop for you. Torch handles this with the `autograd` module.
# 
# The general pipeline looks like this:
# * When creating a tensor, you mark it as `requires_grad`:
#     * __```torch.zeros(5, requires_grad=True)```__
#     * __```torch.tensor(np.arange(5), dtype=torch.float32, requires_grad=True)```__
# * Define some differentiable `y = arbitrary_function(x)`
# * Call `y.backward()`
# * Gradients are now available as ```x.grad```
# 
# __Here's a few examples:__

# In[9]:


x = torch.tensor(0., requires_grad=True)
y = torch.sin(x)
y.backward()
x.grad


# In[10]:


# Remember we defined MSE loss with its gradient last week?
# With torch we could've skipped the gradient part:

def MSELoss(y, yhat):
    return ((y - yhat)**2).mean()

y = torch.normal(0., 1., size=(100,))
yhat = torch.normal(0., 1., size=(100,), requires_grad=True)

loss = MSELoss(y, yhat)
loss.backward()

print("Checking autograd result equals to analytical derivative.")
print("    check result:", torch.allclose(yhat.grad, 2 * (yhat - y) / y.shape[0]))


# __Note:__ calling `backward` multiple times accumulates the sum of gradients:

# In[11]:


x = torch.tensor(0., requires_grad=True)
y1 = torch.sin(x)
y2 = torch.sin(x)
#y1 = np.sin(x)
#y2 = np.sin(x)


print("x.grad =", x.grad)
y1.backward()
print("x.grad =", x.grad)
y2.backward()
print("x.grad =", x.grad)


# so typically when using it inside learning loop you want to zero the accumulated gradients between the consequtive `backward()` calls, e.g.:

# In[12]:


# Generate linear data:
x = torch.tensor(np.random.uniform(-3, 5, 100), dtype=torch.float32)
y = 2.5 * x + torch.normal(mean=0., std=0.4, size=x.shape)

w = torch.tensor(0., dtype=torch.float32, requires_grad=True)
w_values_history = [
    w.item() # this returns a python number with the value of the tensor
]

for _ in range(100):
    loss = MSELoss(y, w * x)
    loss.backward()
    with torch.no_grad(): #  = "don't calculate gradients in the block below"
        w -= 0.01 * w.grad

    w.grad.zero_() # just for the lulz try commenting this out DON'T FORGET TO ZERO THE GRADIENT, IS THE REMINDER
    w_values_history.append(w.item())

plt.plot(w_values_history);


# Calling `backward()` for a tensor:

# In[13]:


x = torch.linspace(-2, 2, 1000, requires_grad=True)
y = x**2 #A diagonal matrix

# Note: since `x` and `y` are both vectors, the derivative is
# defined in the form of Jacobian dy/dx.
# For such cases `backward()` is implemented to calculate
# the Jacobian multiplied by some other vector that you
# have to provide as an argument e.g. `y.backward(some_vector)` -
# this will return Jacobian dy/dx times `some_vector`.

# Since our Jacobian is diagonal, the following code will return
# per-element derivative values:
y.backward(torch.ones_like(y)) #The derivative

plt.plot(x.detach(), y.detach(), label='y = x^2')
plt.plot(x.detach(), x.grad, label="dy/dx")
plt.legend();
plt.grid(True);


# Second derivative example:

# In[15]:


x = torch.linspace(-2, 2, 1000, requires_grad=True)
y = x**3 

first_derivative, = torch.autograd.grad(
    y, x, torch.ones_like(y),
    create_graph=True # "create_graph" required to be able to calculate derivative of the derivative later
)
second_derivative, = torch.autograd.grad(first_derivative, x, torch.ones_like(first_derivative))

plt.plot(x.detach(), y.detach(), label='y = x^3')
plt.plot(x.detach(), first_derivative.detach(), label="dy/dx")
plt.plot(x.detach(), second_derivative, label="d^2y/dx^2")
plt.legend();
plt.grid(True);


# ---------------------
# ## High-level pytorch
# 
# So far we've been dealing with low-level torch API. While it's absolutely vital for any custom losses or layers, building large neura nets in it is a bit clumsy.
# 
# Luckily, there's also a high-level torch interface with a pre-defined layers, activations and training algorithms. 
# 
# We'll cover them as we go through a simple image recognition problem: classifying handwritten digits.
# 

# In[16]:


from torchvision.datasets import MNIST
from IPython.display import clear_output
ds_train = MNIST(".", train=True, download=True)
ds_test = MNIST(".", train=False, download=True)

X_train, y_train = ds_train.data.reshape(-1, 784).to(torch.float32) / 255., ds_train.targets
X_test, y_test = ds_test.data.reshape(-1, 784).to(torch.float32) / 255., ds_test.targets

clear_output()
print(f"Train size = {len(X_train)}, test_size = {len(X_test)}")


# In[19]:


plt.figure(figsize=(4, 4), dpi=100)
plt.axis('off')
plt.imshow(
    torch.transpose(
        torch.cat(
            [X_train[y_train == c][:10] for c in range(10)], axis=0
        ).reshape(10, 10, 28, 28),
        1, 2
    ).reshape(280, 280)
);


# Let's start with layers.

# In[18]:


from torch import nn
import torch.nn.functional as F


# There's a vast library of popular layers and architectures already built for ya'.
# 
# We'll train a single hidden layer fully connected neural network.

# In[21]:


# GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# create a network that stacks layers on top of each other
model = nn.Sequential(
    nn.Linear(784, 100), # add first "dense" layer with 784 input
                         # units and 100 output units (hidden layer
                         # with 100 neurons).
    nn.ReLU(),
    nn.Linear(100, 10), # "dense" layer with 10 output
                        # units (for 10 classes).
).to(device)


# In[22]:


print("Weight shapes:")
for w in model.parameters():
    print("  ", w.shape)


# In[23]:


# now we can simply call it on our data to get the model output:
prediction = model(X_train.to(device))
print(prediction.shape)


# So far our last layer is linear, with outputs (class scores) possibly taking all values in the range $(-\infty, +\infty)$. For training, we can use the same approach as we did for the multinomial logistic regression: model class probabilities and minimize negative log likelihood.
# 
# Given the ouput $\mathbf{x}=\{x_1,...x_K\}$ of the last layer, the probability modeling can be done with e.g. the **softmax** function:
# $$P(y=C|\mathbf{x})=\frac{\exp(x_C)}{\sum_{C'}\exp(x_{C'})}$$
# 
# This formula may be numerically unstable for large components of $\bf{x}$. To avoid that, you can divide both numerator and denominator by $\exp\left(\max x_{C'}\right)$, and then substitute it into the negative log likelihood.
# 
# In PyTorch, both this steps are combined in the `torch.nn.CrossEntropyLoss`, which can be used like this:

# In[24]:


loss_fn = nn.CrossEntropyLoss()
loss_fn(prediction, y_train.to(device))


# One last thing we need — the optimizer. Optimizers are high level objects that implement all those fancy gradient descent modifications - like Adam, RMSprop and others.
# 
# To use one, simply create that object and pass it the parameters you want it to optimize:

# In[25]:


opt = torch.optim.Adam(model.parameters(), lr=1e-3)


# then, inside the training loop you:
#  1. calculate the gradients (`loss.backward()`)
#  1. use the optimizer to update the model parameters (`opt.step()`)
#  1. zero the gradients (`opt.zero_grad()`)

# Here's the training loop:

# In[26]:


from tqdm import trange  # utility function to show progress bar

num_epochs = 10
batch_size = 512

# some quantities to plot
train_losses = []
test_losses = []
test_accuracy = []

# "epoch" = one pass through the dataset
for i_epoch in range(num_epochs):
    shuffle_ids = np.random.permutation(len(X_train)) # shuffle the data
    for idx in trange(0, len(X_train), batch_size):
        # get the next chunk (batch) of data:
        batch_X = X_train[shuffle_ids][idx : idx + batch_size].to(device)
        batch_y = y_train[shuffle_ids][idx : idx + batch_size].to(device)

        # all the black magic:
        loss = loss_fn(model(batch_X), batch_y)
        loss.backward()
        opt.step()
        opt.zero_grad()

        # remember the loss value at this step
        train_losses.append(loss.item())

    # evaluate test loss and metrics
    test_prediction = model(X_test.to(device))
    test_losses.append(
        loss_fn(test_prediction, y_test.to(device)).item()
    )
    test_accuracy.append(
        (test_prediction.argmax(axis=1) == y_test.to(device)).to(float).mean()
    )

    # all the rest is simply plotting

    clear_output(wait=True)
    plt.figure(figsize=(8, 3), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train')
    plt.plot(
        np.linspace(0, len(train_losses), len(test_losses) + 1)[1:],
        test_losses, label='test'
    )
    plt.ylabel("Loss")
    plt.xlabel("# steps")
    plt.legend();

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracy)
    plt.ylabel("Test accuracy")
    plt.xlabel("# epochs");
    plt.show()


# ## XOR problem
# 
# The best way to learn is to code it yourself. Try building a nerual network to solve the XOR problem below. Can it be solved by a network with no hidden layers?
# 
# *Hint: for a binary classification loss function consider checking out `torch.nn.BCEWithLogitsLoss`*

# In[27]:


X = np.random.uniform(-1, 1, size=(4000, 2))
y = ((X[:,0] >= 0) ^ (X[:,1] >= 0)).astype(int)

plt.figure(figsize=(3, 3), dpi=100)
plt.scatter(*X.T, c=y, cmap='PiYG', s=5);


# In[ ]:


# Define and train your model below:
# model = ...

# YOUR CODE HERE
raise NotImplementedError()

xx1, xx2 = np.meshgrid(
    np.linspace(-1.1, 1.1, 100),
    np.linspace(-1.1, 1.1, 100),
)
yy = model(torch.tensor(
    np.stack([xx1.ravel(), xx2.ravel()], axis=1),
    dtype=torch.float32
).to(device)).cpu().detach().numpy().reshape(xx1.shape)

plt.figure(figsize=(3, 3), dpi=100)
plt.contourf(xx1, xx2, yy, levels=20, cmap='PiYG', alpha=0.7)
plt.scatter(*X.T, c=y, cmap='PiYG', s=1);


# In[ ]:


assert isinstance(model, nn.Module)

np.random.seed(42)
X = np.random.uniform(-1, 1, size=(4000, 2))
y = ((X[:,0] >= 0) ^ (X[:,1] >= 0)).astype(int)

prediction = model(
    torch.tensor(X, dtype=torch.float32).to(device)
).cpu().detach().numpy().squeeze()
accuracy = ((prediction > 0.).astype(int) == y).mean()
print("Accuracy:", accuracy)
assert accuracy > 0.95

"""
