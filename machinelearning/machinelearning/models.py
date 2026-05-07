from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'


        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()

        self.w = Parameter(ones(1, dimensions))

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        return tensordot(self.w, x, dims=[[1], [1]])

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return 1 if self.run(x).item() >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            converged = False
            while not converged:
                converged = True
                for batch in dataloader:
                    x = batch['x']
                    y = batch['label']
                    if self.get_prediction(x) != y.item():
                        self.w += y * x
                        converged = False


class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(1, 256)
        self.layer2 = Linear(256, 256)
        self.layer3 = Linear(256, 1)

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        h1 = relu(self.layer1(x))
        h2 = relu(self.layer2(h1))
        return self.layer3(h2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        return mse_loss(self.forward(x), y)

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on

        """
        # batch_size=40 evenly divides the 200-point dataset
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        dataloader = DataLoader(dataset, batch_size=40, shuffle=True)

        while True:
            total_loss = 0.0
            count = 0
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
            if total_loss / count < 0.02:
                break


class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        self.layer1 = Linear(input_size, 256)
        self.layer2 = Linear(256, 128)
        self.layer3 = Linear(128, output_size)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        h1 = relu(self.layer1(x))
        h2 = relu(self.layer2(h1))
        return self.layer3(h2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        return cross_entropy(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        # batch_size=100 evenly divides the 60,000-example MNIST training set
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

        while True:
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()

            # stop once validation accuracy clears 97.5%
            if dataset.get_validation_accuracy() >= 0.975:
                break
