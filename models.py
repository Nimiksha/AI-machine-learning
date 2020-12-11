import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

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
        """
        "*** YOUR CODE HERE ***"
        # return the score - i.e the the dot product of the given weight and the weight vector
        score = nn.DotProduct(x, self.get_weights())
        return score

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # get the score assigned to the perceptron (from the function defined before)
        score = self.run(x)
        # using nn.as_scalar to convert the score into a floating-point number, as recommended in question
        # if the predicted class is + ve return 1 else return -1
        if nn.as_scalar(score) >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        # initialize training to be True
        training = True
        # set the batch size to iterate over as 1
        batch_size = 1

        # set up loop to train the perceptron, runs till an error-free pass over the dataset is completed
        while training:
            # set training to False to break the loop when the perceptron converges
            training = False
            # iterate over the the perceptron training data for a batch size of 1
            # used as recommended in the question to retrieve batches of training examples
            for (x, y) in dataset.iterate_once(batch_size):
                # get the predicted class value (+1 or -1) of the input features using get_prediction() defined before
                class_prediction = self.get_prediction(x)
                # get the python floating point number for the correct label y
                scalar_value = nn.as_scalar(y)
                # compare the predicted class for the input data point to the value received
                # if the predicted value is not equal to the scalar value received,
                # keep on training (set it back to True) and update the value of parameters by updating the weights
                if class_prediction != scalar_value:
                    training = True
                    self.get_weights().update(x, scalar_value)


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # initialize weight and bias for the model parameter.
        # The trainable parameters of perceptron are chosen arbitrarily for the network to be sufficiently large
        # to approximate sin(x) over the given interval
        self.weight1 = nn.Parameter(1, 100)
        self.bias1 = nn.Parameter(1, 100)

        self.weight2 = nn.Parameter(100, 1)
        self.bias2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Based on the Linear Regression example given in the question
        # get the linear shape and compute the model's predictions for y
        xm1 = nn.Linear(x, self.weight1)
        predicted_y1 = nn.AddBias(xm1, self.bias1)
        # for non-linearity
        fx1 = nn.ReLU(predicted_y1)

        # In this model, I am choosing to only do 2 layer deep network
        xm2 = nn.Linear(fx1, self.weight2)
        fx_net = nn.AddBias(xm2, self.bias2)
        return fx_net

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # construct loss node (reference linear regression example)
        loss = nn.SquareLoss(self.run(x), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # initialize training to be False
        training = False
        # arbitrarily set batch size to be 5 and learning rate to be -0.005
        batch_size = 10
        learning_rate = -0.01

        # set up loop to train the neural network, which runs till it converges
        while not training:
            for (x, y) in dataset.iterate_once(batch_size):
                # get the loss value
                loss = self.get_loss(x, y)
                # get the gradient values for the loss with respect to the parameters
                gradient_w1, gradient_w2, gradient_b1, gradient_b2 = nn.gradients(loss, [self.weight1, self.weight2, self.bias1, self.bias2])
                # update the parameters (both weight and bias) as required based on the learning rate
                # m.update(grad_wrt_m, multiplier)
                self.weight1.update(gradient_w1, learning_rate)
                self.weight2.update(gradient_w2, learning_rate)
                self.bias1.update(gradient_b1, learning_rate)
                self.bias2.update(gradient_b2, learning_rate)

                # calculate the total loss averaged across all examples in the dataset
                total_loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
                # check if the final loss is <= 0.02, and if it is, the test passes
                if nn.as_scalar(total_loss) <= 0.02:
                    training = True
                    return


class DigitClassificationModel(object):
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
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # similar to Q2, but the values are changed
        # 784 based on the question, and 200 is arbitrarily chosen
        self.weight1 = nn.Parameter(784, 500)
        self.bias1 = nn.Parameter(1, 500)

        self.weight2 = nn.Parameter(500, 10)
        self.bias2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # similar implementation to Regression model
        xm1 = nn.Linear(x, self.weight1)
        predicted_y1 = nn.AddBias(xm1, self.bias1)
        fx1 = nn.ReLU(predicted_y1)

        xm2 = nn.Linear(fx1, self.weight2)
        fx_net = nn.AddBias(xm2, self.bias2)
        return fx_net

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SquareLoss(self.run(x), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Almost similar implementation to Q2
        # arbitrarily initialized batch_size to be 10 and consequently learning rate to be -0.75 keeping in mind
        # that smaller batches require lower learning rates
        batch_size = 25
        learning_rate = -0.9888

        # set up loop to train the neural network, which runs till it achieves an accuracy of over 97%
        while dataset.get_validation_accuracy() < 0.97:
            for (x, y) in dataset.iterate_once(batch_size):
                # get the loss value
                loss = self.get_loss(x, y)
                # get the gradient values for the loss with respect to the parameters
                gradient_w1, gradient_w2, gradient_b1, gradient_b2 = nn.gradients(loss, [self.weight1, self.weight2, self.bias1, self.bias2])
                # update the parameters (both weight and bias) as required based on the learning rate
                # m.update(grad_wrt_m, multiplier)
                self.weight1.update(gradient_w1, learning_rate)
                self.weight2.update(gradient_w2, learning_rate)
                self.bias1.update(gradient_b1, learning_rate)
                self.bias2.update(gradient_b2, learning_rate)


# Skip Q4 (not required)
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
