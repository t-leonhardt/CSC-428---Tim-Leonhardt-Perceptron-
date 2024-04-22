# The code, including results, can also be found here:
#               https://www.kaggle.com/code/tleonhardt/csc-428-tim-leonhardt-lab-8

import numpy as np

class Perceptron:
    def __init__(self, input_size, threshold, learning_rate, initial_weights):
        self.weights = initial_weights
        self.threshold = threshold
        self.learning_rate = learning_rate

    def predict(self, input_data):
        activation = np.dot(self.weights, input_data)
        return 1 if activation >= self.threshold else 0

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            print("\nEpoch", epoch + 1)
            for i in range(len(X)):
                print("\tInitial weights:", [f"{w:.1f}" for w in self.weights])
                input_data = X[i]
                target = y[i]
                prediction = self.predict(input_data)
                error = target - prediction
                self.weights += self.learning_rate * error * input_data
                print("\tInput:", input_data, "\n\tDesired Output:", target, "\tActual Output:", prediction)
                print("\tError:", error, "\t\tFinal weights:", [f"{w:.1f}" for w in self.weights], "\n")
                

initial_weights = [0.3, -0.1]
threshold = 0.2
learning_rate = 0.1
epochs = 5
input_size=2


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])


perceptron = Perceptron(input_size, threshold, learning_rate, initial_weights)
perceptron.train(X, y, epochs)


test_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]


print("Testing the trained perceptron:")
for inputs in test_inputs:
    prediction = perceptron.predict(inputs)
    print("Input:", inputs, "Predicted Output:", prediction)    

