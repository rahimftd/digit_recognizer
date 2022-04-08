import pandas as pd
import torch
import numpy
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

class LogisticRegression(torch.nn.Module):

    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


input_data = numpy.genfromtxt('train.csv', delimiter=',')
input_data = input_data.astype(float)
print(f'Input data dimensions: ${input_data.shape}')
X = input_data[1:, 1:]
y = input_data[1:, :1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print(f'X_train dimensions: ${X_train.shape}')
print(f'X_test dimensions: ${X_test.shape}')
print(f'y_train dimensions: ${y_train.shape}')
print(f'y_test dimensions: ${y_test.shape}')

num_classifiers = 10

X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
all_y_train = {}
for i in range(num_classifiers):
    all_y_train[i] = torch.from_numpy((y_train == i).astype(float)).type(torch.FloatTensor)


num_features = X_train.size(dim=1)
num_examples = X_train.size(dim=0)
learning_rate = 0.0001
num_epochs = 1000

classifiers = {}
for model_index in range(num_classifiers):
    loss_data = []
    loss_labels = []
    print(f'Training Model # {model_index}...')
    classifiers[model_index] = LogisticRegression(num_features)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(classifiers[model_index].parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        y_predicted = classifiers[model_index](X_train)
        loss = criterion(y_predicted, all_y_train[model_index])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Track loss
        loss_data.append(loss.item())
        loss_labels.append(epoch)
    # plt.plot(loss_labels, loss_data)
    # plt.show()

X_eval = torch.from_numpy(numpy.genfromtxt('test.csv', delimiter=',')[1:, :]).type(torch.FloatTensor)
with torch.no_grad():
    results = numpy.empty(0)
    for i, x in enumerate(X_eval):
        max_y_test_pred = -math.inf
        predicted_class = 0
        for classifier_index in range(num_classifiers):
            classifier = classifiers[classifier_index]
            y_test_pred = classifier(x)
            if y_test_pred > max_y_test_pred:
                max_y_test_pred = y_test_pred
                predicted_class = classifier_index
        results = numpy.append(results, predicted_class)
    results = torch.tensor(results, dtype=torch.int)
    results = torch.reshape(results, (results.shape[0], 1))
    results = results.numpy()
    results_formatted = []
    for image_id, classification in enumerate(results):
        results_formatted.append([image_id + 1, classification[0]])
    df = pd.DataFrame(results_formatted, columns=['ImageId', 'Label'])
    df.to_csv('output.csv', index=False)
