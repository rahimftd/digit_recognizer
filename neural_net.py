import pandas as pd
import torch
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class NeuralNet(torch.nn.Module):

    def __init__(self, input_features, hidden_layer_size, output_classes):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(input_features, hidden_layer_size)
        self.l2 = torch.nn.Linear(hidden_layer_size, output_classes)

    def forward(self, X):
        hidden_layer = torch.sigmoid(self.l1(X))
        return self.l2(hidden_layer)


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

X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_train = torch.reshape(y_train, (-1,)).type(torch.LongTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)


num_features = X_train.size(dim=1)
num_examples = X_train.size(dim=0)
hidden_layer_size = 100
num_classes = 10
learning_rate = 0.1
num_epochs = 1000
# batch_size = 100

model = NeuralNet(num_features, hidden_layer_size, num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print("Starting model training...")
loss_data = []
epoch_data = []
for epoch in range(num_epochs):
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    if (epoch + 1) % 10 == 0:
        loss_data.append(loss.item())
        epoch_data.append(epoch+1)
        print(f'Loss @ Epoch {epoch + 1}: {loss.item():.4f}')
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
plt.plot(epoch_data, loss_data)
plt.show()

with torch.no_grad():
    X_eval = torch.from_numpy(numpy.genfromtxt('test.csv', delimiter=',')[1:, :]).type(torch.FloatTensor)
    y_eval = model(X_eval).type(torch.LongTensor)
    results = []
    for i, row in enumerate(y_eval):
        classification = torch.argmax(row)
        results.append([i+1, classification.item()])
    df = pd.DataFrame(results, columns=['ImageId', 'Label'])
    df.to_csv('output.csv', index=False)
