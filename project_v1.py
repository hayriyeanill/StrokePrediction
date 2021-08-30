# remove tenserflow info from console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.get_logger().setLevel('INFO')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


# In every epoch, you can callback to a code function, having checked the metrics.
# If they're what you want to say, then you can cancel the training at that point.
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


data = pd.read_csv("healthcare-dataset-stroke-data.csv", sep=',')

# DATA PRE PROCESSING
# Encode to numeric values
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
class_names = {index: label for index, label in enumerate(le.classes_)}

data['ever_married'] = le.fit_transform(data['ever_married'])
class_names = {index: label for index, label in enumerate(le.classes_)}

data['work_type'] = le.fit_transform(data['work_type'])
class_names = {index: label for index, label in enumerate(le.classes_)}

data['Residence_type'] = le.fit_transform(data['Residence_type'])
class_names = {index: label for index, label in enumerate(le.classes_)}

data['smoking_status'] = le.fit_transform(data['smoking_status'])
class_names = {index: label for index, label in enumerate(le.classes_)}

# Fill missing value in bmi with mean of column bmi
data['bmi'].fillna(value=data['bmi'].mean(), inplace=True)


class NeuralNetworkClassification:
    def __init__(self):
        #  10 input 1 output
        self.X = data.iloc[:, 1:-1].values
        self.y = data.iloc[:, -1].values
        # train-test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=42)
        # scaler
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.x_train)
        self.X_test = sc.fit_transform(self.x_test)

        self.callbacks = myCallback()

        # EXTREME LEARNING MACHINE
        self.H = 0
        self.beta = 0

    def visualize_confusion_matrix(self, conf_matrix, cmap):
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts = ['{0:0.0f}'.format(value) for value in conf_matrix.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in
                             conf_matrix.flatten() / np.sum(conf_matrix)]
        labels = ["{0}\n{1}\n{2}".format(v1, v2, v3) for v1, v2, v3 in
                  zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(conf_matrix, annot=labels, fmt="", cmap=cmap)
        plt.show()

    def plot_accuracy_vs_loss(self, history, title):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.plot(epochs, acc, label='Training accuracy')
        plt.plot(epochs, val_acc, label='Validation accuracy')
        plt.title('Training and validation accuracy for ' + title)
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title('Training and validation loss for ' + title)
        plt.legend()
        plt.show()

    def simple_perceptron_model(self, epoch):
        sp = Sequential()
        # Add an input layer
        sp.add(Dense(units=10, activation='relu', input_dim=self.X_train.shape[1]))
        # Add an output layer
        sp.add(Dense(1, activation='sigmoid'))
        sp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        sp_history = sp.fit(self.X_train, self.y_train, epochs=epoch, validation_split=0.2,
                            batch_size=1, verbose=1, callbacks=[self.callbacks])
        sp_pred = sp.predict(self.X_test)
        sp_pred = (sp_pred > 0.5).astype(int)
        sp_cm = confusion_matrix(self.y_test, sp_pred)
        plt.title("Simple Perceptron Confusion Matrix for epoch " + str(epoch))
        title = "Simple Perceptron for epoch " + str(epoch)
        self.visualize_confusion_matrix(sp_cm, 'Greens')
        self.plot_accuracy_vs_loss(sp_history, title)

    def multilayer_perceptron_model(self, epoch):
        mlp = Sequential()
        # Add an input layer
        mlp.add(Dense(units=10, activation='relu', input_dim=self.X_train.shape[1]))
        # Add an hidden layer
        mlp.add(Dense(units=5, activation='sigmoid'))
        # Add an hidden layer
        mlp.add(Dense(units=5, activation='relu'))
        # Add an output layer
        mlp.add(Dense(1, activation='sigmoid'))
        mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        mlp_history = mlp.fit(self.X_train, self.y_train, epochs=epoch, validation_split=0.2, batch_size=1, verbose=1,
                              callbacks=[self.callbacks])
        mlp_pred = mlp.predict(self.X_test)
        mlp_pred = (mlp_pred > 0.5).astype(int)
        mlp_cm = confusion_matrix(self.y_test, mlp_pred)
        plt.title("Multilayer Perceptron Confusion Matrix for epoch " + str(epoch))
        title = "Multilayer Perceptron for epoch " + str(epoch)
        self.visualize_confusion_matrix(mlp_cm, 'Reds')
        self.plot_accuracy_vs_loss(mlp_history, title)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def predict_elm_model(self, X, weight, bias, beta):
        X = np.array(X)
        y = np.dot(self.sigmoid((np.dot(X, weight.T)) + bias), beta)
        return y

    def train_elm_model(self, X, y, weight, bias):
        X = np.array(X)
        y = np.array(y)
        # Sigmoid activation function
        self.H = self.sigmoid(np.dot(X, weight.T) + bias)
        # Calculate the Moore-Penrose pseudo inverse matrix
        H_moore_penrose = np.dot(np.linalg.inv(np.dot(self.H.T, self.H)), self.H.T)
        # Calculate the output weight matrix beta
        self.beta = np.dot(H_moore_penrose, y)
        return np.dot(self.H, self.beta)

    def elm_model(self):
        input_size = self.X.shape[1]
        hidden_size = [10, 20, 100, 200, 1000]
        acc_list = []
        for h in hidden_size:
            # Initialize random weight and bias with range [-0.5, 0.5]
            weight = np.array(np.random.uniform(-0.5, 0.5, (h, input_size)))
            bias = np.array(np.random.uniform(0, 1, (1, h)))

            self.train_elm_model(self.X_train, self.y_train.reshape(-1, 1), weight, bias)
            elm_pred = self.predict_elm_model(self.X_test, weight, bias, self.beta)
            elm_pred = (elm_pred > 0.5).astype(int)
            print('Acc elm model for hidden layer is ', h, accuracy_score(self.y_test, elm_pred))
            acc_list.append([h, accuracy_score(self.y_test, elm_pred)])
            elm_cm = confusion_matrix(self.y_test, elm_pred)
            plt.title("ELM Confussion Matrix for Hidden Size is " + str(h))
            self.visualize_confusion_matrix(elm_cm, 'Oranges')

        acc_list_df = pd.DataFrame(acc_list, columns=['h', 'accuracy'])
        sns.lineplot(data=acc_list_df, x='h', y='accuracy')
        plt.title("Hidden Size vs Accuracy")
        plt.xlabel("h")
        plt.ylabel("acc")
        plt.show()

    def svc(self):
        kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
        for kernel in kernel_list:
            svc = SVC(kernel=kernel)
            svc.fit(self.X_train, self.y_train)
            svc_pred = svc.predict(self.X_test)
            cm_svc = confusion_matrix(self.y_test, svc_pred)
            acc_svc = accuracy_score(self.y_test, svc_pred)
            print("acc_svc " + kernel, acc_svc)
            plt.title("SVC Confusion Matrix for " + kernel)
            self.visualize_confusion_matrix(cm_svc, 'Purples')

    def run(self):
        self.simple_perceptron_model(epoch=20)
        self.multilayer_perceptron_model(epoch=20)
        self.simple_perceptron_model(epoch=50)
        self.multilayer_perceptron_model(epoch=50)
        self.elm_model()
        self.svc()


NeuralNetworkClassification().run()
