from utils import L2, sigmoid, mini_batches, relu, activation_fn_derivative, tanh
import numpy as np
from sklearn.datasets import make_classification, make_multilabel_classification
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, roc_auc_score, multilabel_confusion_matrix, roc_curve, confusion_matrix
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler
import warnings
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer


class DNN():
    def __init__(self, n_layers):
        pass

    def init_layers(self):
        pass


class Model():
    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.loss = 0
        self.loss_per_epoch = []
        self.cost = 0
        self.cost_per_epoch = []
        self.validation_loss_per_epoch = []
        self.validation_cost_per_epoch = []

    def add(self, layer):
        self.layers.append(layer)
        return self

    def compile(self, learning_rate=0.1, optimizer=None, loss=None, initializer="glorot", epsilon=0.001,
                init_constant=0):

        assert self.layers[0].layer_shape is not None
        # first layer should have ncols of input to init weights matrix of proper size
        # shape of weights matrix (n_units of layer l n_units of layer l-1
        self.layers[0].layer_shape = (self.layers[0].n_units, self.layers[0].layer_shape[0])
        self.layers[0].init_weights(method=initializer, epsilon=epsilon, init_constant=init_constant)
        # add learning rate to first layer
        self.layers[0].lr = learning_rate

        for idx in range(1, len(self.layers)):

            self.layers[idx].layer_shape = (self.layers[idx].n_units, self.layers[idx - 1].n_units)
            self.layers[idx].init_weights(method=initializer, epsilon=epsilon, init_constant=init_constant)
            # add learning rate to each layer
            self.layers[idx].lr = learning_rate
            # else: output layer doesn't need any weights

            # add an identity to output layer
            if idx == len(self.layers) - 1:
                self.layers[idx].isOutputLayer = True


    def fit(self, X, y, val_X, val_y, regularization=L2(0.001), validation_split=0.2, epochs=100, batch_size=10):

        train_X = X
        train_y = y
        print("=====TRAIN and Validation for lambda===== {}".format(regularization._lambda))
        print(np.unique(np.argmax(train_y, axis=1), return_counts=True))
        print(np.unique(np.argmax(val_y, axis=1), return_counts=True))
        print("===============================")
        for epoch in range(epochs):
            self.cost = 0
            self.loss = 0
            outputs = []
            list_mini_batches = mini_batches(train_X, batch_size=batch_size)
            for mini_batch in list_mini_batches:
                self.cost = 0
                self.loss = 0
                self.loss_per_batch = 0
                for row_idx, row in enumerate(mini_batch):
                    row = np.hstack((1, row)).reshape(-1, 1)
                    input = row
                    for idx in range(len(self.layers)):
                        # if idx !=0:
                        output = self.layers[idx].forward_pass(input)
                        input = output
                    outputs.append(input)
                    # BACK PROP STOPS BEFORE THE INPUT LAYER : REASON FROM RANGE HAVING 1
                    _y = y[row_idx].reshape(-1, 1)
                    # for last layer
                    deltas = 0
                    for idx in reversed(range(len(self.layers))):

                        if not self.layers[idx].isOutputLayer:
                            weights = self.layers[idx + 1].weights
                            activations = self.layers[idx].layer_activations
                            # rip off the delta of bias term when back proping
                            deltas = deltas[1:] if not self.layers[idx + 1].isOutputLayer else deltas
                        else:
                            weights = self.layers[-1].weights  # for output layer, there are no weights
                            activations = self.layers[-1].layer_activations

                        deltas, deltas_acc = self.layers[idx].backward_pass(deltas, weights, activations, _y)
                        self.layers[idx].deltas = deltas
                        if not self.layers[idx].isOutputLayer:
                            self.layers[idx + 1].deltas_acc += deltas_acc
                            if idx == 0:
                                deltas = self.layers[idx].deltas[1:]  # rip off bias delta
                                deltas_acc = np.tile(row.T, deltas.shape) * deltas
                                self.layers[idx].deltas_acc += deltas_acc
                        if self.layers[idx].isOutputLayer:
                            expected = _y
                            pred = self.layers[-1].layer_activations
                            self.loss += np.sum(np.square(expected - pred))
                            self.cost += (self.cost_function(expected, pred))

                i = 0
                for layer in self.layers:
                    # reset delta acc for each epoch
                    # average_acc_detlas
                    layer.deltas_acc = (1 / len(train_X)) * layer.deltas_acc
                    layer.update_weights(regularize=regularization, n_samples=len(mini_batch))
                    # reset deltas_acc for next epoch
                    layer.deltas_acc = 0
                    i += 1

            pred_val_y = self.predict(val_X)
            pred_val_y = np.array(pred_val_y)
            expected_val_y = val_y
            self.validation_loss_per_epoch.append((1 / len(val_y)) * np.sum(np.square(expected_val_y - pred_val_y)))
            # dirty hack to get penalty calculated in previous execution
            weights_sum = 0
            for layer in self.layers:
                if not layer.isOutputLayer:
                    # rip of bias
                    weights_sum += np.sum(layer.weights[:, 1:] ** 2)
                else:
                    weights_sum += np.sum(layer.weights ** 2)
            weights_sum = (regularization._lambda / 2) * weights_sum

            # penalty = (0.5) * weights_sum * regularization._lambda
            self.validation_cost_per_epoch.append(
                (-1 / len(val_y)) * (self.cost_function(expected_val_y, pred_val_y) - weights_sum))
            # self.loss_per_batch += ((1 / len(mini_batch)) * self.loss)
            self.loss_per_epoch.append((1 / len(train_X)) * self.loss)
            self.cost_per_epoch.append((-1 / len(train_X)) * (self.cost - weights_sum))
        print("Validation - metrics {} hidden_units {} layers".format(self.layers[0].n_units, len(self.layers) - 1))
        print(classification_report(y_true=np.argmax(expected_val_y, axis=1), y_pred=np.argmax(pred_val_y, axis=1)))
        print(f1_score(y_true=np.argmax(expected_val_y, axis=1), y_pred=np.argmax(pred_val_y, axis=1), average="micro"))
        print(
            confusion_matrix(y_true=np.argmax(expected_val_y, axis=1), y_pred=np.argmax(pred_val_y, axis=1)),
            "C_matrix")
        print(roc_auc_score(y_true=expected_val_y, y_score=pred_val_y), "validation_ROC")

        return outputs

    def predict(self, X):
        outputs = []
        for row_idx, row in enumerate(X):
            row = np.hstack((1, row)).reshape(-1, 1)
            for idx in range(len(self.layers)):
                output = self.layers[idx].forward_pass(row)
                row = output
            outputs.append(row.ravel())

        return outputs

    def cost_function(self, expected_output, pred):
        return np.sum((expected_output * np.log(pred) + (
                (1 - expected_output) * np.log(1 - pred))))

    def tanh_costFunction_persample(self, expected_output, activation_derivative_hypothesis_matrix):
        return np.sum((((expected_output + 1) / 2) * np.log((activation_derivative_hypothesis_matrix + 1) / 2) + (
                (1 - ((expected_output + 1) / 2)) * np.log(1 - (activation_derivative_hypothesis_matrix + 1) / 2))))

    def summary(self):
        for layer in self.layers:
            print(layer)


class Dense():
    def __init__(self, n_units, input_shape=None, epsilon=0.0001, activation=relu, regularization=None):
        self.n_units = n_units
        self.lr = None
        self.activation = activation
        self.regularization = regularization
        self.layer_shape = input_shape
        self.epsilon = epsilon
        self.inputs_from_prev_layer = None
        self.weights = None
        self.layer_activations = None
        # self.isInputLayer=False
        self.isOutputLayer = False
        # delta for a training sample
        self.deltas = 0
        # accumlated delta over a batch
        self.deltas_acc = 0
        # linkedlist like access
        self.prev = None
        self.next = None

    def forward_pass(self, input):
        self.layer_activations = self.activation(np.matmul(self.weights, input))
        if not self.isOutputLayer:
            self.layer_activations = np.vstack((1, self.layer_activations))
        return self.layer_activations

    def backward_pass(self, delta_from_prev_layer, prev_layer_weights, activations_from_prev, expected):
        if self.isOutputLayer:
            # strip off the bias term added during forward pass while comparing with expected
            error = self.layer_activations - expected
            # no need to multiply with the differential
            deltas = error
            # delta_acc comes from next layer for each layer, so 0 initially for o/p layer
            deltas_acc = 0
        else:
            weights = prev_layer_weights

            error = np.matmul(weights.T, delta_from_prev_layer)
            # activations of curr layers and deltas of prev layer to update weights
            deltas_acc = np.tile(self.layer_activations.T, delta_from_prev_layer.shape) * delta_from_prev_layer
            deltas = error * activation_fn_derivative(self.activation)(self.layer_activations)
        return deltas, deltas_acc

    def update_weights(self, n_samples, regularize=None):
        # ignore bias in deltas
        l2_penalty_vector = 0
        if regularize is not None:
            l2_penalty_vector = (regularize._lambda / n_samples) * self.weights
            # don't penalize bias terms
            l2_penalty_vector[:, 0] = 0
        self.weights -= self.lr * (self.deltas_acc + l2_penalty_vector)
        return self.weights

    def init_weights(self, bias=True, method=None, epsilon=0.001, init_constant=None):
        if method == "glorot" and method == "zeros" and method == "random":
            raise Exception("Both Glorot, random and  Zero intilization can't be used at the same time")
        if method == "glorot":
            warnings.warn("Glorot initlization ignores user defined epsilon")
            self.epsilon = np.sqrt(6.0 / sum(self.layer_shape))
            if bias:
                self.weights = np.random.uniform(low=-self.epsilon, high=self.epsilon,
                                                 size=self.layer_shape[0] * (self.layer_shape[1] + 1)) \
                    .reshape(self.layer_shape[0], self.layer_shape[1] + 1)
            else:
                self.weights = np.random.uniform(low=-self.epsilon, high=self.epsilon,
                                                 size=self.layer_shape[0] * self.layer_shape[1]).reshape(
                    self.layer_shape)

        elif method == "zeros":
            self.epsilon = epsilon
            if bias:
                self.weights = np.zeros((self.layer_shape[0], self.layer_shape[1] + 1))
            else:
                self.weights = np.zeros((self.layer_shape))

        elif method == "random":
            self.epsilon = epsilon
            if bias:
                self.weights = np.random.uniform(low=-self.epsilon, high=self.epsilon,
                                                 size=self.layer_shape[0] * (self.layer_shape[1] + 1)) \
                    .reshape(self.layer_shape[0], self.layer_shape[1] + 1)
            else:
                self.weights = np.random.uniform(low=-self.epsilon, high=self.epsilon,
                                                 size=self.layer_shape[0] * self.layer_shape[1]).reshape(
                    self.layer_shape)
        elif method == "symmetric":
            if bias:
                self.weights = np.repeat(init_constant,
                                         repeats=self.layer_shape[0] * (self.layer_shape[1] + 1)).reshape(
                    self.layer_shape[0], self.layer_shape[1] + 1)
            else:
                self.weights = np.repeat(init_constant, repeats=self.layer_shape[0] * (self.layer_shape[1])).reshape(
                    self.layer_shape[0], self.layer_shape[1])

        else:
            warnings.warn("initializer {} not found! so using default linspace initializer".format(method))
            self.epsilon = epsilon
            if bias:
                self.weights = np.linspace(start=-self.epsilon, stop=self.epsilon,
                                           num=self.layer_shape[0] * (self.layer_shape[1] + 1)) \
                    .reshape(self.layer_shape[0], self.layer_shape[1] + 1)
            else:
                self.weights = np.linspace(start=-self.epsilon, stop=self.epsilon,
                                           num=self.layer_shape[0] * self.layer_shape[1]).reshape(self.layer_shape)

    def __str__(self):
        return """
        Layer Info : \n
        =====================================================\n
        shape of weights {} \n
        shape of inputs {} \n
        =====================================================\n
        weights {} \n
        =====================================================\n
        Regularization {}
        """.format(self.weights.shape, self.inputs_from_prev_layer, self.weights, self.regularization)


def preprocess_data(data, upsample=False):
    data_x = data.iloc[:, 0:-1]
    data_y = data.iloc[:, -1:np.newaxis]
    if upsample:
        sampler = RandomOverSampler()
        data_x_sampled, data_y_sampled = sampler.fit_sample(data_x, data_y)
        print(data_x_sampled.shape[0] - data_x.shape[0], 'new random picked points')
        data_x = data_x_sampled
        data_y = data_y_sampled.reshape(-1, 1)
    # standard_x=StandardScaler().fit_transform(data_x)
    standard_x = (data_x - data_x.mean()) / data_x.std()
    return np.hstack((standard_x, np.array(data_y)))


if __name__ == "__main__":
    seed = 162
    EPOCHS = 2500
    _lambdas=[0.01]
    # replace this with your requried features
    required_cols = ["all_NBME_avg_n4","all_mcqs_avg_n20", "CBSE_01","CBSE_02","LEVEL"]
    #replace this with ur dataset file
    data_raw = pd.read_csv("BSOM_Dataset_for_HW3.csv")
    print("selected random seed {}".format(seed))
    np.random.seed(seed)
    print(list(data_raw.columns))
    data_raw = data_raw.drop(["BCR_PI_05", "Random_ID"], axis=1)
    imputed_data = SimpleImputer(strategy="most_frequent").fit_transform(data_raw)
    data = pd.DataFrame(imputed_data, columns=data_raw.columns)
    y = data[["LEVEL"]]
    X = data.drop("LEVEL", axis=1)


    # feature selection using filter method
    n_best = 10
    k_best = SelectKBest(chi2, k=n_best)
    selected_features = k_best.fit_transform(X, y)
    cols = list(data.columns[np.nonzero(k_best.get_support())])
    scores = k_best.scores_[np.nonzero(k_best.get_support())]
    print(" {} best features {} with scores {}".format(n_best, cols ,scores))

    #display a bar plot
    sns.barplot(cols, scores)
    plt.xlabel("features")
    plt.ylabel("Chi_Squared")
    plt.title("Chi_sqaured vs features")
    plt.xticks(rotation=90)
    plt.show()

    for col in cols:
        if col in required_cols:
            index = cols.index(col)
            del cols[index]

    print(cols)


    data = data.dropna()
    # using stratified split to ensure all classes are present while testing
    pre_processed_data = preprocess_data(data, upsample=False)
    train, test = train_test_split(pre_processed_data, shuffle=True, test_size=0.2, random_state=seed,
                                   stratify=pre_processed_data[:, -1])  # stratify=pre_processed_data[:, -1]
    train, validation = train_test_split(train, shuffle=True, test_size=0.2, random_state=seed, stratify=train[:, -1])
    X_train, X_test, X_val = train[:, :-1].astype(np.float64), test[:, :-1].astype(np.float64), validation[:,
                                                                                                :-1].astype(np.float64)
    y_train, y_test, y_val = train[:, -1], test[:, -1], validation[:, -1]


    print("=" * 10)
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_val, return_counts=True))
    print(np.unique(y_test, return_counts=True))
    print("=" * 10)
    # converting strings to numerics labels
    y_train = np.array(pd.get_dummies(y_train), dtype=np.int32)
    y_test = np.array(pd.get_dummies(y_test), dtype=np.int32)
    y_val = np.array(pd.get_dummies(y_val), dtype=np.int32)

    hidden_units = 2
    _layer = 1
    for _lambda in _lambdas:  #
        model = Model()
        n_cols = X_train.shape[-1]
        model.add(Dense(hidden_units, activation=sigmoid, input_shape=(n_cols,)))
        model.add(Dense(hidden_units, activation=sigmoid))
        model.add(Dense(4, activation=sigmoid))
        model.compile(learning_rate=0.5, initializer="random", epsilon=0.1, init_constant=1.0)
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        model.fit(X_train, y_train, X_val, y_val, regularization=L2(_lambda=_lambda), epochs=EPOCHS,
                  validation_split=0.2, batch_size=len(X_train))
        print("Training metrics for {} hidden units  {} layers".format(hidden_units, _layer))
        pred = model.predict(X_train)
        print(roc_auc_score(y_true=y_train, y_score=pred), "train_ROC")
        print(classification_report(y_true=np.argmax(y_train, axis=1),
                                    y_pred=np.argmax(pred, axis=1)))  # labels=np.unique(np.argmax(y_train, axis=1))
        print(f1_score(y_true=np.argmax(y_train, axis=1), y_pred=np.argmax(pred, axis=1), average="micro"))
        print(confusion_matrix(y_true=np.argmax(y_train, axis=1), y_pred=np.argmax(pred, axis=1)),
              "C_matrix")

        print("Test metrics for {} hidden units {} layers ".format(hidden_units, _layer))
        pred = model.predict(X_test)
        print(roc_auc_score(y_true=y_test, y_score=pred), "test_ROC")
        print(confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(pred, axis=1)), "C_matrix")

        print(classification_report(y_true=np.argmax(y_test, axis=1),
                                    y_pred=np.argmax(pred, axis=1)))  # labels=np.unique(np.argmax(y_test, axis=1))
        print(f1_score(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(pred, axis=1), average="micro"))

        print(
            "=======training cost and validation cost minimas for lambda {} and {} hidden units {} layers ===========".format(
                _lambda, hidden_units, _layer))
        index = np.argmin(model.cost_per_epoch)
        print(model.cost_per_epoch[index], index + 1)

        index = np.argmin(model.validation_cost_per_epoch)
        print(model.validation_cost_per_epoch[index], index + 1)

        plt.title("Cost vs epoch \n lambda {}, {} hidden units, {} layer \n and {} init weights ".format(_lambda,
                                                                                                         hidden_units,
                                                                                                         _layer,
                                                                                                         "Random intialization"))
        plt.plot(model.cost_per_epoch, label="training_cost")
        plt.plot(model.validation_cost_per_epoch, label="validation_cost")
        plt.axhline(model.validation_cost_per_epoch[index], linestyle=":", label="min_validation_cost")
        plt.axvline(index, linestyle=":", label="min_val_cost_epoch")
        plt.legend()
        plt.show()
