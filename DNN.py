
from utils import L2, sigmoid, mini_batches, relu, activation_fn_derivative, tanh
import numpy as np
from sklearn.datasets import make_classification, make_multilabel_classification
import pandas as pd
from sklearn.model_selection  import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler
import warnings
import seaborn as sns

class DNN():
    def __init__(self, n_layers):
        pass
    def init_layers(self):
        pass


class Model():
    def __init__(self):
        self.layers=[]
        self.optimizer= None
        self.loss = 0
        self.loss_per_epoch = []
        self.cost = 0
        self.cost_per_epoch = []
        self.validation_loss_per_epoch =[]
        self.validation_cost_per_epoch= []




    def add(self, layer):
        self.layers.append(layer)
        return self

    def compile(self, learning_rate=0.1 , optimizer=None, loss=None, initializer="glorot", epsilon=0.001):
        # init input layer
        #self.layers[0] = Input_Layer(self.layers[0])
        #self.layers.append(Dense())
        # creating a input layer
        # input_layer = InputLayer(Dense(self.layers[0].layer_shape[0]))
        # output_layer = OutputLayer(self.layers[-1])
        # self.layers = [input_layer] + self.layers[:-1] + [output_layer]
        assert self.layers[0].layer_shape is not None
        # first layer should have ncols of input to init weights matrix of proper size
        # shape of weights matrix (n_units of layer l n_units of layer l-1
        self.layers[0].layer_shape= (self.layers[0].n_units, self.layers[0].layer_shape[0])
        self.layers[0].init_weights(method=initializer, epsilon=epsilon)
        # add learning rate to first layer
        self.layers[0].lr = learning_rate
        # self.layers[0].prev = None


        for idx in range(1,len(self.layers)):
            # shape of weights matrix (n_units of layer l, n_units of layer l-1)
            # if isinstance(self.layers[idx], InputLayer):
            #     self.layers[idx].layer_shape = (self.layers[idx].n_units, self.layers[idx].layer_shape[0])
            #     self.layers[idx].init_weights()
            #     self.layers[idx].lr = learning_rate
            # elif not isinstance(self.layers[idx], OutputLayer):
            self.layers[idx].layer_shape = (self.layers[idx].n_units, self.layers[idx - 1].n_units)
            self.layers[idx].init_weights(method=initializer, epsilon=epsilon)
            # add learning rate to each layer
            self.layers[idx].lr = learning_rate
            #else: output layer doesn't need any weights

            #add an identity to output layer
            if idx == len(self.layers)-1:
                self.layers[idx].isOutputLayer=True


            # if isinstance(layer, Dense):
            #     # init weights in each layer
            #     layer.init_weights()
            #     pass

    def fit(self, X, y, val_X, val_y, regularization=L2(0.001), validation_split=0.2, epochs = 100, batch_size=10):
        update_trend = [[],[]]
        # n_cols_x, n_cols_y = X.shape[-1], y.shape[-1]
        # data = np.hstack((X,y))
        # train_data, validation = train_test_split( data,test_size=validation_split, random_state=seed, stratify=np.argmax(data[:, -n_cols_y:], axis=1)) #stratify=np.argmax(data[:, -n_cols_y:], axis=1)
        # train_X, train_y = train_data[:,:n_cols_x] , train_data[:, -n_cols_y: ]
        # val_X, val_y = validation[:, :n_cols_x], validation[:, -n_cols_y:]
        train_X= X
        train_y = y
        print("=====TRAIN and Validation for lambda===== {}".format(regularization._lambda))
        print(np.unique(np.argmax(train_y, axis=1), return_counts=True))
        print(np.unique(np.argmax(val_y, axis=1), return_counts=True))
        print("===============================")
        for epoch in range(epochs):
            self.cost = 0
            self.loss = 0
            outputs = []
            list_mini_batches= mini_batches(train_X, batch_size=batch_size)
            for mini_batch in list_mini_batches:
                self.cost = 0
                self.loss = 0
                self.loss_per_batch=0
                for row_idx, row in enumerate(mini_batch):
                    row = np.hstack((1, row)).reshape(-1,1)
                    input = row
                    for idx in range(len(self.layers)):
                        #if idx !=0:
                        output=self.layers[idx].forward_pass(input)
                        input = output
                    outputs.append(input)
                    # BACK PROP STOPS BEFORE THE INPUT LAYER : REASON FROM RANGE HAVING 1
                    _y = y[row_idx].reshape(-1,1)
                    # for last layer
                    deltas = 0
                    for idx in reversed(range(len(self.layers))):

                        if not self.layers[idx].isOutputLayer:
                            weights = self.layers[idx+1].weights
                            activations = self.layers[idx].layer_activations
                            # rip off the delta of bias term when back proping
                            deltas= deltas[1:] if not self.layers[idx+1].isOutputLayer else deltas
                        else:
                            weights = self.layers[-1].weights # for output layer, there are no weights
                            activations = self.layers[-1].layer_activations

                        deltas, deltas_acc = self.layers[idx].backward_pass(deltas, weights, activations,  _y)
                        self.layers[idx].deltas = deltas
                        if not self.layers[idx].isOutputLayer:
                            self.layers[idx+1].deltas_acc += deltas_acc
                            if idx==0:
                                deltas = self.layers[idx].deltas[1:] # rip off bias delta
                                deltas_acc=np.tile(row.T, deltas.shape) * deltas
                                self.layers[idx].deltas_acc += deltas_acc
                        if self.layers[idx].isOutputLayer:
                            expected = _y
                            pred = self.layers[-1].layer_activations
                            self.loss += np.sum(np.square(expected-pred))
                            self.cost += (self.cost_function(expected, pred))

                i =0
                for layer in self.layers:
                    # reset delta acc for each epoch
                    # average_acc_detlas
                    layer.deltas_acc = (1/len(train_X))*layer.deltas_acc
                    layer.update_weights(regularize=regularization, n_samples=len(mini_batch))
                    # reset deltas_acc for next epoch
                    layer.deltas_acc = 0
                    i+=1

            pred_val_y= self.predict(val_X)
            pred_val_y = np.array(pred_val_y)
            expected_val_y = val_y
            self.validation_loss_per_epoch.append((1/len(val_y)) * np.sum(np.square(expected_val_y-pred_val_y)))
            self.validation_cost_per_epoch.append((-1/len(val_y)) * self.cost_function(expected_val_y, pred_val_y))
                #self.loss_per_batch += ((1 / len(mini_batch)) * self.loss)
            self.loss_per_epoch.append((1 / len(train_X)) * self.loss)
            self.cost_per_epoch.append((-1 / len(train_X)) * self.cost)
        print("Validation - metrics", epoch+1)
        print(classification_report(y_true=np.argmax(expected_val_y, axis=1),y_pred=np.argmax(pred_val_y, axis=1)))
        print(f1_score(y_true=np.argmax(expected_val_y, axis=1), y_pred=np.argmax(pred_val_y, axis=1), average="weighted"))

        # if len(self.loss_per_epoch) > 2:
            #     if np.isclose(self.loss_per_epoch[-1], self.loss_per_epoch[-2], atol=0.000001):
            #         print("the algo converged at {}".format(epoch))
            #         print("loss at epoch {} and {} are {}, {}".format(epoch-1, epoch, self.loss_per_epoch[-2], self.loss_per_epoch[-1]))
            #         break


            #self.loss = np.mean(np.sum(outputs - y, axis=0))**(1/2)
            # """
            # epoch: {} \n
            # \t training loss :{}
            # \t validation loss: {}
            # \t accuracy: {}
            # """.format(self.loss, 0, 0)
        #self.cost_per_epoch.reverse()
        #self.loss_per_epoch.reverse()
        return outputs


    def predict(self, X):
        outputs=[]
        for row_idx, row in enumerate(X):
            row = np.hstack((1, row)).reshape(-1, 1)
            for idx in range(len(self.layers)):
                # if idx !=0:
                # print("weights during predictions for layer {}".format(idx))
                # print(self.layers[idx].weights)
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
    def __init__(self,n_units, input_shape=None, epsilon=0.0001,  activation=relu, regularization=None):
        self.n_units = n_units
        self.lr =  None
        self.activation= activation
        self.regularization  = regularization
        self.layer_shape = input_shape
        self.epsilon = epsilon
        self.inputs_from_prev_layer = None
        self.weights=None
        self.layer_activations = None
        # self.isInputLayer=False
        self.isOutputLayer=False
        # delta for a training sample
        self.deltas = 0
        # accumlated delta over a batch
        self.deltas_acc=0
        # linkedlist like access
        self.prev=None
        self.next=None
    def forward_pass(self, input):
        self.layer_activations = self.activation(np.matmul(self.weights, input))
        if not self.isOutputLayer:
            self.layer_activations = np.vstack((1, self.layer_activations))
        return self.layer_activations

    # def backward_pass(self, delta_from_prev_layer,activation_prev_layer):
    #     derivative_of_activations = activation_prev_layer*(1-activation_prev_layer)
    #     # add 1 as bias activation
    #     derivative_of_activations_with_bias = np.vstack((1,derivative_of_activations))
    #     self.deltas = np.matmul(self.weights.T, delta_from_prev_layer)*(derivative_of_activations_with_bias)  #ignore activations
    #     #assert self.lr is not None
    #     # todo: check the sign whether its +ve or -ve
    #     #self.weights = self.weights - self.lr * self.layer_activations *  self.deltas
    #     return self.deltas

    def backward_pass(self, delta_from_prev_layer, prev_layer_weights ,activations_from_prev,expected):
        if self.isOutputLayer:
            # strip off the bias term added during forward pass while comparing with expected
            error = self.layer_activations - expected
            # no need to multiply with the differential
            deltas = error
            # delta_acc comes from next layer for each layer, so 0 initially for o/p layer
            deltas_acc=0
        else:
            # layer 2 weights belong to layer 1 and layer 1 weights belong to input layer
            weights = prev_layer_weights
            # rip off bias activations and deltas when back proping
            # #activations_from_prev[1:]

            #delta_from_prev_layer = delta_from_prev_layer[1:]

            error = np.matmul(weights.T, delta_from_prev_layer)
            # activations of curr layers and deltas of prev layer to update weights
            deltas_acc = np.tile(self.layer_activations.T, delta_from_prev_layer.shape) * delta_from_prev_layer
            deltas = error* activation_fn_derivative(self.activation)(self.layer_activations)
        return deltas,  deltas_acc


    def update_weights(self, n_samples ,regularize=L2(0.001)):
        #ignore bias in deltas
        l2_penalty_vector=0
        if regularize is not None:
            l2_penalty = regularize._lambda * (1/n_samples) * np.sum(self.weights)
            l2_penalty_vector = np.tile(l2_penalty, reps=len(self.deltas_acc)).reshape(-1, 1)
            # don't penalize bias terms
            l2_penalty_vector[0]=0
        self.weights -= (self.lr * self.deltas_acc) + l2_penalty_vector
        return self.weights


    def init_weights(self,bias=True, method="glorot", epsilon=0.001):
        if method == "glorot" and method=="zeros" and method=="random":
            raise Exception("Both Glorot, random and  Zero intilization can't be used at the same time")
        if method=="glorot":
            warnings.warn("Glorot initlization ignores user defined epsilon")
            self.epsilon = np.sqrt(6.0 / sum(self.layer_shape))
            if bias:
                self.weights= np.random.uniform(low= -self.epsilon, high= self.epsilon, size=self.layer_shape[0] * (self.layer_shape[1]+1))\
                    .reshape(self.layer_shape[0], self.layer_shape[1]+1)
            else:
                self.weights = np.random.uniform(low=-self.epsilon, high=self.epsilon,
                                           size=self.layer_shape[0] * self.layer_shape[1]).reshape(self.layer_shape)

        elif method=="zeros":
            self.epsilon = epsilon
            if bias:
                self.weights= np.zeros((self.layer_shape[0], self.layer_shape[1]+1))
            else:
                self.weights = np.zeros((self.layer_shape))

        elif method=="random":
            self.epsilon = epsilon
            if bias:
                self.weights = np.random.uniform(low=-self.epsilon, high=self.epsilon,
                                                 size=self.layer_shape[0] * (self.layer_shape[1] + 1)) \
                    .reshape(self.layer_shape[0], self.layer_shape[1] + 1)
            else:
                self.weights = np.random.uniform(low=-self.epsilon, high=self.epsilon,
                                                 size=self.layer_shape[0] * self.layer_shape[1]).reshape(
                    self.layer_shape)
        else:
            self.epsilon = epsilon
            if bias:
                self.weights= np.linspace(start= -self.epsilon, stop= self.epsilon, num=self.layer_shape[0] * (self.layer_shape[1]+1))\
                    .reshape(self.layer_shape[0], self.layer_shape[1]+1)
            else:
                self.weights= np.linspace(start= -self.epsilon, stop= self.epsilon, num=self.layer_shape[0] * self.layer_shape[1]).reshape(self.layer_shape)
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

def preprocess_data(data, upsample = False):
    data_x = data.iloc[:, 0:-1]
    data_y = data.iloc[:,-1:np.newaxis]
    if upsample:
        sampler = RandomOverSampler()
        data_x_sampled, data_y_sampled = sampler.fit_sample(data_x, data_y)
        print(data_x_sampled.shape[0] - data_x.shape[0], 'new random picked points')
        data_x = data_x_sampled
        data_y  = data_y_sampled.reshape(-1, 1)
    #standard_x=StandardScaler().fit_transform(data_x)
    standard_x = (data_x - data_x.mean())/data_x.std()
    return np.hstack((standard_x, np.array(data_y)))



if __name__ == "__main__":
    seed= 162
    EPOCHS = 842
    print("selected random seed {}".format(seed))
    np.random.seed(seed)
    # X,y  =  make_classification(n_samples=1000, n_features=4, n_informative=4, n_classes=4, n_redundant=0, n_repeated=0, random_state=123)
    # data_X = pd.DataFrame(X)
    # data_y = pd.DataFrame(y)
    # data = pd.concat([data_X, data_y], axis=1, sort=False)

    # print(X)
    # print(y)
    data_raw = pd.read_csv("BSOM_Dataset_for_HW3.csv")
    data = data_raw[["all_mcqs_avg_n20","all_NBME_avg_n4", "CBSE_01" , "CBSE_02","LEVEL"]]
    data=data.dropna()
    # train=data.sample(frac=0.8, random_state=seed)
    # validation = train.sample(frac=0.2, random_state=seed)
    # train = train.drop(validation.index)
    # test = data.drop(train.index)
    #
    # train = preprocess_data(train)
    # validation = preprocess_data(validation)
    # test = preprocess_data(test)


    # using stratified split to ensure all classes are present while testing
    pre_processed_data =  preprocess_data(data,upsample=False)
    train, test= train_test_split(pre_processed_data, shuffle=True, test_size=0.2, random_state=seed, stratify=pre_processed_data[:, -1]) #stratify=pre_processed_data[:, -1]
    train, validation = train_test_split(train, shuffle=True, test_size=0.2, random_state=seed, stratify=train[:, -1])
    X_train, X_test, X_val= train[:,:-1].astype(np.float64), test[:,:-1].astype(np.float64), validation[:,:-1].astype(np.float64)
    y_train, y_test, y_val = train[:,-1], test[:,-1], validation[:,-1]

    # sns.barplot(np.unique(y_train, return_counts=True)[0], np.unique(y_train, return_counts=True)[1])
    # plt.title("Class Distribution of training data")
    # plt.show()
    # sns.barplot(np.unique(y_val, return_counts=True)[0], np.unique(y_val, return_counts=True)[1])
    # plt.title("Class Distribution of validation data")
    # plt.show()
    # sns.barplot(np.unique(y_test, return_counts=True)[0], np.unique(y_test, return_counts=True)[1])
    # plt.title("Class Distribution of test data")
    # plt.show()

    print("=" * 10)
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_val, return_counts=True))
    print(np.unique(y_test, return_counts=True))
    print("=" * 10)
    # converting strings to numerics labels
    y_train = np.array(pd.get_dummies(y_train), dtype=np.int32)
    y_test= np.array(pd.get_dummies(y_test), dtype=np.int32)
    y_val = np.array(pd.get_dummies(y_val), dtype=np.int32)

    #y_train, y_test = train_test_split(np.array(labels), test_size=0.2, random_state=seed)
    cost_per_lambda = []
    for _lambda in [0.0]: #
        model = Model()
        n_cols = X_train.shape[-1]
        model.add(Dense(5, activation=sigmoid ,input_shape=(n_cols,)))
        # model.add(Dense(5, activation=sigmoid))
        # model.add(Dense(2, activation=sigmoid))
        model.add(Dense(4, activation=sigmoid))
        # model.add(Dense(6, activation=relu))
        # model.add(Dense(4, activation=relu))
        #model.add(Dense(4))
        model.compile(learning_rate=0.5, initializer="random", epsilon=0.1)
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1,1)

        #EPOCHS = 2500
        model.fit(X_train, y_train, X_val, y_val, regularization=L2(_lambda=_lambda), epochs=EPOCHS, validation_split=0.2 ,batch_size=len(X_train))
        print("Training metrics for lambda ", _lambda)
        pred = model.predict(X_train)
        print(classification_report(y_true=np.argmax(y_train, axis=1), y_pred=np.argmax(pred, axis=1))) #labels=np.unique(np.argmax(y_train, axis=1))
        #print(f1_score(y_true=np.argmax(y_train, axis=1), y_pred=np.argmax(pred, axis=1),average="weighted"))
        print("Test metrics for lambda ",_lambda)
        pred = model.predict(X_test)
        #cost = self.cost_function()
        print(classification_report(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(pred, axis=1))) #labels=np.unique(np.argmax(y_test, axis=1))
        #print(f1_score(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(pred, axis=1),average="weighted"))
        #print(np.argmax(pred,axis=1))

        # if len(y.shape) > 1:
        #     model.fit(X, y, epochs=1)
            #print(model.predict(X))
        # else:
        #     print(model.fit(X, y.reshape(-1, 1), epochs=20))
        print("=======training cost and validation cost minimas for lambda {}===========".format(_lambda))
        index=np.argmin(model.cost_per_epoch)
        print(model.cost_per_epoch[index], index+1)

        index = np.argmin(model.validation_cost_per_epoch)
        print(model.cost_per_epoch[index], index+1)

        print("=======training loss and validation loss minimas for lambda {}===========".format(_lambda))
        index = np.argmin(model.loss_per_epoch)
        print(model.loss_per_epoch[index], index+1)

        index = np.argmin(model.validation_loss_per_epoch)
        print(model.loss_per_epoch[index], index+1)
        print("============================================================")


        plt.title("Cost vs epoch for lambda {}".format(_lambda))
        plt.plot(model.cost_per_epoch, label="training_cost")
        plt.plot(model.validation_cost_per_epoch, label="validation_cost")
        plt.legend()
        #plt.show()
        plt.savefig("Fig/Cost-vs-epoch-{}-{}.png".format(_lambda, EPOCHS))

        plt.title("loss vs epoch for lambda {}".format(_lambda))
        plt.plot(model.loss_per_epoch, label="training_loss")
        plt.plot(model.validation_loss_per_epoch, label="validation_loss")
        plt.legend()
        #plt.show()
        plt.savefig("Fig/loss-vs-epoch-{}-{}.png".format(_lambda, EPOCHS))



    #model.add(Dense(10, activation=relu))# for all testing purposes

    # from keras.models import Sequential
    # from keras.layers import Dense
    # from sklearn.datasets import make_classification, make_multilabel_classification
    # from keras.optimizers import  SGD
    # from random import seed
    #
    # # X, y= make_multilabel_classification(n_samples=10, n_features=4, n_classes=4, n_labels=1, random_state=123)
    # model = Sequential()
    # ncols= X_train.shape[-1]
    # model.add(Dense(3, activation="relu",input_shape=(ncols,)))
    # model.add(Dense(2, activation="relu"))
    # model.add(Dense(4, activation="relu"))
    # model.compile(optimizer=SGD(lr=0.001) , loss='mean_squared_error', metrics=["accuracy"])
    # model.summary()
    # seed(123)
    # model.fit(X_train,y_train, epochs=EPOCHS, batch_size=len(X_train))
    #
    # print(np.argmax(model.predict(X_test), axis=1))
    #print(model)
    #model.add()
    #model.summary()
