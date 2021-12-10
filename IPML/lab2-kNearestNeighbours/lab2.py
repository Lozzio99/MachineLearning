import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import average
from pandas import unique
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error


# %%
class kNN:
    def __init__(self, k=3, exp=2):
        # constructor for kNN classifier
        # k is the number of neighbor for local class estimation
        # exp is the exponent for the Minkowski distance
        self.s = 1e-6
        self.k = k
        self.exp = exp

    def fit(self, xtr, ytr):
        # training k-NN method X_train is the training data given with input attributes. n-th row correponds to n-th
        # instance. Y_train is the output data (output vector): n-th element of Y_train is the output value for n-th
        # instance in X_train.
        self.X_train = xtr
        self.Y_train = ytr

    def get_discrete_classification(self, X_test):
        # predict-class k-NN method
        # X_test is the test data given with input attributes. Rows correpond to instances
        # Method outputs prediction vector Y_pred_test:  n-th element of Y_pred_test is the prediction
        # for n-th instance in X_test

        Y_pred_test = []  # prediction vector Y_pred_test for all the test instances
        # in X_test is initialized to empty list []

        for i in range(len(X_test)):  # iterate over all instances in X_test
            test_instance = X_test.iloc[i]  # i-th test instance

            distances = []  # list of distances of the i-th test_instance for all the
            # train_instance s in X_train, initially empty.

            for j in range(len(self.X_train)):  # iterate over all instances in X_train
                train_instance = self.X_train.iloc[j]  # j-th training instance
                distance = self.minkowski_distance(test_instance,
                                                   train_instance)  # distance between i-th test instance and j-th
                # training instance
                distances.append(distance)  # add the distance to the list of distances of the i-th test_instance

            # Store distances in a dataframe. The dataframe has the index of Y_train in order to keep the
            # correspondence with the classes of the training instances
            df_dists = pd.DataFrame(data=distances, columns=['dist'], index=self.Y_train.index)

            # Sort distances, and only consider the k closest points in the new dataframe df_knn
            df_nn = df_dists.sort_values(by=['dist'], axis=0)
            df_knn = df_nn[:self.k]

            # Note that the index df_knn.index of df_knn contains indices in Y_train of the k-closed training
            # instances to the i-th test instance. Thus, the dataframe self.Y_train[df_knn.index] contains the
            # classes of those k-closed training instances. Method value_counts() computes the counts (number of
            # occurrences) for each class in self.Y_train[df_knn.index] in dataframe predictions.
            predictions = self.Y_train[df_knn.index].value_counts()

            # the first element of the index predictions.index contains the class with the highest count; i.e. the
            # prediction y_pred_test.
            y_pred_test = predictions.index[0]
            # add the prediction y_pred_test to the prediction vector Y_pred_test for all the test instances in X_test
            Y_pred_test.append(y_pred_test)

        return Y_pred_test

    def minkowski_distance(self, x1, x2):
        # computes the Minkowski distance of x1 and x2 for two labeled instances (x1,y1) and (x2,y2)

        # Set initial distance to 0
        distance = 0

        # Calculate Minkowski distance using the exponent exp
        for i in range(len(x1)):
            distance = distance + abs(x1[i] - x2[i]) ** self.exp

        distance = distance ** (1 / self.exp)

        return distance

    def get_class_probs(self, X_test):
        # function to evaluate posterior classes probabilities
        classes = unique(self.Y_train)  # set of unique classes
        # create dataframe with columns = prob(class_i) and index = test_instances.index
        probs = pd.DataFrame(data=np.zeros((len(X_test.index), len(classes))),
                             columns=classes,
                             index=X_test.index)

        for i in range(len(X_test)):  # for each test instance
            predictions = self.predict(X_test, i)
            # calculate probability for each class
            for x in predictions.index:
                instance = probs.iloc[i]
                instance.loc[x] = self.get_class_probability(predictions[x], len(classes))

        # return join of the two tables
        return pd.concat([X_test, probs], axis=1)

    def get_class_probability(self, n_instances_class_i, n_classes):
        # function to evaluate the probability for an instance to belong to a certain class
        # given #(instances of class i) and total #classes
        return (n_instances_class_i + self.s) / (self.k + n_classes * self.s)

    def get_prediction(self, X_test):
        vals = pd.DataFrame(data=np.zeros((len(X_test.index), 1)),
                            columns=['regression value'],
                            index=X_test.index)
        for i in range(len(X_test)):
            predictions = self.predict(X_test, i)
            vals.iloc[i] = average(predictions.index)
        return pd.concat([X_test, vals], axis=1)

    def predict(self, X_test, i):
        distances = []
        for j in range(len(self.X_train)):  # find neighbours
            distance = self.minkowski_distance(X_test.iloc[i], self.X_train.iloc[j])
            distances.append(distance)

        df_dists = pd.DataFrame(data=distances, columns=['dist'], index=self.Y_train.index)
        df_knn = df_dists.sort_values(by=['dist'], axis=0)[:self.k]
        return self.Y_train[df_knn.index].value_counts()  # select k-nearest


##################################################
# Data preparation: Normalizing dataset
##################################################
def normalize(df):
    return (df - df.min()) / (df.max() - df.min())


# %%
##################################################
# Data read: Training and Test set creation
##################################################
def read_data(file_name):
    data = pd.read_csv(file_name)
    y = data['class']
    x = data.drop(['class'], axis=1)
    data.head()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.34, random_state=10)
    return x_train, x_test, y_train, y_test


# %%
##################################################
# Hold-out testing: Variation on k parameter
##################################################
def hold_out_k_range(xtr, xt, ytr, yt, k_range, title):
    tr_acc = np.zeros(len(k_range))
    t_acc = np.zeros(len(k_range))
    i = 0
    for k in k_range:
        print("testing kNN with k=", k)
        clf = kNN(k)
        clf.fit(xtr, ytr)
        yp_tr = clf.get_discrete_classification(xtr)
        yp_t = clf.get_discrete_classification(xt)
        tr_acc[i] = accuracy_score(ytr, yp_tr)
        t_acc[i] = accuracy_score(yt, yp_t)
        i += 1
    #########################################
    # Plot of training and test accuracies
    #########################################
    plt.plot(k_range, tr_acc, 'ro-', k_range, t_acc, 'bv--')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.show()
    return


##################################################
# Hold-out testing: Variation on exp parameter
##################################################
def hold_out_exp_range(xtr, xt, ytr, yt, exp_range, title):
    tr_acc = np.zeros(len(exp_range))
    t_acc = np.zeros(len(exp_range))
    i = 0
    for exp in exp_range:
        print("testing kNN with exp=", exp)
        clf = kNN(k=3, exp=exp)
        clf.fit(xtr, ytr)
        yp_tr = clf.get_discrete_classification(xtr)
        yp_t = clf.get_discrete_classification(xt)
        tr_acc[i] = accuracy_score(ytr, yp_tr)
        t_acc[i] = accuracy_score(yt, yp_t)
        i += 1
    #########################################
    # Plot of training and test accuracies
    #########################################
    plt.plot(exp_range, tr_acc, 'ro-', exp_range, t_acc, 'bv--')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('exp')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.show()
    return


##################################################
# Hold-out testing: Variation on k parameter
##################################################
def hold_out_regression_value_k_range(xtr, xt, ytr, yt, k_range, title):
    tr_acc = np.zeros(len(k_range))
    t_acc = np.zeros(len(k_range))
    i = 0
    for k in k_range:
        print("testing kNN with k=", k)
        clf = kNN(k)
        clf.fit(xtr, ytr)
        yp_tr = clf.get_prediction(xtr)['regression value']
        yp_t = clf.get_prediction(xt)['regression value']
        tr_acc[i] = mean_abs_err(ytr.values, yp_tr)
        t_acc[i] = mean_abs_err(yt.values, yp_t)
        i += 1
    #########################################
    # Plot of training and test accuracies
    #########################################
    plt.plot(k_range, tr_acc, 'ro-', k_range, t_acc, 'bv--')
    plt.legend(['Training Error', 'Test Error'])
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.title(title)
    plt.show()
    return


def mean_abs_err(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def accuracy_testing_k_range(x_train, x_test, y_train, y_test, s):
    hold_out_k_range(x_train, x_test, y_train, y_test, k_r, s + ' non-normalized k range')
    hold_out_k_range(normalize(x_train), normalize(x_test), y_train, y_test, k_r, s + ' normalized k range')


def accuracy_testing_exp_range(x_train, x_test, y_train, y_test, s):
    hold_out_exp_range(x_train, x_test, y_train, y_test, exp_r, s + ' non-normalized exp range')
    hold_out_exp_range(normalize(x_train), normalize(x_test), y_train, y_test, exp_r, s + ' normalized exp range')


k_r = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]

exp_r = [2, 100, 1000, 10000]

# Test  the kNN classifier  on  the diabetes and glass classification  data  sets
# for the case when the data is not normalized and the case when the data is normalized.
# Indicate whether the training and hold-out accuracy rates improve with normalization.
filename = '../data/autoprice.csv'
# X_Train, X_Test, Y_Train, Y_Test = read_data(filename)
# accuracy_testing_k_range(X_train, X_test, Y_train, Y_test, 'diabetes')

# %%
# filename = 'data/glass.csv'
X_Train, X_Test, Y_Train, Y_Test = read_data(filename)
hold_out_regression_value_k_range(X_Train, X_Test, Y_Train, Y_Test, k_r, 'non-normalized autoprice')
hold_out_regression_value_k_range(normalize(X_Train), normalize(X_Test), Y_Train, Y_Test, k_r, 'normalized autoprice')
exit()
# %%
accuracy_testing_k_range(X_Train, X_Test, Y_Train, Y_Test, 'glass')
accuracy_testing_exp_range(X_Train, X_Test, Y_Train, Y_Test, 'glass')
