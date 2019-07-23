import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from classify.preprocess import process_data

# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def train_svm(finetune, x_train, x_test, x_val,
              train_data, test_data, val_data, add_extra):
    _, [train_labels, test_labels, val_labels], _ = process_data(train_data, test_data, val_data)

    if finetune:
        print("# ---------- Fine tuning SVM -----------#")
        param_grid = [
            {'C': [1, 10], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']},
        ]

        # do cross validation on train + val
        grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5)
        grid_search.fit(np.concatenate((x_train, x_val), axis=0), (train_labels + val_labels))

        print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))

        model = svm.SVC(kernel=grid_search.best_params_['kernel'], C=grid_search.best_params_['C'],
                        gamma=grid_search.best_params_['gamma'])

    else:
        model = svm.SVC(kernel='rbf', C=1, gamma=0.1)


    # SVM requires 2D inputs
    if len(x_train.shape) == 3:
        # reshape 3D to 2D:
        nsamples, nx, ny = x_train.shape
        x_train = x_train.reshape((nsamples, nx * ny))

        nsamples, nx, ny = x_val.shape
        x_val = x_val.reshape((nsamples, nx * ny))

        nsamples, nx, ny = x_test.shape
        x_test = x_test.reshape((nsamples, nx * ny))


    print("Fitting model")
    model.fit(x_train, train_labels)
    # Evaluate
    print("Evaluating model")
    acc_train = model.score(x_train, train_labels)
    acc_val = model.score(x_val, val_labels)
    acc_test = model.score(x_test, test_labels)
    # Predict Output
    print("Predict output")
    predicted = model.predict(x_test)

    f1 = f1_score(test_labels, predicted)
    recall = recall_score(test_labels, predicted)
    precision = precision_score(test_labels, predicted)

    return [acc_train, acc_val, acc_test, recall, precision, f1], predicted
