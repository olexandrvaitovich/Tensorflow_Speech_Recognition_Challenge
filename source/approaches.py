from models import build_cnn_v1, build_nn_v1, build_nn_v2, build_cnn_lstm_v1, build_lstm_v1, build_cnn1d_v1
import numpy as np
from keras.optimizers import Adam, RMSprop
from time import time


class ApproachNN:
    def __init__(self, **kwargs):
        self._sets = kwargs

        model_types = {'CNN_V1': build_cnn_v1, 
                       'NN_V1':build_nn_v1, 
                       'NN_V2':build_nn_v2, 
                       'CNN_LSTM_V1': build_cnn_lstm_v1,
                       'LSTM_V1': build_lstm_v1,
                       'CNN1D_V1': build_cnn1d_v1}

        optimizers = {'Adam': Adam, 'RMSprop': RMSprop}

        self._model = model_types[self._sets['type']](self._sets['input_shape'])
        
        if self._sets['mode'] == 'inference':
            self._model.load_weights(self._sets['exp_dir']+"\\weights.h5")

        self._model.compile(optimizer=optimizers[self._sets['optimizer']](self._sets['lr'], decay=3e-4  ), metrics=['acc'],
                            loss='categorical_crossentropy')
        print(self._model.summary())


    def train(self, train_gen, val_gen):

        steps = int(self._sets['num_samples']/self._sets['batch_size'])

        val_loss = np.inf

        for e in range(self._sets['epochs']):
            start = time()

            print(f"Epoch {e}/{self._sets['epochs']}")

            Loss = []
            Acc = []

            for j in range(steps):
                X = []
                Y = []
                for k in range(self._sets['batch_size']):
                    x, y = next(train_gen)
                    X.append(x)
                    Y.append(y)

                X = np.array(X)
                Y = np.array(Y)
                self._model.train_on_batch(X, Y)

                loss, acc = self._model.evaluate(X, Y, verbose=0)

                print(f"{j}/{steps} - loss: {loss} - acc: {acc}")

                Loss.append(loss)
                Acc.append(acc)

            loss, acc, _ = self.evaluate(val_gen)

            print(f"{steps}/{steps} - Loss: {np.mean(Loss)} - Acc: {np.mean(Acc)} - Val_loss: {loss} - Val_acc: {acc}")
            print(f"Time: {time()-start}")

            if loss < val_loss:
                self._model.save_weights(self._sets['exp_dir']+"\\weights.h5")

    def evaluate(self, gen, verbose=False):
        from sklearn.metrics import accuracy_score, confusion_matrix

        total_loss = 0
        total_acc = 0
        samples_num = 0
        Y_true = []
        Y_pred = []

        firstX = None
        firstY = None
        gotFirst = False

        while True:
            x, y = next(gen)
            if not gotFirst:
                firstY = y[:]
                firstX = x[:]
                gotFirst = True
                continue
            x = np.expand_dims(x, 0)
            y = np.expand_dims(y, 0)
            loss, acc = self._model.evaluate(x, y, verbose=0)
            prediction = self._model.predict(x)
            acc = accuracy_score(y, prediction>0.5)
            Y_true.append(np.argmax(y))
            Y_pred.append(np.argmax(prediction))
            total_acc += acc
            total_loss += loss
            samples_num += 1
            if (x==firstX).all() and (y==firstY).all():
                break

        total_loss /= samples_num
        total_acc /= samples_num
        
        if verbose:
            print(f'Final loss: {total_loss} - Final acc: {total_acc}')
        
        return total_loss, total_acc, confusion_matrix(np.array(Y_true), np.array(Y_pred))

    def predict(self, X):
        return self._model.predict(X)


def build_nn(**kwargs):
    return ApproachNN(**kwargs)


