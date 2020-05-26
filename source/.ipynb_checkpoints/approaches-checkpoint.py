from models import build_cnn_v1
import numpy as np
from keras.optimizers import Adam, RMSprop
from time import time


class ApproachCNN:
    def __init__(self, **kwargs):
        self._sets = kwargs

        model_types = {'V1': build_cnn_v1}

        optimizers = {'Adam': Adam, 'RMSprop': RMSprop}

        self._model = model_types[self._sets['type']](self._sets['input_shape'])
        
        if self._sets['mode'] == 'inference':
            self._model.load_weights(self._sets['exp_dir']+"\\weights.h5")

        self._model.compile(optimizer=optimizers[self._sets['optimizer']](self._sets['lr']), metrics=['acc'],
                            loss='categorical_crossentropy')

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

                X = X.reshape(X.shape[0], X.shape[2], X.shape[3], X.shape[4])
                Y = Y.reshape(Y.shape[0], Y.shape[2])

                self._model.train_on_batch(X, Y)

                loss, acc = self._model.evaluate(X, Y, verbose=0)

                print(f"{j}/{steps} - loss: {loss} - acc: {acc}")

                Loss.append(Loss)
                Acc.append(Acc)

            loss, acc = self.evaluate(val_gen)

            print(f"{steps}/{steps} - loss: {np.mean(Loss)} - acc: {np.mean(Acc)} - val_loss: {loss} - val_acc: {acc}")
            print(f"Time: {time()-start}")

            if loss < val_loss:
                self._model.save_weights(self._sets['exp_dir']+"\\weights.h5")

    def evaluate(self, gen, verbose=False):
        from sklearn.metrics import accuracy_score

        total_loss = 0
        total_acc = 0

        for i in range(self._sets['num_val_samples']):
            x, y = next(gen)
            loss, acc = self._model.evaluate(x, y, verbose=0)
            prediction = self._model.predict(x)
            acc = accuracy_score(y, prediction>0.5)
            total_acc += acc
            total_loss += loss
        
        if verbose:
            print(f'final loss: {total_loss} - final acc: {total_acc}')
        
        return total_loss, total_acc

    def predict(self, X):
        return self._model.predict(X)


def build_cnn(**kwargs):
    return ApproachCNN(**kwargs)
