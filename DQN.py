from keras.models import Sequential
from keras.layers import Conv2D, Flatten
from keras.optimizers import Adam, schedules
from keras import Input
# import keras


class MyLRSchedule(schedules.LearningRateSchedule):
    def __init__(self, lr, lr_min, lr_decay):
        self.lr = lr
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        self.lr_decay_steps = 10000

    def __call__(self, step):
        # return max(self.lr * self.lr_decay ^ (step / self.lr_decay_steps), self.lr_min)
        return self.lr * self.lr_decay ^ (step / self.lr_decay_steps)


def create_dqn(lr, lr_dr, lr_min, input_shape, num_filters): 
    lr_schedule = schedules.ExponentialDecay(
        lr,
        decay_steps=10000,
        decay_rate=lr_dr,
        staircase=False)

    model = Sequential()
    model.add(Input(shape = input_shape+(9,)))
    model.add(Conv2D(num_filters, (3,3), padding='same', activation='relu', use_bias=True, 
                    data_format='channels_last'))
    model.add(Conv2D(num_filters, (3,3), padding='same', activation='relu', use_bias=True))
    model.add(Conv2D(num_filters, (3,3), padding='same', activation='relu', use_bias=True))
    model.add(Conv2D(num_filters, (3,3), padding='same', activation='relu', use_bias=True))
    model.add(Conv2D(num_filters, (3,3), padding='same', activation='relu', use_bias=True))
    model.add(Conv2D(num_filters, (3,3), padding='same', activation='relu', use_bias=True))
    model.add(Conv2D(1, (1,1), padding='same', activation='linear', use_bias=True))
    model.add(Flatten())
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mse')

    return model


if __name__ == "__main__":
    model = create_dqn(0.001, 0.9956, 0.0001, (10,20), 64)
    model.summary()
