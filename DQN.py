from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam

def create_dqn(lr, input_shape, num_filters, num_nodes):
    model = Sequential()
    model.add(Conv2D(num_filters, (3,3), padding='same', activation='relu', use_bias=True, 
                     input_shape=input_shape+(9,)))
    model.add(Conv2D(num_filters, (3,3), padding='same', activation='relu', use_bias=True))
    model.add(Conv2D(num_filters, (3,3), padding='same', activation='relu', use_bias=True))
    model.add(Conv2D(num_filters, (3,3), padding='same', activation='relu', use_bias=True))
    model.add(Conv2D(num_filters, (3,3), padding='same', activation='relu', use_bias=True))
    model.add(Conv2D(num_filters, (3,3), padding='same', activation='relu', use_bias=True))
    model.add(Conv2D(1, (1,1), padding='same', activation='linear', use_bias=True))
    model.add(Flatten())

    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    return model


if __name__ == "__main__":
    model = create_dqn(0.01, (10,20), 64, 1)
    model.summary()
