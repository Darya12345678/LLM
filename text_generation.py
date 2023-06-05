import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import to_categorical

# Исходные данные
text = 'This is a sample text for language modeling'
seq_length = 10

# Создание словаря символов
chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# Генерация входных и выходных последовательностей
dataX = []
dataY = []
for i in range(0, len(text) - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

# Преобразование входных последовательностей в форму LSTM
X = np.reshape(dataX, (len(dataX), seq_length, 1))
X = X / float(len(chars))
y = to_categorical(dataY)

# Создание модели LSTM
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Обучение модели
model.fit(X, y, epochs=20, batch_size=128)

# Генерация текста на основе модели
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print('Seed:')
print('"' + ''.join([chars[value] for value in pattern]) + '"')

for i in range(100):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(chars))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = chars[index]
    seq_in = [chars[value] for value in pattern]
    print(result, end='', flush=True)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]