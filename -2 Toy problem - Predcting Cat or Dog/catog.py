from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import pandas as pd

NAME = "TensorBoard_Catog"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("test.pickle","rb")
Test = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("id.pickle","rb")
Id = pickle.load(pickle_in)
pickle_in.close()

X = X/255.0
Test = Test/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3,callbacks=[tensorboard])
mypre = model.predict(Test)

pickle_out = open("result.pickle","wb")
pickle.dump(mypre, pickle_out)
pickle_out.close()

a = mypre.T.reshape(5000,)
fin_pre = a.round().astype(int)

mysub = pd.DataFrame({'id':Id,'label':fin_pre}).sort(['id'])
mysub.sort_values(['id'],inplace=True)
mysub.to_csv('mysub.csv', index = False)