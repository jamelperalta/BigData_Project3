import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
from keras import optimizers

# **************** PREPARING DATA ************************

# getting the data for model training and tweets
data = pd.read_csv("/usr/jamelperalta/jamel_project3/Code/Input/cleantextlabels7.csv")
testData = pd.read_csv("/usr/jamelperalta/jamel_project3/Code/Input/tweets.csv", dtype=object, error_bad_lines=False)

trainingSize = int(len(data) * .8)
print ("Train size: %d" % trainingSize)
print ("Test size: %d" % (len(data) - trainingSize))

trainingPosts = data['post'][:trainingSize]
trainingTags = data['tags'][:trainingSize]

testingPosts = data['post'][trainingSize:]
testingTags = data['tags'][trainingSize:]

projectPostsSize = len(testData)
projectPosts = testData['words'][:projectPostsSize]

# only work with the 3000 most popular words found in our dataset
maxWords = 1000
# creating a new tokenizer
tokenize = text.Tokenizer(num_words=maxWords, char_level=False)
# feed the tweet to the tokenizer
tokenize.fit_on_texts(trainingPosts)

xTrain = tokenize.texts_to_matrix(trainingPosts)
xTest = tokenize.texts_to_matrix(testingPosts)
xProject = tokenize.texts_to_matrix(projectPosts)

encoder = LabelEncoder()
encoder.fit(trainingTags)
yTrain = encoder.transform(trainingTags)
yTest = encoder.transform(testingTags)

numClasses = np.max(yTrain) + 1
yTrain = utils.to_categorical(yTrain, numClasses)
yTest = utils.to_categorical(yTest, numClasses)

print('xTrain shape:', xTrain.shape)
print('xTest shape:', xTest.shape)
print('yTrain shape:', yTrain.shape)
print('yTest shape:', yTest.shape)

batch_size = 32
epochs = 4

# **************** TRAIN NETWORK ************************

model = Sequential()
model.add(Dense(256, input_shape=(maxWords,)))
model.add(Activation('relu'))
model.add(Dense(512, input_shape=(maxWords,)))
model.add(Activation('relu'))
model.add(Dense(256, input_shape=(maxWords,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(numClasses))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# **************** EXECUTE MODEL ************************

history = model.fit(xTrain, yTrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2)

score = model.evaluate(xTest, yTest,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

textLabels = encoder.classes_

# Setting the values into an array
postsArr = []
for i in range(projectPostsSize):
    prediction = model.predict(np.array([xProject[i]]))
    predicted_label = textLabels[np.argmax(prediction)]
    post = projectPosts.iloc[i]
    row = (str(post), str(predicted_label))
    postsArr.append(row)

# Setting the array into a DataFrame
df = pd.DataFrame(np.array(postsArr))
df.columns = ['Tweet', 'Labels']
result = df.groupBy("Labels").count()

# output the csv
result.to_csv("/usr/jamelperalta/jamel_project3/Code/Output/jameloutput2.csv")
