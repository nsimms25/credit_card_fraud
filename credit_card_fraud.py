"""
Using a neural network to train and predict if a credit card transaction is fraudulent.

"""

import pandas as pd
import numpy as np
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# Import the sequential model and dense layer
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('creditcard.csv')

# Analyze data attributes
# print(data.head())
# print(data.info())
# print(data.describe())
'''
   Time        V1        V2        V3  ...       V27       V28  Amount  Class
0   0.0 -1.359807 -0.072781  2.536347  ...  0.133558 -0.021053  149.62      0
1   0.0  1.191857  0.266151  0.166480  ... -0.008983  0.014724    2.69      0
2   1.0 -1.358354 -1.340163  1.773209  ... -0.055353 -0.059752  378.66      0
3   1.0 -0.966272 -0.185226  1.792993  ...  0.062723  0.061458  123.50      0
4   2.0 -1.158233  0.877737  1.548718  ...  0.219422  0.215153   69.99      0

[5 rows x 31 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
Time      284807 non-null float64
V1        284807 non-null float64
V2        284807 non-null float64
V3        284807 non-null float64
V4        284807 non-null float64
V5        284807 non-null float64
V6        284807 non-null float64
V7        284807 non-null float64
V8        284807 non-null float64
V9        284807 non-null float64
V10       284807 non-null float64
V11       284807 non-null float64
V12       284807 non-null float64
V13       284807 non-null float64
V14       284807 non-null float64
V15       284807 non-null float64
V16       284807 non-null float64
V17       284807 non-null float64
V18       284807 non-null float64
V19       284807 non-null float64
V20       284807 non-null float64
V21       284807 non-null float64
V22       284807 non-null float64
V23       284807 non-null float64
V24       284807 non-null float64
V25       284807 non-null float64
V26       284807 non-null float64
V27       284807 non-null float64
V28       284807 non-null float64
Amount    284807 non-null float64
Class     284807 non-null int64
dtypes: float64(30), int64(1)
memory usage: 67.4 MB
None
                Time            V1  ...         Amount          Class
count  284807.000000  2.848070e+05  ...  284807.000000  284807.000000
mean    94813.859575  3.919560e-15  ...      88.349619       0.001727
std     47488.145955  1.958696e+00  ...     250.120109       0.041527
min         0.000000 -5.640751e+01  ...       0.000000       0.000000
25%     54201.500000 -9.203734e-01  ...       5.600000       0.000000
50%     84692.000000  1.810880e-02  ...      22.000000       0.000000
75%    139320.500000  1.315642e+00  ...      77.165000       0.000000
max    172792.000000  2.454930e+00  ...   25691.160000       1.000000
'''
# After looking at the data there has been some processing done, all columns are of type float64 or int64.
# This will makes pre-processing much easier, and computation time much quicker versus strings for example.

# Separate labels from data.
labels = data['Class']
data = data.drop('Class', axis=1)
# print(labels.head())
# print(data.head())

# Train test split to train on one set and test on another, in order to gain better understanding of true accuracy.
orig_X_train, orig_X_test, orig_y_train, orig_y_test = train_test_split(data, labels)

# Create a Sequential Model for the original characteristics.
model1 = Sequential()

model1.add(Dense(50, input_shape=(30, ), activation='relu'))
model1.add(Dense(50, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))

# Compile your model
model1.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Display a summary of your model
model1.summary()

# Train your model for 20 epochs
model1.fit(orig_X_train, orig_y_train, epochs=5)

# Evaluate your model accuracy on the test set
accuracy = model1.evaluate(orig_X_test, orig_y_test)[1]

# Print accuracy
print('Accuracy:', accuracy)

'''
Final accuracy of the training output below.

Accuracy: 0.9985112777730962

The accuracy of the data set was surprising, the large amount of data and most likely the features helped the model
find many of the possible different ways the legitimate and fraudulent transactions occurred.  
'''
