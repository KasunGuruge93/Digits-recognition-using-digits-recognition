import matplotlib.pyplot as plt 
import numpy as np


from sklearn import datasets
digits = datasets.load_digits(n_class=10, return_X_y=False)

import matplotlib.pyplot as plt


#####################

def plot_multi(i):
    '''Plots 9 digits, starting with digit i'''
    nplots = 9
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(3,3,j+1)
        plt.imshow(digits.images[i+j], cmap='binary')
        plt.title(digits.target[i+j])
        plt.axis('off')
    plt.show()
plot_multi(7)



# Isolate the `digits` data
digits_data = digits.data

                                                ###### inspecting the digits

# Print the number of unique labels
number_digits = len(np.unique(digits.target))

# Isolate the `images`
digits_images = digits.images

print(digits.target[0])
print(digits.data[0])
print(digits.images[0])
#############################

y = digits.target
x = digits_data



x_train = x[:1000]
y_train = y[:1000]
x_test = x[1000:]
y_test = y[1000:]


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(15,), activation='logistic', alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.1, verbose=True)
mlp.fit(x_train,y_train)

predictions = mlp.predict(x_test)



from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
# we just look at the 1st 50 examples in the test sample

from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.heatmap(confusion_matrix(y_test,predictions),annot=True)