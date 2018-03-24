from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from matplotlib import pyplot as plt

digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))

print('Data records: ', len(images_and_labels))
print("shape of raw image data: {0}". format(digits.images.shape))
print("shape of data: {0}". format(digits.data.shape))

# Divide data set: training set, testing set
Xtrain, Xtest, Ytrain, Ytest = train_test_split(digits.data, digits.target, test_size=0.20, random_state=2)

plt.figure(figsize=(8, 6), dpi=200)

# for index, (image, label) in enumerate(images_and_labels[:8]):
#     plt.subplot(2, 4, index+1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Digit: %i' % label, fontsize=20)


# Start training the model by SVM
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(Xtrain, Ytrain)

# Prediction
Ypred = clf.predict(Xtest)

print('SVM model accuracy:', accuracy_score(Ytest, Ypred))
print('SVM model score result:', clf.score(Xtest, Ytest))

# plot testing result (first 4)
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig = fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.text(0.05, 0.05, str(Ypred[i]), fontsize=32, transform=ax.transAxes, color='green' if Ypred[i] == Ytest[i] else 'red')
    ax.text(0.8, 0.05, str(Ytest[i]), fontsize=32, transform=ax.transAxes, color='black')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

# save satisfied model
model_name = 'basic_digit_svm.pkl'
joblib.dump(clf, model_name)

# import dumped model
clf = joblib.load(model_name)
Ypred_new = clf.predict(Xtest)
clf.score(Xtest, Ypred_new)