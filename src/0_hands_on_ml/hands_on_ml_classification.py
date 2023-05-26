from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml

if __name__ == '__main__':
    '''
    Multiclass Classification
    OvR - one-versus-the-rest
    OvO - one-versus-all 
    '''

    mnist = fetch_openml('mnist_784', version=1)
    mnist.keys()
    print(mnist.keys())

    X, y = mnist['data'], mnist['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    svm_clf = SVC()
    svm_clf.fit(X_train, y_train)  # all number range from 0 to 9
    y_pred = svm_clf.predict(X_test)

    print("F1 sclearn = ", classification_report(y_test, y_pred))
