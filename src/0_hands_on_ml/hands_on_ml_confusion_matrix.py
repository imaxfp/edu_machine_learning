from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, \
    accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == '__main__':
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml('mnist_784', version=1)
    mnist.keys()
    print(mnist.keys())

    # Get 70000 images. Each image has 784 features
    # Image 28x28 pixels == 784
    # Each pixel represent 0 - 256
    X, y = mnist['data'], mnist['target']
    binary_label_5_not5 = list(map(lambda x: int(x == '5'), y))

    #
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # some_digit = X[1]
    # some_digit_image = some_digit.reshape(28, 28)
    # plt.imshow(some_digit_image, cmap='binary')
    # plt.axis('off')
    # plt.show()

    label = y[1]
    print(label)

    # Train and test random splitting
    # X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    X_train, X_test, y_train, y_test = train_test_split(X, binary_label_5_not5, test_size=0.4, random_state=0)

    # However, by partitioning the available data into three sets, we drastically reduce the number
    # of samples which can be used for learning the model, and the results can depend on a particular
    # random choice for the pair of (train, validation) sets.

    # The training set is split into k smaller sets (other approaches are described below,
    # but generally follow the same principles). The following procedure is followed for each of the k “folds”

    '''
    Normally you split the data into 3 sets.
    Training: used to train the model and optimize the model’s hyperparameters.
    
    Testing: used to check that the optimized model works on unknown data to test that the model generalizes well
    
    Validation: during optimizing some information about test set leaks into the model by your choice of the parameters 
    so you perform a final check on completely unknown data
    
    Thanks to cross validation you perform multiple train_test split and while one fold can achieve extraordinary good results the other might underperform.
    '''

    # Stochastic gradient Descent
    # sgd_clf = SGDClassifier(random_state=42)
    # sgd_clf.fit(X_train, y_train)
    # predictions = sgd_clf.predict(X_test)

    # Measure Accuracy Using Cross-Validation
    # https://scikit-learn.org/stable/modules/cross_validation.html

    # TODO Confusion Matrix
    '''
    Main idea - Count the number of times instances of class A a classified as class B
    “Good confusion matrix usage separates a good data scientist from a hack.” "C"
    
    It is a performance measurement for machine learning classification problem where output can be two or more classes
    For binary classification, these are the True Positive, True Negative, False Positive and False Negative categories.
    
    
    True Positive:  
        You predicted positive and it’s true. You predicted that a woman is pregnant and she actually is.
    True Negative:  
        You predicted negative and it’s true. You predicted that a man is not pregnant and he actually is not.
    False Positive: (Type 1 Error)
        You predicted positive and it’s false. You predicted that a man is pregnant but he actually is not.
    False Negative: (Type 2 Error) 
        You predicted negative and it’s false. You predicted that a woman is not pregnant but she actually is.
        Just Remember, We describe predicted values as Positive and Negative and actual values as True and False.

     
    '''
    logreg = LogisticRegression(C=1e5)
    logreg.fit(X_train, y_train)
    # Generate predictions with the model using our X values
    y_pred = logreg.predict(X)
    # Get the confusion matrix
    cf_matrix = confusion_matrix(binary_label_5_not5, y_pred)

    import seaborn as sns

    # Show confusion matrix sectors
    labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.show()

    # Show confusion matrix sectors in percentage
    sns.heatmap(cf_matrix, annot=True)
    sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
    plt.show()

    true_negative = cf_matrix[0][0]
    true_positive = cf_matrix[1][1]

    false_positive = cf_matrix[0][1]
    false_negative = cf_matrix[1][0]

    '''
    Accuracy 
    The number of correct predictions divided by the total number of predictions. 
    
    In many cases in which classification accuracy is not a good indicator of your model performance:
     1. When your class distribution is imbalanced (one class is more frequent than others)
     2.      
    '''
    all = true_positive + true_negative + false_positive + false_negative
    all_correct = true_positive + true_negative
    accuracy = (all_correct / all)
    print("Accuracy = ", accuracy)
    print("Accuracy sklearn = ", accuracy_score(binary_label_5_not5, y_pred))

    '''
    Precision = TP / TP + FP
    
    Precision is a good measure to determine, when the costs of False Positive is high.
    For instance, email spam detection. 
    In email spam detection, a false positive means that an email that is non-spam (actual negative)
    has been identified as spam (predicted spam). The email user might lose important emails 
    if the precision is not high for the spam detection model.
    '''

    precision = true_positive / (true_positive + false_positive)
    print("Precision = ", precision)
    print("Precision sklearn = ", precision_score(binary_label_5_not5, y_pred))

    '''
    Recall = TP / TP + FN
    
    Recall actually calculates how many of the Actual Positives our model capture through labeling it as (True Positive)
    Similarly, in sick patient detection. If a sick patient (Actual Positive) goes through the test 
    and predicted as not sick (Predicted Negative). The cost associated with False Negative will be extremely high
    if the sickness is contagious. 
    '''
    recall = true_positive / (true_positive + false_negative)
    print("Recall or TRP (true positive rate) = ", recall)
    print("Recall sklearn = ", recall_score(binary_label_5_not5, y_pred))

    '''
    F1 Score 2 * (precision * recall) / (precision + recall) = 2 / (1/precision + 2/recall)
    
    F1 is a function of Precision and Recall    
    F1 Score is needed when you want to seek a balance between Precision and Recall.
    '''
    f1 = 2 / (1 / precision + 2 / recall)
    print("F1 score = ", f1)
    print("F1 sclearn = ", f1_score(binary_label_5_not5, y_pred))

    '''
    Precision/Recall Trade of
    '''

    print("Classification report", classification_report(binary_label_5_not5, y_pred))

    '''
    Sensitivity and Specificity
    https://towardsdatascience.com/20-popular-machine-learning-metrics-part-1-classification-regression-evaluation-metrics-1ca3e282a2ce
    Sensitivity and specificity are two other popular metrics mostly used in medical and biology related fields, and are defined as:
    
    Sensitivity= Recall= TP/(TP+FN)
    Specificity= True Negative Rate= TN/(TN+FP)
    
    '''

    '''
    
    '''
