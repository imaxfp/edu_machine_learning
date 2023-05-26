if __name__ == '__main__':
    # create the range 1 to 25
    from pandas import np

    rn = range(1, 26)

    # %%

    from sklearn.model_selection import KFold

    kf3 = KFold(n_splits=3, shuffle=False)

    # %%

    # the Kfold function retunrs the indices of the data. Our range goes from 1-25 so the index is 0-24
    for train_index, test_index in kf3.split(rn):
        print("indexes of train and test")
        print(train_index, test_index)

    # %%

    # to get the values from our data, we use np.take() to access a value at particular index
    for train_index, test_index in kf3.split(rn):
        print("Elements train and test")
        print(np.take(rn, train_index), np.take(rn, test_index))

    # Important to say that the number of fold influences that size of your test set. 3 folds tests on 33% of the data
    # while 5 folds on 1/5 which equals to 20% of the data
