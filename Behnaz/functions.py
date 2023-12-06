# Function to fill the missing values with median
def fix_missing_value_median(train_df, col, test_df=None):

    train_df[col] = train_df[col].fillna(train_df[col].median())
    if test_df is not None:
        test_df[col] = test_df[col].fillna(test_df[col].median())



# Function for scaling (just need to change the name of the scaler if you want to use different one)
def scaling(train, test, numeric_columns):
    categorical_columns = ['device_brand', 'os', 'Wi-fi 802.11 2.4GHz', 'Wi-fi 802.11 5GHz', 'purchase_year']

    train_numeric = train[numeric_columns]
    test_numeric = test[numeric_columns]
    
    transformer = MinMaxScaler()
        
    train_numerical_scaled = pd.DataFrame(transformer.fit_transform(train_numeric), columns=numeric_columns)
    test_numerical_scaled = pd.DataFrame(transformer.transform(test_numeric), columns=numeric_columns)
    
    train_categoric = train[categorical_columns]
    test_categoric = test[categorical_columns]
    
    train_scaled = pd.concat([train_categoric.reset_index(drop=True), train_numerical_scaled.reset_index(drop=True)], axis=1)
    test_scaled = pd.concat([test_categoric.reset_index(drop=True), test_numerical_scaled.reset_index(drop=True)], axis=1)
    
    return train_scaled, test_scaled



# Function for knn imputation
def knn_imputation_train(data, col, data_test=None):
    data_column_imputation = data[[col]]

    imputer = KNNImputer(n_neighbors=11) 

    data_imputed_train = imputer.fit_transform(data_column_imputation)
    data_imputed_test = None
    if data_test is not None:
        data_imputed_test = imputer.transform(data_test[[col]])
    
    return data_imputed_train, data_imputed_test



# Function for lig transform
def log_transform(data, col):
    log_data = np.log(data[col].values - np.min(data[col].values) + 1)
    return log_data



# Function for binary encoding
def binary_encoding(data, col):
    encoded_data = np.where(data[col].str.contains("yes"), 1, 0)
    return encoded_data



# Function for one hot encoding
def one_hot_encoder(train, test, col):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    train_encoded = encoder.fit_transform(train[[col]])

    test_encoded = encoder.transform(test[[col]])

    feature_names = encoder.get_feature_names_out([col])

    train_encoded = pd.DataFrame(train_encoded, columns=feature_names)
    test_encoded = pd.DataFrame(test_encoded, columns=feature_names)

    train = pd.concat([train, train_encoded], axis=1)
    train.drop(col, axis=1, inplace=True)
    
    test = pd.concat([test, test_encoded], axis=1)
    test.drop(col, axis=1, inplace=True)
    
    return train, test



# Function for linear regression
def lin_reg(x_train, x_test, y_train):
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    y_predict = lin_reg.predict(x_test)
    return y_predict



# Function for adding bias
def add_ones(data):
    data_b = np.c_[np.ones((len(data), 1)), data]
    return data_b



# Function for Batch gradient descent
def batch_gradient_descent(x_train, x_test, y_train, y_test, eta=0.000000095, n_iterations=50,
                          return_values=False, plot_learning_curves=True, theta_multiply=None):
    
    # **** Notes ****
    # 1) I introduced the variable 'plot_learning_curves' to allow the option of not always plotting the learning curves.
    # 2) The varibale return_values is when i want to only return y_predicts (For point 5)
    # 3) Refer to the comments later to gain a better understanding of the variable 'theta_multiply'.
    
    # adding ones
    x_train = add_ones(x_train)
    x_test = add_ones(x_test)
    
    # checking if target is numpy
    if not isinstance(y_train, np.ndarray):
        y_train = y_train.values
    
    if not isinstance(y_test, np.ndarray):
        y_test = y_test.values
    
    np.random.seed(42)
    theta = np.random.randn(x_train.shape[1],1)
    if theta_multiply is not None:
        theta *= theta_multiply

    rmse_test = []
    rmse_train = []
    m = x_train.shape[0]
    for iteration in range(n_iterations):
        y_predict_train = x_train.dot(theta)

        gradients = (2 / m) * x_train.T.dot(y_predict_train - y_train)
        theta = theta - eta * gradients
        y_predict_test = x_test.dot(theta)

        rmse_test.append(np.sqrt(mean_squared_error(y_test, y_predict_test)))
        rmse_train.append(np.sqrt(mean_squared_error(y_train, y_predict_train)))
    
    # if we need the final precited and actual values just return them
    if return_values:
        return y_test, y_predict_test, y_train, y_predict_train

    # otherwise print the last rmses and plot the curves if needed
    else:
        # print the final rmse values
        print(f"Final rmse for train:{rmse_train[-1]}")
        print(f"Final rmse for test:{rmse_test[-1]}")
        if plot_learning_curves:
            plt.plot(range(1, n_iterations + 1), rmse_train, label="Train_data", color='blue')
            plt.plot(range(1, n_iterations + 1), rmse_test, label="Test_data", color='red')
            plt.xlabel('Number of Iterations')
            plt.ylabel('RMSE')
            plt.title('RMSE Comparison')
            plt.grid(True)
            plt.legend()
            plt.show()



# Function for finding the best eta for Batch gradient descent
def find_best_eta(x_train, x_test, y_test, y_train, n_iterations, theta_multiply=None, plot_eta_rms=True, eta_values=None):
    

    # **** Notes ****
    # 1) I introduced the variable 'plot_learning_curves' to allow the option of not always plotting the learning curves.
    # 2) Refer to the comments later to gain a better understanding of the variable 'theta_multiply'.


    # adding ones
    x_train = add_ones(x_train)
    x_test = add_ones(x_test)
    
    # checking if target is numpy
    if not isinstance(y_train, np.ndarray):
        y_train = y_train.values
    
    if not isinstance(y_test, np.ndarray):
        y_test = y_test.values
    
    if eta_values is None:
        eta_values = np.linspace(1e-10, 1e-7, 1000)  # Adjust the range of eta values
    
    best_eta = None
    best_rmse = float('inf')
    
    m = x_train.shape[0]
    
    all_rmse_test = []
    for eta in eta_values:
        np.random.seed(42)
        theta = np.random.randn(x_train.shape[1],1)
        if theta_multiply is not None:
            theta *= theta_multiply

        for iteration in range(n_iterations):
            gradients = (2/m) * x_train.T.dot(x_train.dot(theta) - y_train.reshape(-1, 1))
            theta = theta - eta * gradients

        # Make predictions on the test set
        y_pred = x_test.dot(theta)

        # Calculate RMSE on the test set
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        all_rmse_test.append(rmse_test)

        # Update the best learning rate if the current one is better
        if rmse_test < best_rmse:
            best_rmse = rmse_test
            best_eta = eta
    # if wanted plot the rmse values on test for each eta
    if plot_eta_rms:
        plt.plot(eta_values, all_rmse_test, label="Test data", color='blue')
        plt.xlabel('Number of Etas')
        plt.ylabel('RMSE')
        plt.title('RMSE for different eta values')
        plt.grid(True)
        plt.legend()
        plt.show()

    return best_eta, best_rmse



# Function for mini batch
def mini_batch_gradient_descent(x_train, x_test, y_train, y_test, eta=1e-8,
                              n_epochs=100, plot_learning_curves=True, theta_multiply=None):
    

    # **** Notes ****
    # 1) I introduced the variable 'plot_learning_curves' to allow the option of not always plotting the learning curves.
    # 2) Refer to the comments to gain a better understanding of the variable 'theta_multiply'.


    # adding ones
    x_train = add_ones(x_train)
    x_test = add_ones(x_test)
    
    # checking if target is numpy
    if not isinstance(y_train, np.ndarray):
        y_train = y_train.values
    
    if not isinstance(y_test, np.ndarray):
        y_test = y_test.values
    
    batch_size = 128
    np.random.seed(42)
    theta = np.random.randn(x_train.shape[1],1)
    if theta_multiply is not None:
        theta *= theta_multiply
    
    m = x_train.shape[0]  # number of samples in the training set
    n_batches = m // batch_size

    rmse_test = []
    rmse_train = []
    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = x_train[shuffled_indices]
        y_shuffled = y_train[shuffled_indices]

        for i in range(0, m, batch_size):
            xi = X_b_shuffled[i:i + batch_size]
            yi = y_shuffled[i:i + batch_size]
            gradients = 2 / batch_size * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - eta * gradients
        
        y_predict_train = x_train.dot(theta)
        y_predict_test = x_test.dot(theta)
        rmse_test.append(np.sqrt(mean_squared_error(y_test, y_predict_test)))
        rmse_train.append(np.sqrt(mean_squared_error(y_train, y_predict_train)))
    

    print(f"last rsme for train:{rmse_train[-1]}")
    print(f"last rsme for test:{rmse_test[-1]}")
    if plot_learning_curves:
        plt.plot(range(1, n_epochs + 1), rmse_train, label="Train_data", color='blue')
        plt.plot(range(1, n_epochs + 1), rmse_test, label="Test_data", color='red')
        plt.xlabel('Number of Epochs')
        plt.ylabel('RMSE')
        plt.title('RMSE Comparison')
        plt.grid(True)
        plt.legend()
        plt.show()



# Function for finding the best eta for mini batch
def find_best_eta_mini_batch(x_train, x_test, y_test, y_train, n_epochs, theta_multiply=None, plot_eta_rms=True, eta_values=None):
    

        
    # **** Notes ****
    # 1) I introduced the variable 'plot_learning_curves' to allow the option of not always plotting the learning curves.
    # 2) Refer to the comments to gain a better understanding of the variable 'theta_multiply'.
    

    # adding ones
    x_train = add_ones(x_train)
    x_test = add_ones(x_test)
    
    # checking if target is numpy
    if not isinstance(y_train, np.ndarray):
        y_train = y_train.values
    
    if not isinstance(y_test, np.ndarray):
        y_test = y_test.values
    
    if eta_values is None:
        eta_values = np.linspace(1e-11, 1e-8, 1000)  # Adjust the range of eta values
    
    best_eta = None
    best_rmse = float('inf')
    
    batch_size = 128
    m = x_train.shape[0]
    n_batches = m // batch_size

    
    all_rmse_test = []
    for eta in eta_values:
        np.random.seed(42)
        theta = np.random.randn(x_train.shape[1],1)
        if theta_multiply is not None:
            theta *= theta_multiply

        for epoch in range(n_epochs):
            shuffled_indices = np.random.permutation(m)
            X_b_shuffled = x_train[shuffled_indices]
            y_shuffled = y_train[shuffled_indices]

            for i in range(0, m, batch_size):
                xi = X_b_shuffled[i:i + batch_size]
                yi = y_shuffled[i:i + batch_size]
                gradients = 2 / batch_size * xi.T.dot(xi.dot(theta) - yi)
                theta = theta - eta * gradients

        # Make predictions on the test set
        y_pred = x_test.dot(theta)

        # Calculate RMSE on the test set
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        all_rmse_test.append(rmse_test)

        # Update the best learning rate if the current one is better
        if rmse_test < best_rmse:
            best_rmse = rmse_test
            best_eta = eta
    # if wanted plot the rmse values on test for each eta
    if plot_eta_rms:
        plt.plot(eta_values, all_rmse_test, label="Test data", color='blue')
        plt.xlabel('Number of Etas')
        plt.ylabel('RMSE')
        plt.title('RMSE for different eta values')
        plt.grid(True)
        plt.legend()
        plt.show()

    return best_eta, best_rmse



# Function to apply polynomial
def apply_poly(x_train, x_test, degree):
    poly = PolynomialFeatures(degree=degree)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)
    
    return x_train_poly, x_test_poly



# Function to apply Lasso regularization with the best alpha
def lasso_best_alpha(x_train, x_test, y_train, y_test):
    alpha_values = np.linspace(0, 0.01, 100)

    best_alpha = None
    best_rmse = float('inf')
    
    rmse_test_values = []
    rmse_train_values = []

    for alpha in alpha_values:
        lasso_reg = Lasso(alpha=alpha)
        lasso_reg.fit(x_train, y_train)
        y_pred_test = lasso_reg.predict(x_test)
        y_pred_train = lasso_reg.predict(x_train)

        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        rmse_test_values.append(rmse_test)
        
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_train_values.append(rmse_train)

        if rmse_test < best_rmse:
            best_rmse = rmse_test
            best_alpha = alpha

    return best_alpha, best_rmse, rmse_test_values, rmse_train_values



# Function for applying ridge regression
def ridge_best_alpha(x_train, x_test, y_train, y_test):
    alpha_values = np.linspace(0, 0.001, 100)

    best_alpha = None
    best_rmse = float('inf')
    
    rmse_values = []

    for alpha in alpha_values:
        lasso_reg = Ridge(alpha=alpha)
        lasso_reg.fit(x_train, y_train)
        y_pred = lasso_reg.predict(x_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_values.append(rmse)

        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    return best_alpha, best_rmse, rmse_values



# Function to get rmse only for one brand in general model (Task 5 project 1)
def rmse_general_model_brand(col, x_train, x_test, y_train, y_test):
    sample_weight_train = np.zeros(x_train.shape[0])
    sample_weight_train[x_train[col] == 1] = 1

    sample_weight_test = np.zeros(x_test.shape[0])
    sample_weight_test[x_test[col] == 1] = 1

    test_y, y_pred_test, train_y, y_pred_train = batch_gradient_descent(x_train, x_test, y_train, y_test, return_values=True)

    rmse_train = np.sqrt(mean_squared_error(train_y, y_pred_train, sample_weight=sample_weight_train))
    rmse_test = np.sqrt(mean_squared_error(test_y, y_pred_test, sample_weight=sample_weight_test))

    return rmse_train, rmse_test



# Function to calculate the difference between two rmse
def calculate_diff_rmse(rmse1, rmse2):
    perc = ((rmse1 - rmse2) / rmse2) * 100
    return perc