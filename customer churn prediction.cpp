#include <iostream>
#include <vector>
#include <cmath>

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}
std::vector<double> trainLogisticRegression(const std::vector<std::vector<double>>& X, 
                                            const std::vector<double>& y, 
                                            double learning_rate, int epochs) {
    int m = X.size();          // Number of samples
    int n = X[0].size();       // Number of features
    std::vector<double> weights(n, 0.0); // Initialize weights to zero

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<double> gradients(n, 0.0);

        for (int i = 0; i < m; ++i) {
            double z = 0.0;
            for (int j = 0; j < n; ++j) {
                z += weights[j] * X[i][j];
            }
            double prediction = sigmoid(z);
            double error = prediction - y[i];

            for (int j = 0; j < n; ++j) {
                gradients[j] += error * X[i][j];
            }
        }

        // Update weights
        for (int j = 0; j < n; ++j) {
            weights[j] -= learning_rate * gradients[j] / m;
        }
    }

    return weights;
}double predict(const std::vector<double>& weights, const std::vector<double>& features) {
    double z = 0.0;
    for (int i = 0; i < weights.size(); ++i) {
        z += weights[i] * features[i];
    }
    return sigmoid(z);
}
int main() {
    // Example dataset with 2 features per customer (X) and labels (y)
    std::vector<std::vector<double>> X = {
        {1.0, 30.0},
        {1.0, 40.0},
        {1.0, 50.0},
        {1.0, 60.0},
        {1.0, 70.0}
    };
    std::vector<double> y = {0, 0, 1, 1, 1}; // 0 - No churn, 1 - Churn

    // Train the logistic regression model
    double learning_rate = 0.01;
    int epochs = 1000;
    std::vector<double> weights = trainLogisticRegression(X, y, learning_rate, epochs);

    // Test with a new sample
    std::vector<double> new_customer = {1.0, 45.0};
    double probability = predict(weights, new_customer);
    std::cout << "Churn probability: " << probability << std::endl;

    if (probability > 0.5) {
        std::cout << "Customer is likely to churn." << std::endl;
    } else {
        std::cout << "Customer is likely to stay." << std::endl;
    }

    return 0;
}
