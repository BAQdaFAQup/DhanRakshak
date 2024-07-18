DhanRakshak

--DhanRakshak is an advanced tool designed to detect and prevent online payment scams. By leveraging machine learning algorithms, DhanRakshak identifies suspicious transactions and provides real-time alerts to help users avoid falling victim to fraud. The project involves generating synthetic transaction data, preprocessing it, training and evaluating a Random Forest Classifier, and fine-tuning the model for optimal performance.

Features

1.Real-time Detection: Monitors transactions and detects fraudulent activities as they occur.
2.Machine Learning Algorithms: Utilizes a Random Forest Classifier to identify patterns indicative of scams.
3.Synthetic Data Generation: Generates realistic synthetic transaction data for testing and training purposes.
4.Data Balancing: Employs SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalances.
5.Model Tuning: Includes hyperparameter tuning with GridSearchCV for optimizing model performance.
6.Comprehensive Reports: Provides detailed classification reports and ROC-AUC analysis.
7.Feature Importance: Evaluates and displays the importance of different features in the model.



Synthetic Data Generation

The generate_synthetic_data function creates realistic synthetic transaction data, including attributes such as transaction type, amount, user details, device information, and geolocation. This data is crucial for training and testing the model.

Data Preprocessing

The preprocess_data function cleans and preprocesses the generated data. This includes handling missing values, encoding categorical variables, and normalizing numerical features.

Data Splitting and Balancing

The split_and_balance_data function splits the data into training and testing sets and balances the training data using SMOTE to handle class imbalances. This step ensures that the model is trained on a representative dataset.

Model Training and Evaluation

Model Training: The train_and_evaluate_model function trains a Random Forest Classifier on the balanced data. It also evaluates the initial model using metrics like accuracy, confusion matrix, and classification report.
Model Evaluation: The evaluate_tuned_model function further evaluates the tuned model using the ROC-AUC score and generates probabilities for ROC curve plotting.
Hyperparameter Tuning
The tune_model function uses GridSearchCV for hyperparameter tuning, optimizing the Random Forest model to find the best parameters for improved performance.

Saving the Model

The save_model function saves the trained model to a file named best_rf_classifier.pkl for future use.

Feature Importance

The feature_importances function evaluates and displays the importance of various features in determining fraudulent transactions. This helps in understanding which features are most influential in the model's predictions.

Plotting ROC Curve

The plot_roc_curve function plots the Receiver Operating Characteristic (ROC) curve to visualize the performance of the model in distinguishing between classes.
