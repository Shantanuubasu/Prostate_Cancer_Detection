# Prostate_Cancer_Detection

## ABSTRACT
 This project focuses on the analysis and classification of medical imaging data to enhance
 diagnostic accuracy. We utilized a dataset comprising NIfTI images with three modalities: K
trans, ADC, and T2W. The images were preprocessed to generate 3D matrices, followed by
 padding to accurately identify target areas. After splitting the data into a 70:30 ratio for training
 and testing, we experimented with several machine learning models including CNN, Random
 Forest, SVM, Logistic Regression, and KNN. The respective accuracies were 69.38%, 70.41%,
 74.37%, 70.41%, and 70.42%.
 Given these results were suboptimal, we developed a hybrid CNN model with hyperparameter
 tuning, employing ReLU and Softmax activation functions. This enhanced model significantly
 improved performance, achieving a final accuracy of 78.57%. The results demonstrate the
 potential of hybrid CNN architectures in improving medical image classification and underscore
 the importance of preprocessing, hyperparameter tuning, and advanced model design in such
 applications.

