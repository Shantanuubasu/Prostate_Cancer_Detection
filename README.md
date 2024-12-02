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

 ## INTRODUCTION
This project focuses on the application of machine learning techniques to improve diagnostic accuracy in
 medical imaging, specifically using data from three imaging modalities: K-trans, ADC, and T2W. Medical
 image classification plays a crucial role in the early detection and diagnosis of various diseases, and with the
 increasing availability of medical imaging data, there is a significant opportunity to leverage advanced
 computational methods for more accurate analysis. The dataset used in this project consists of NIfTI images,
 which were preprocessed to create 3D matrices, followed by padding to ensure uniformity and accuracy in
 the analysis. Various machine learning models, including CNN, Random Forest, SVM, Logistic Regression,
 and KNN, were tested to classify the images based on predefined categories. After initial evaluation, a
 hybrid CNN model was developed and fine-tuned to enhance performance. This project aims to explore the
 potential of deep learning in medical image analysis, emphasizing the importance of data preparation, model
 optimization, and hyperparameter tuning to achieve higher classification accuracy. The final results
 demonstrate that deep learning, particularly hybrid CNN architectures, can significantly improve the
 performance of medical image classification systems, offering promising implications for clinical decision
making and diagnostic workflows.
 The primary goal of this project is to demonstrate the effectiveness of machine learning models in
 automating the analysis of medical images, thereby aiding in faster and more accurate diagnoses. The
 project began with the collection and preparation of NIfTI images from three distinct imaging modalities: K
trans, ADC, and T2W. After preprocessing the data, which involved transforming the images into 3D
 matrices and applying padding for uniformity, various machine learning algorithms were tested for their
 ability to classify the images. The initial models, including traditional classifiers such as Random Forest,
 SVM, Logistic Regression, and KNN, provided insights into their strengths and limitations. However,
 recognizing the complex nature of medical imaging data, a hybrid CNN model was developed, incorporating
 advanced deep learning techniques and hyperparameter tuning to enhance performance. The results of this
 study show that, with proper preprocessing and model optimization, machine learning models—especially
 deep learning models like CNNs—can provide significant improvements in medical image classification,
 ultimately paving the way for more reliable and efficient diagnostic tools in healthcare.

## LITERATURE REVIEW
In this project, Python, alongside essential libraries like Pandas, NumPy, Matplotlib,
 Tensorflow, and Keras, serves as the foundation, empowering seamless data
 manipulation, analysis, visualization, and CNN model implementation for robust
 insights extraction.

 ### Python Libraries
 1. Pandas: A powerful data manipulation and analysis library for Python, providing
 data structures and functions to work with structured data efficiently.
 2. NumPy: A fundamental package for numerical computing in Python, offering
 support for large, multi-dimensional arrays and matrices, along with a collection of
 mathematical functions.
 3. Matplotlib: A comprehensive plotting library in Python, enabling the creation of
 a wide variety of static, animated, and interactive visualizations.
 4. Keras:· Keras is a high-level API for building and training deep learning models.
 Initially developed as a separate library, it is now integrated into TensorFlow,
 providing a user-friendly interface for constructing neural networks.
 5. Datetime:The datetime module is part of Python's standard library, and provides
 classes for working with dates and times. It allows for manipulation, formatting,
 and parsing of date and time information.
 6. Nibabel:Nibabel is a Python library designed to read, write, and manipulate
 medical imaging data, particularly neuroimaging data. It supports various file
 formats such as NIfTI, Analyze, and others commonly used in neuroimaging and
 brain research.
 7. Tensorflow:TensorFlow is an open-source machine learning and deep learning
 framework developed by Google. It is widely used for building and deploying
 machine learning models, including neural networks for tasks such as image
 recognition, natural language processing, and more.

## Problem Statement / Requirement Specifications
 We developed a hybrid CNN model with hyperparameter tuning, employing ReLU
 and Softmax activation functions. This enhanced model significantly improved
 performance, achieving a final accuracy of 78.57%.

 ### Project Planning
 1. Define Objectives: Clearly outline the project's goal, such as
 developing a convolution neural network model to classify patients with
 Prostate Cancer.
 
 2. Data Collection: A dataset of MRI scans of prostate glands,was
 collected from a reliable medical institution
 
 3. Data Preprocessing: Cleanse and preprocess the data to handle
 missing values, outliers, and normalize features for better model
 performance.
 
 4. Model Selection: Choose an appropriate support vector machine
 algorithm and kernel type based on the nature of the data and desired
 outcomes.
 
 5. Hyperparameter Tuning: Optimize the model's hyperparameters
 using techniques like grid search or random search to enhance its
 predictive capabilities.
 
 6. Training: Train the support vector machine model on a portion of
 the dataset, ensuring sufficient data for both training and validation.
 
 7. Evaluation: Assess the model's performance using appropriate
 evaluation metrics such as accuracy, precision, recall, and F1-score.
 
 8. Validation: Validate the model's performance on an independent
 dataset to confirm its generalization ability and robustness.
 
 9. Deployment: Deploy the trained model into a production
 environment, ensuring seamless integration with end-user systems or
 applications.

10. Monitoring and Maintenance: Continuously monitor the model's
 performance in real-world settings and update it periodically to adapt to
 changing data patterns or requirements.

### System Design

#### Design Constraints

##### Hardware requirements:

Memory: 8 GB RAM
Free Storage: 4 GB (SSD Recommended)
Screen Resolution: 1920 x 800
OS: Windows 10 (64-bit)
CPU: Intel Core i5-8400 3.0 GHz or better

##### System Architecture OR Block Diagram:
<img width="200" alt="image" src="https://github.com/Shantanuubasu/Prostate_Cancer_Detection/blob/main/model_architecture.png">

## Implementation
 The implementation of this project using Convolutional Neural Networks (CNNs)
 involved several key steps to effectively classify medical imaging data. Initially, the
 NIfTI images from the three modalities (K-trans, ADC, and T2W) were
 preprocessed to generate 3D matrices, followed by padding to ensure consistent
 image dimensions and preserve important spatial features. The preprocessed images
 were then divided into training and testing sets. A CNN architecture was designed
 to automatically learn spatial hierarchies from these images, which is particularly
 well-suited for medical imaging tasks. The network consisted of multiple
 convolutional layers for feature extraction, followed by pooling layers to reduce
 dimensionality and retain essential features. The final layers included fully
 connected layers with a Softmax activation function for multi-class classification.
 Despite initial accuracy results of around 69%, the model was further optimized
 using hyperparameter tuning, adjusting parameters like learning rate, batch size, and
 number of epochs. The final tuned CNN model significantly improved classification
 performance, achieving an accuracy of 78.57%, demonstrating the power of CNNs
 in automating feature extraction and classification in medical image analysis.

 ### Methodology OR Proposal
 In this project, we focused on improving diagnostic accuracy through the analysis
 and classification of medical imaging data using NIfTI images with three modalities:
 K-trans, ADC, and T2W. The images were preprocessed to create 3D matrices, with
 padding applied to ensure accurate identification of target areas. The dataset was
 split into 70% training and 30% testing sets, and a variety of machine learning
 models were tested, including CNN, Random Forest, SVM, Logistic Regression,
 and KNN, yielding accuracies ranging from 69.38% to 74.37%. To enhance
 performance, we developed a hybrid CNN model, incorporating hyperparameter
 tuning and utilizing ReLU and Softmax activation functions. This approach led to a
 significant improvement in performance, achieving a final accuracy of 78.57%. The
 results highlight the effectiveness of hybrid CNN architectures and emphasize the
 crucial role of preprocessing, hyperparameter tuning, and model design in
 enhancing medical image classification tasks.

### Testing OR Verification Plan:
<img width="296" alt="image" src="https://github.com/Shantanuubasu/Prostate_Cancer_Detection/blob/main/Test%20Split.png">

### Result Analysis OR Screenshots:
<img width="262" alt="image" src="https://github.com/user-attachments/assets/f0f6934b-47fd-4f10-8d7f-49cad5951b16">

### Confusion Matrix
![image](https://github.com/user-attachments/assets/054348a1-03c8-4d8c-ab06-b222c42aad08)

##  Performance Measures of Existing Model Vs Our Model:

###  SVM Performance Metrics of Existing Model:

![image](https://github.com/user-attachments/assets/305176cd-7e89-4e5a-acbe-12825cc18d59)

###  LRPerformance Metrics of Existing Model:

![image](https://github.com/user-attachments/assets/9813d6a0-9ec8-45ee-bde4-103e0718b53a)

### K-NNPerformance Metrics of Existing Model:

![image](https://github.com/user-attachments/assets/f4cade83-fbf3-462f-aeea-93db92a83693)

###  Our Model:

![image](https://github.com/user-attachments/assets/30664e05-a423-40a0-9518-77efed0809b6)

## Standards Adopted

### Design Standards :

1.User-Centric Approach: Prioritize user needs and preferences to create intuitive and user-friendly interfaces or experiences.

2.Modularity: Design the project in a modular fashion, breaking it into smaller, manageable components or modules to facilitate easier development, testing, and maintenance.

3.Scalability: Ensure the project's architecture and design can accommodate future growth and expansion without significant restructuring or performance degradation.

4.Consistency: Maintain consistency in design elements such as layout, color scheme, typography, and terminology across the project to provide a cohesive user experience.

5.Accessibility: Design with accessibility in mind to ensure all users, including those with disabilities, can access and use the project effectively.

6.Performance: Optimize the project for performance, considering factors such as loading times, response times, and resource utilization to deliver a responsive and efficient experience.

7.Security: Implement robust security measures to protect sensitive data, prevent unauthorized access, and mitigate potential security threats or vulnerabilities.

8.Documentation: Document the project's design decisions, architecture, components, APIs, and usage guidelines comprehensively to aid in understanding, maintenance, and future development.

9.Testing: Incorporate testing methodologies and practices throughout the design process to identify and rectify issues early, ensuring the project meets quality standards and user expectations.

10.Feedback Mechanism: Establish mechanisms for gathering feedback from stakeholders, users, and team members throughout the design and development lifecycle to iterate and improve upon the project continuously.

###  Coding Standards

1.Naming Conventions: Use descriptive and meaningful names for variables, functions, classes, and other identifiers. Follow a consistent naming convention, such as camelCase or snake_case.

2.Indentation and Formatting: Use consistent indentation (spaces or tabs) and formatting (e.g., braces placement, line length) to enhance code readability and maintainability.

3.Comments and Documentation: Include comments to explain the purpose of code blocks, complex algorithms, and non-obvious logic. Document functions, classes, and modules using docstrings to provide usage instructions and clarify behavior.

4.Code Organization: Organize code into logical modules, packages, and directories. Follow a modular and hierarchical structure to facilitate code reuse, scalability, and maintainability.

5.Error Handling: Implement robust error handling mechanisms to gracefully handle exceptions and errors, providing informative error messages and logging for debugging purposes.

6.Code Reusability: Write reusable code by breaking functionality into small, cohesive functions and classes. Avoid duplication of code and favor composition over inheritance.

7.Testing Standards: Write unit tests to verify the correctness of individual components and integration tests to validate the interactions between components. Follow test-driven development (TDD) or behavior-driven development (BDD) practices to ensure code quality and reliability.

Performance Optimization: Optimize code performance by minimizing computational complexity, avoiding unnecessary resource allocation, and utilizing efficient algorithms and data structures.

###  Testing Standards

1.Test Planning: Develop a comprehensive test plan outlining the testing approach, objectives, scope, resources, and timelines. Identify test scenarios, test cases, and testing environments based on project requirements and priorities.

2.Test Case Design: Design test cases covering functional requirements, edge cases, boundary conditions, error handling scenarios, and user interactions. Ensure test cases are clear, concise, and traceable to requirements.

3.Test Automation: Automate repetitive and time-consuming test cases using test automation frameworks and tools. Prioritize automation for regression testing, smoke testing, and critical path scenarios to increase efficiency and coverage.

4.Test Execution: Execute test cases systematically, recording test results, observations, and defects in a test management system. Perform both manual and automated testing across various platforms, browsers, and devices as needed.

5.Regression Testing: Conduct regression testing to verify that recent code changes do not adversely affect existing functionality. Prioritize regression test suites based on criticality and frequency of code changes.

6.Performance Testing: Evaluate system performance, scalability, and responsiveness under different load levels using performance testing tools. Identify and address performance bottlenecks, memory leaks, and resource utilization issues.

7.Security Testing: Perform security testing to identify vulnerabilities, weaknesses, and threats in the software application. Conduct penetration testing, vulnerability scanning, and code analysis to enhance security posture.

8.Usability Testing: Validate the user interface design, navigation flow, and overall user experience through usability testing. Gather feedback from end-users to identify usability issues and areas for improvement.

9.Compatibility Testing: Ensure compatibility across different platforms, operating systems, browsers, and devices. Test for cross-browser compatibility, screen resolutions, accessibility, and localization requirements.

10.Integration Testing: Validate the interactions and interfaces between software modules, components, and third-party systems through integration testing. Verify data exchange, communication protocols, and error handling between integrated components.

11. Acceptance Testing: Conduct acceptance testing with stakeholders to validate that the software meets specified requirements and business goals. Obtain sign-off from stakeholders before deploying the software to production.


## Conclusion and Future Scope

### Conclusion

In conclusion, this project has demonstrated the potential of deep learning in
 accurately detecting prostate cancer from MRI images. The developed 3D
 CNN model achieved significant accuracy in classifying benign and
 malignant cases. While promising results have been obtained, further
 research is needed to refine the model and address potential limitations. By
 incorporating larger datasets, exploring advanced architectures, and
 integrating multi-modal information, we can strive to improve the model's
 performance and clinical impact. Ultimately, this research aims to contribute
 to the development of more accurate and efficient diagnostic tools for
 prostate cancer.

 ###	Future Scope:

 1. Incorporating Additional Data Modalities: Integrating information from
 other imaging modalities, such as PET scans or ultrasound, can provide
 complementary insights and improve diagnostic accuracy.
 
 2. Enhancing Model Interpretability: Developing techniques to visualize and
 understand the decision-making process of the deep learning model can
 increase trust and facilitate clinical adoption.
 
 3. Real-time Analysis: Optimizing the model for real-time processing of MRI
 images can enable immediate clinical decision-making.

 4. Personalized Medicine: Leveraging patient-specific information, such as
 genetic factors and clinical history, to tailor the model's predictions to
 individual patients.

 6. Continuous Learning: Implementing mechanisms for the model to learn from
 new data and adapt to changing clinical practices.

## REFERENCES

 1. https://ieeexplore.ieee.org/document/9349466
 2. https://www.cdc.gov/prostate/cancer/diagnosis/index.html#:~:text=A%20biopsy%20is%20a%20procedure,looked%20at%20under%20the%20microscope

### NOTE: This project was created for academic usage by:

<a href="https://github.com/Shantanuubasu">Shantanuubasu</a>
<a href="https://github.com/ankit221209">Ankit Ghosh</a>
<a href="https://github.com/ChaitanyaPunja">Chaitanya Punja</a>
<a href="https://github.com/Ank-Prak">Ankit Prakash</a>
<a href="https://github.com/Sayanjones">Sayan Mandal</a>


