# ECE414-Machine-Learning-Final-Project

https://www.kaggle.com/c/machinelearning2017

https://docs.google.com/document/d/1c03VymnWqVeg7t88u9tCs_beqM-N41rFE6UKy07DWls/edit?usp=sharing

## The Problem:
No prior knowledge of situation.  Given only training data with a known number of features we must design and build a classifier to correctly label a test dataset. 

## The Purpose:
Become familiar with out of the box classifiers, learn how to tune parameters, and practice good cross-validation skills.  

## Logistic Regression

We attempted classification with both L1 and L2 penalties. Using L1 required longer computational time and did not show any improvement over L2. This was also our first time using kf.split, the part of the sklearn package used for k-folds cross validation in order to get more accurate classification percentages. We increased max_iter in order to get better convergence, and tried various solver types, although newton-cg proved to be the best one for our dataset. We also imported a library that we used to print out the confusion matrix for our data, to allow for further analysis into the inaccuracies of the classifier. We tried normalizing our data in order to improve results - MinMaxScaler markedly improved results, and we continued to use it for future classifiers. Finally, we ran a 24 hour grid search over several parameters, including tolerance, C, and different solver types, but did not find any combination of parameters that improved the performance of our classifier.

## SGD Classifier

We again tried classification with both L1 and L2 penalties, with L2 being the clear choice again, although our results were still less satisfying than with Logistic Regression. We again increased max_iter to improve convergence. We also tried different loss parameters, but did not see marked improvement over any of them. Finally, we ran a shorter grid search only over tolerance and loss type that did not show marked improvement.

## Quadratic Discriminant Analysis

If it weren’t for the fact that attempting this method was required, we likely would not have used it. Because QDA can only assign quadratic decision boundaries, good classification results could obviously not be expected from noisy data in 128 space. Our best attempt at using this classifier resulted in approximately 15% correct classification.

## Gaussian Process Classifier

Similarly disappointing, Gaussian Process Classifier performed at sub-par levels as compared to logistic regression and SGD, while requiring significantly longer run time.

## Decision Tree Classifier

Because the data was so noisy, it is not unreasonable to attempt classifying based on a decision tree. It could be that this would have revealed otherwise indistinguishable patterns in the data, in which certain data trends were reliant on others. However, this proved to not be the case, and the Decision Tree Classifier performed at levels around 30%.

## Notes on Normalization

We tried several different normalization and dimensionality changing techniques on all of the above classifiers. Overall, MinMaxScaler proved the most reliable, as the noisiness of the data meant that maintaining relative distances between points was useful for classification. Normalizer and StandardScaler performed well for some of the above classifiers, but not all of them, and still performed only as well as MinMaxScaler. In terms of dimensionality reduction, SelectKBest and VarianceThreshold both did very little for improving classification results, which allows us to conclude that all 128 features are significant for classifying observations. Finally, increasing dimensionality with PolynomialFeatures did nothing for percentage correct while exponentially increasing training time. Even using PolynomialFeatures and then reducing dimensionality, results were not improved. Conclusion: data normalization was useful on the dataset, but only while maintaining relative distance between points. Changes of dimensionality were not at all useful.

## Conclusion

Our best performing results were on SGD and LogisticRegression, with 63% classification. Judging by the results on the Kaggle page, results between 60% and 70% are about as accurate as these out-of-box classifiers can get without resulting to processor-intensive neural nets.
