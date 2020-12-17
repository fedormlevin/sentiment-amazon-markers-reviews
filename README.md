# sentiment-amazon-markers-reviews
Sentiment Analysis of Reviews from Amazon (acrylic markers/pens)
# Summary

I am using logistic regression on imbalanced 2 label dataset of Amazon reviews to predict the sentiment of the review (e.g. Positive/Negative) <br>
I add features, tune the model and compare the results at the end

## Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
import nltk, re, pprint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score
```
## Dataset
Removing neutral ratings and select only nessesary columns from the reviews dataset:
```python
df = resul[resul['Rating'] != 3]
df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)
```
Our dataset is inbalanced. Majority label "1" accounts for around 93% of the population
```python
df['Positively Rated'].value_counts()/df.shape[0]
```
Label | Ratio
---|---------
1 | 0.925511
0 | 0.074489
## Default Model
Splitting data into training and test sets
```python
X_train, X_test, y_train, y_test = train_test_split(df['Body'], 
                                                    df['Positively Rated'], 
                                                    random_state=2)
```
Default Logistic Regression model: <br>
Transforming the documents in the training data to a document-term matrix:
```python
X_train_vectorized = vect.transform(X_train)
```
Training the model:
```python
model_simple = LogisticRegression()
model_simple.fit(X_train_vectorized, y_train)
```
Predicting the transformed test documents and printing the results:
```python
predictions_simple = model_simple.predict(vect.transform(X_test))
print(f'Accuracy Score: {accuracy_score(y_test,predictions_simple)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, predictions_simple)}')
print(f'Area Under Curve: {roc_auc_score(y_test, predictions_simple)}')
print(f'Recall score: {recall_score(y_test,predictions_simple)}')
print(f'Precision score: {precision_score(y_test,predictions_simple)}')
```
Accuracy Score: 0.9503404084901882<br>
Confusion Matrix:<br>
[[ 106 -  91]<br>
 [  33 - 2267]]<br>
Area Under Curve: 0.7618616199514456<br>
Recall score: 0.9856521739130435<br>
Precision score: 0.9614079728583546<br>
Although the Accuracy Score is high, almost half of the negative reviews (0) estimated incorrectly (False Positives = 91)<br>
What are the features with smallest (predicting negatives reviews) and largest (predicting positive) coefficients?
```python
# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model_simple.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
```
Smallest Coefs:<br>
['return' 'half' 'not' 'never' 'terrible' 'splatter' 'disappointed'
 'cheap' 'say' 'dried']<br>
Largest Coefs:<br>
['easy' 'love' 'perfect' 'great' 'excellent' 'best' 'fun' 'perfectly'
 'nice' 'beautiful']<br>
 Let's try to test the model using generic input "markers are good" and "markers are not good"
```python
print(model_simple.predict(vect.transform(['markers are good',
                                    'markers are not good'])))
```
Output: [1 1] - both strings were predicted as Positive

## Adding n-grams and Tfidf:
Adding n-grams to feature to compare word pairs (e.g. "good" and "not good"). Also, getting rid of most frequent words that do not add to the quality of prediction (e.g. "you")
```python
# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
vect_count = CountVectorizer(min_df=7, ngram_range=(1,2), 
                             stop_words = frozenset(["you", "your", "zu"])).fit(X_train)
vect_tfidf = TfidfVectorizer(min_df=7, ngram_range=(1,2), 
                             stop_words = frozenset(["you", "your", "zu"])).fit(X_train)

X_train_vectorized = vect_count.transform(X_train)
X_train_vectorized_tfidf = vect_tfidf.transform(X_train)
```
## CountVectorizer:
```python
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
X_test_vectorized = vect_count.transform(X_test)

predictions = model.predict(vect_count.transform(X_test))

#print('AUC: ', roc_auc_score(y_test, predictions))

# performance
print(f'Accuracy Score: {accuracy_score(y_test,predictions)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, predictions)}')
print(f'Area Under Curve: {roc_auc_score(y_test, predictions)}')
print(f'Recall score: {recall_score(y_test,predictions)}')
print(f'Precision score: {precision_score(y_test,predictions)}')
```
Accuracy Score: 0.9583500200240288<br>
Confusion Matrix:<br>
[[ 116 -  81]<br>
 [  23 - 2277]]<br>
Area Under Curve: 0.7894162436548222<br>
Recall score: 0.99<br>
Precision score: 0.9656488549618321<br>
## Tfidf Vectorizer:
```python
model = LogisticRegression()
model.fit(X_train_vectorized_tfidf, y_train)
X_test_vectorized_tfidf = vect_tfidf.transform(X_test)

predictions_tfidf = model.predict(vect_tfidf.transform(X_test))

#print('AUC: ', roc_auc_score(y_test, predictions))

# performance
print(f'Accuracy Score: {accuracy_score(y_test,predictions_tfidf)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, predictions_tfidf)}')
print(f'Area Under Curve: {roc_auc_score(y_test, predictions_tfidf)}')
print(f'Recall score: {recall_score(y_test,predictions_tfidf)}')
print(f'Precision score: {precision_score(y_test,predictions_tfidf)}')
```
Accuracy Score: 0.9427312775330396<br>
Confusion Matrix:<br>
[[  59 - 138]<br>
 [   5 2295]]<br>
Area Under Curve: 0.6486592363716619<br>
Recall score: 0.9978260869565218<br>
Precision score: 0.9432799013563502<br>

Altough we have added n-gram features to out models they still don't work well on generic reviews "markers are good" and "markers are not good". Both models have no very strong AUC values. We will add weights to LR model in order to increase the impact of minority class (Negative reviews "0")

## Tfidf parametered LR with class_weight
```python
# define class weights
model_w = LogisticRegression(random_state=13,C=680,fit_intercept=True, 
                             penalty='l2',class_weight={0: 99.991, 1: 0.009} )
model_w.fit(X_train_vectorized_tfidf, y_train)
X_test_vectorized_tfidf = vect_tfidf.transform(X_test)


predictions_w_idf = model_w.predict(vect_tfidf.transform(X_test))

#print('AUC: ', roc_auc_score(y_test, predictions))

# performance
print(f'Accuracy Score: {accuracy_score(y_test,predictions_w_idf)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, predictions_w_idf)}')
print(f'Area Under Curve: {roc_auc_score(y_test, predictions_w_idf)}')
print(f'Recall score: {recall_score(y_test,predictions_w_idf)}')
print(f'Precision score: {precision_score(y_test,predictions_w_idf)}')
```
Accuracy Score: 0.8466159391269523<br>
Confusion Matrix:<br>
[[ 189  -  8]<br>
 [ 375 1925]]<br>
Area Under Curve: 0.8981736923416463<br>
Recall score: 0.8369565217391305<br>
Precision score: 0.9958613554061045<br>
The above model has strong AUC. However, still, doesn't correctly labels generic reveiw reviews "markers are good" and "markers are not good": it assigns Negative label to both reviews caused by our set hight sensitivity to negative class
```python
print(model_w.predict(vect_tfidf.transform(['markers are good',
                                    'markers are not good'])))
```
## CountVectorizer with added regularization and weights
Trying to use CountVectorized parameterd LR with weights, which doesn't take into account same word distributions accross documents comparing to Tfidf
```python
model_w = LogisticRegression(random_state=13,C=80,fit_intercept=True, 
                             penalty='l2',class_weight={0: 99.991, 1: 0.009} )
model_w.fit(X_train_vectorized, y_train)
X_test_vectorized = vect_count.transform(X_test)


predictions_w = model_w.predict(vect_count.transform(X_test))

#print('AUC: ', roc_auc_score(y_test, predictions))

# performance
print(f'Accuracy Score: {accuracy_score(y_test,predictions_w)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, predictions_w)}')
print(f'Area Under Curve: {roc_auc_score(y_test, predictions_w)}')
print(f'Recall score: {recall_score(y_test,predictions_w)}')
print(f'Precision score: {precision_score(y_test,predictions_w)}')
```
Accuracy Score: 0.8446135362434922<br>
Confusion Matrix:<br>
[[ 174 -  23]<br>
 [ 365 - 1935]]<br>
Area Under Curve: 0.8622765393952769<br>
Recall score: 0.841304347826087<br>
Precision score: 0.9882533197139939<br>
Although this model has lightly low AUC score (0.86 comparing to 0.90 in Tfidf), it does well in generic reviews "markers are good" and "markers are not good"
```python
print(model_w.predict(vect_count.transform(['markers are good',
                                    'markers are not good'])))
```
Output: [1 - 0]
# Conclusion
For handling imbalance dataset weights were added to LR model and further tuning (adding regulazation parameter and CountVectorizer/Tfidf) was needed to get the best results in AUC as well as tests on generic strings. <br>
Comparing the results:

Statistics/Model | Default Model | CountVectorizer with n-grams | Tfidf with n-grams | CountVectorizer with regularization and weights | Tfidf with regularization and weights
-----------------|------------------|--------------------|----------------|-------|-----
Accuracy Score | 0.9503404084901882 | 0.9583500200240288 | 0.9427312775330396 | 0.8446135362434922 | 0.8466159391269523
Area Under Curve | 0.7618616199514456 | 0.7894162436548222| 0.6486592363716619 | 0.8622765393952769 | 0.8981736923416463
Recall score | 0.9856521739130435 | 0.99 | 0.9978260869565218 | 0.841304347826087 | 0.8369565217391305
Precision score | 0.9614079728583546 | 0.9656488549618321 | 0.9432799013563502 | 0.9882533197139939 | 0.9958613554061045
Generic test<br>("markers are good";<br>"markers are not good") | Both predicted as Positive | Both as Predicted positive | Both predicted as Positive | Predicted correctly - Positive, Negative| Both predicted as Negative
