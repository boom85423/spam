# Analyze SPAM data

## Motivation
* Annoying message interfere us to view the comment of youtube
* Make a SPAM classifier to filter the annoying comment
* Eventually, we will retrieve a pure social media

## Implement
* Preprocessing
    + Use encoding UTF-8 to read the data
    + Split the comment by space to feed the D2V
* Doc2Vec
    + Input: many different length comment
    + Output: fixed length vectors e.g. $V_{100\times1}$
    + The vector size is dominated by max of comment length
* DNN
    + Input: vector of comments
    + Output: spam or ham
    + Loss function: cross entropy of sigmoid
    + Optimizer: Adaptive Moment Estimation
    + Tuning the parameters e.g. batch size, learning rate

## Conclusion
* Performance
![](https://i.imgur.com/Y4RY2qT.png)
* Demo
    + python spamClassifier.py [--data] [--text]
    + --data=Youtube04-Eminem.csv
    + --text="Please leave your comment to classfy..."

