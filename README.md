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
![](https://i.imgur.com/Ls8nA82.png)

## Conclusion
* Demo  

    ```
    $ python spamClassifier.py [--data=Youtube03-LMFAO.csv] [--text="I am spam"]
    ```
    + --data: file you wanna train
    + --text: comment you wanna classify
* Performance

    ![](https://i.imgur.com/t88ermQ.png)

* Furthermore

    ![](https://i.imgur.com/itOAsfC.png)

    + Compare the ACC on identical model 
    + LMFAO is close to Eminem and Shakira

[Github](https://github.com/boom85423/spam)