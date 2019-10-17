Restricted Boltzmann Machine
================
Sijia Yue
10/16/2019

### Introduction

Restricted Boltzmann Machine is an undirected graphical model. RBMs can be used for data reduction (like PCA) and can also be adjusted for classification purposes.

RBMs consist of only two layers of nodes, a hidden layer with hidden nodes and a visible layer consisting of nodes that represent the data. In most applications the visible layer is represented by binary units. Here is a graphical representation of the RBM below:

``` r
knitr::include_graphics("RBM.png")
```

![](RBM.png)

### Load the library

``` r
library(devtools)
```

    ## Loading required package: usethis

``` r
library(RBM)
```

### Load MNIST data

The MNIST dataset is a hand-written numbers dataset and was downloaded from Kaggle. The library already has the dataset built in and normalized the data between 0 and 1.

``` r
data(MNIST)
train <- MNIST$trainX
test <- MNIST$testX
TrainY <- MNIST$trainY
TestY <- MNIST$testY
```

Try to plot a digit.

``` r
image(matrix(MNIST$trainX[2, ], nrow = 28), col = grey(seq(0, 1, length = 256)))
```

![](RBM_files/figure-markdown_github/unnamed-chunk-3-1.png)

### Using RBM() for classification problems

Use RBM() to fit the Restricted Boltzmann Machine model onto the MNIST dataset.

RBM() arguments

-   x: binary features

-   y: outcomes

-   n.iter: number of iterations

-   n.hidden: number of nodes in hidden layer

-   learning.rate: learning rate (*α*)

-   size.minibatch: size of minibatches

-   lambda: sparsity penalty lambda (*λ*) to prevent the system from overfitting

``` r
modelClassRBM <- RBM(x = train, y = TrainY, n.iter = 1000, n.hidden = 100, size.minibatch = 10, lambda = 0.1)
```

Then use `PredictRBM()` function for prediction.

This function would return two parameters:

-   ConfusionMatrix

-   Accuracy

``` r
predRBM <- PredictRBM(test = test, labels = TestY, model = modelClassRBM)
predRBM$ConfusionMatrix
```

    ##     truth
    ## pred   0   1   2   3   4   5   6   7   8   9
    ##    0 191   0   6   8   1  10   2   0   3   6
    ##    1   0 216   0   3   0   5   0   2   2   0
    ##    2   0   0 162   4   3   1   2   3   1   3
    ##    3   0   1   5 167   0  14   0   1   6   3
    ##    4   0   0   5   0 188   2   0   2   4   5
    ##    5   0   1   1   3   0 110   2   0   6   2
    ##    6   2   1   2   4   5   8 209   0   4   1
    ##    7   0   0   2   2   1   0   0 182   0   8
    ##    8   4   5   5   3   3   8   1   1 148   1
    ##    9   0   1   2   4  25   4   0  11   5 176

``` r
predRBM$Accuracy
```

    ## [1] 0.8745

Reference: <https://github.com/TimoMatzen/RBM#restricted-boltzmann-machine>
