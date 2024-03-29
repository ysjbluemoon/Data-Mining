# Reference
# https://github.com/TimoMatzen/RBM

# Install devtools
install.packages("devtools")
# Load devtools
library(devtools)
# Install RBM
install_github("TimoMatzen/RBM")
# load RBM
library(RBM)

# Load MNIST data
data(MNIST)
train <- MNIST$trainX
test <- MNIST$testX
TrainY <- MNIST$trainY
TestY <- MNIST$testY
image(matrix(MNIST$trainX[2, ], nrow = 28), col = grey(seq(0, 1, length = 256)))

# Using RBM()
modelRBM <- RBM(x = train, n.iter = 1000, n.hidden = 100, size.minibatch = 10)
# Reconstruct the image with modelRBM
ReconstructRBM(test = test[6, ], model = modelRBM)


# Use RBM() in classification problems

# This time we add the labels as the y argument
modelClassRBM <- RBM(x = train, y = TrainY, n.iter = 1000, n.hidden = 100, size.minibatch = 10)
predRBM <- PredictRBM(test = test, labels = TestY, model = modelClassRBM)
predRBM$ConfusionMatrix
predRBM$Accuracy
