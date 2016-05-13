 # download exchange rate data
library(rnn)
library(sigmoid)
library(quantmod)

## todays values
start = '1998-12-14'
end   = '2005-06-01'
getFX("CHF/USD",from="1998-12-14", to = '2001-09-02')
getFX("GBP/USD",from="1998-12-14", to = '2001-09-02')
getFX("JPY/USD",from="1998-12-14", to = '2001-09-02')
getFX("EUR/USD",from="1998-12-14", to = '2001-09-02')

## summaries
summary(CHFUSD)

## simplify
CHFUSD <- as.vector(CHFUSD)
GBPUSD <- as.vector(GBPUSD)
JPYUSD <- as.vector(JPYUSD)
EURUSD <- as.vector(EURUSD)

## take first difference values for future returns
fCHFUSD <- diff(CHFUSD)
fGBPUSD <- diff(GBPUSD)
fJPYUSD <- diff(JPYUSD)
fEURUSD <- diff(EURUSD)

## current values
CHFUSD <- CHFUSD[-length(CHFUSD)]
GBPUSD <- GBPUSD[-length(GBPUSD)]
JPYUSD <- JPYUSD[-length(JPYUSD)]
EURUSD <- EURUSD[-length(EURUSD)]

## sigmoid
chfusd  <- logistic( CHFUSD, k=sd( CHFUSD)^-1, x0=mean( CHFUSD) )
gbpusd  <- logistic( GBPUSD, k=sd( GBPUSD)^-1, x0=mean( GBPUSD) )
jpyusd  <- logistic( JPYUSD, k=sd( JPYUSD)^-1, x0=mean( JPYUSD) )
eurusd  <- logistic( EURUSD, k=sd( EURUSD)^-1, x0=mean( EURUSD) )
fchfusd <- logistic(fCHFUSD, k=sd(fCHFUSD)^-1, x0=mean(fCHFUSD) )
fgbpusd <- logistic(fGBPUSD, k=sd(fGBPUSD)^-1, x0=mean(fGBPUSD) )
fjpyusd <- logistic(fJPYUSD, k=sd(fJPYUSD)^-1, x0=mean(fJPYUSD) )
feurusd <- logistic(fEURUSD, k=sd(fEURUSD)^-1, x0=mean(fEURUSD) )

## inspect data
summary( chfusd)
summary( gbpusd)
summary( jpyusd)
summary( eurusd)
summary(fchfusd)
summary(fgbpusd)
summary(fjpyusd)
summary(feurusd)

# put in matrix form
mchfusd <- matrix(chfusd, nrow = 1)
mgbpusd <- matrix(gbpusd, nrow = 1)
mjpyusd <- matrix(jpyusd, nrow = 1)
meurusd <- matrix(eurusd, nrow = 1)

# stack matrices
X <- array(c(meurusd, mchfusd, mgbpusd, mjpyusd), dim=c(1,993,4))

# response variable in matrix form
y  <- matrix(feurusd, ncol=993)

# binary response variable in matrix form
yB <- matrix(ifelse(feurusd < 0.5, 0, 1), ncol=993)

# train model
model <- trainr(X = X,
                Y = y,
                learningrate = 0.01,
                numepochs = 500,
                hidden_dim = c(10,10) )

# train binary model
modelB <- trainr(X = X,
                Y = yB,
                learningrate = 0.01,
                numepochs = 500,
                hidden_dim = c(10,10) )

# prediction
as.vector(predictr(model, X))

# plot prediction
hist(as.vector(predictr(model,X))-y)

# prediction Binary model
as.vector(round(predictr(modelB,X))-yB)
