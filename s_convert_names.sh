#!/bin/sh

changeName()
{
  grep -rl "$1" ./pints/* | xargs sed -ie "s/$1/$2/g"
}


changeName UnknownNoiseLogLikelihood GaussianLogLikelihood
changeName KnownNoiseLogLikelihood GaussianKnownSigmaLogLikelihood
changeName MultimodalNormalLogPDF MultimodalGaussianLogPDF
changeName NormalLogPDF GaussianLogPDF
changeName HighDimensionalNormalLogPDF HighDimensionalGaussianLogPDF
changeName MultivariateNormalLogPrior MultivariateGaussianLogPrior
changeName NormalLogPrior GaussianLogPrior
changeName Normal Gaussian

rm ./App-1/*.Re