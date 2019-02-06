#!/bin/sh

# changes name instances within PINTS files
changeName()
{
  grep -rl "$1" ./pints/* ./examples/* ./docs/* | xargs perl -i -pe"s/$1/$2/g"
}

# attempt to change file names
# changeFilename()
#   find . | grep "$1" | xargs sed -i "s/$1/$2/g"
# }

# uses homebrew package rename to change file names: brew install rename
rename -vs normal gaussian ./pints/tests/* ./pints/* ./pints/toy/*

changeName UnknownNoiseLogLikelihood GaussianLogLikelihood
changeName KnownNoiseLogLikelihood GaussianKnownSigmaLogLikelihood
changeName MultimodalNormalLogPDF MultimodalGaussianLogPDF
changeName NormalLogPDF GaussianLogPDF
changeName HighDimensionalNormalLogPDF HighDimensionalGaussianLogPDF
changeName MultivariateNormalLogPrior MultivariateGaussianLogPrior
changeName NormalLogPrior GaussianLogPrior
changeName Normal Gaussian
changeName multimodal_normal multimodal_gaussian
changeName high_dimensional_normal high_dimensional_gaussian
changeName normal_logpdf gaussian_logpdf
changeName '_normal ' '_gaussian '

