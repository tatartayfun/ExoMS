# Set the working directory to the location of the Excel file
setwd("C:\\Users\\Hasan\\Desktop\\R")

## load package
library("MALDIquant")
## load example data
data("fiedler2009subset", package="MALDIquant")
## running typical workflow
## transform intensities
# spectra <- transformIntensity(fiedler2009subset, method="sqrt")
## smooth spectra
# spectra <- smoothIntensity(spectra, method="MovingAverage")
## baseline correction
# spectra <- removeBaseline(fiedler2009subset[[1]])

## align spectra
spectra <- alignSpectra(c(fiedler2009subset[[1]]))

# plot spectra
plot(spectra[[1]])

# detect peaks
peaks <- detectPeaks(spectra)

# highlight peaks on the plot
points(peaks[[1]])
