# Set the working directory to the location of the Excel file
setwd("C:\\Users\\artof\\git\\sigGCN")

# Load the necessary packages
library(MALDIquant)
library(MALDIquantForeign)
library(reticulate)
np <- import("numpy")
filepath <- "masses_and_intensities.npy"
np_array <- np$load(filepath, allow_pickle = TRUE)
masses <- as.numeric(np_array[, 1])
intensities <- as.numeric(np_array[, 2])

mass_spectrum <- createMassSpectrum(mass = masses, intensity = intensities)

spectra <- alignSpectra(c(mass_spectrum))
plot(np_array)
peaks <- detectPeaks(spectra)
print(peaks[[1]]@mass)

# Convert numpy array to a data.frame
df <- data.frame(np_array)

# Write the data.frame to an xlsx file
write.xlsx(df, "file.xlsx")

# top10 <- intensity(peaks[[1]]) %in% sort(intensity(peaks[[1]]), decreasing=TRUE)[1:10]
