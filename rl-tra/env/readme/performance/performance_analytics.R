# https://www.rdocumentation.org/packages/PerformanceAnalytics/versions/2.0.4/topics/portfolio_bacon
# https://www.datacamp.com/datalab/w/28c21593-21e6-47d9-8e72-acebdd3be32c/edit
# https://www.datacamp.com/datalab/w/28c21593-21e6-47d9-8e72-acebdd3be32c/edit#9c513e64-2a46-4c6b-9b59-9567e18e8229
# https://www.datacamp.com/datalab/w/28c21593-21e6-47d9-8e72-acebdd3be32c/edit

if(!require('PerformanceAnalytics')) {
    install.packages('PerformanceAnalytics')
    library('PerformanceAnalytics')
}

data(portfolio_bacon)
head(portfolio_bacon, 100)
write.csv(portfolio_bacon)

portfolio_length <- nrow(portfolio_bacon)
print(portfolio_length)

for(rf in seq(0, 0.3, by=0.05)) {
  print(paste("Risk-free rate: ", rf))
  for (i in 1:portfolio_length) {
    result <- AdjustedSharpeRatio(portfolio_bacon[1:i, ], Rf=rf)
    #print(i)
    #print(result)
	write.csv(result)
  }
}

for (fun in c("StdDev", "VaR", "ES", "SemiSD")) {
  for(rf in seq(0, 0.3, by=0.05)) {
    for (i in 1:portfolio_length) {
      result <- SharpeRatio(portfolio_bacon[1:i, ], Rf=rf, FUN=fun)
	    write.csv(result)
    }
  }
}

for(rf in seq(0, 0.3, by=0.05)) {
  #print(paste("Risk-free rate: ", rf))
  for (i in 1:portfolio_length) {
    # could not find function "DownsideSharpeRatio"
    result <- DownsideSharpeRatio(portfolio_bacon[1:i, ], Rf=rf)
    #print(i)
    #print(result)
	  write.csv(result)
  }
}

for (i in 1:portfolio_length) {
    result <- BernardoLedoitRatio(portfolio_bacon[1:i, ])
    #print(i)
    #print(result)
    #write.csv(result, paste0("result_rf_", rf, "_i_", i, ".csv"))
	write.csv(result)
}

for (modified in c(TRUE, FALSE)) {
  for(rf in seq(0, 0.3, by=0.05)) {
    for (i in 2:portfolio_length) {
      result <- BurkeRatio(portfolio_bacon[1:i, ], Rf=rf, modified=modified)
	    write.csv(result)
    }
  }
}

for(rf in seq(0, 0.3, by=0.05)) {
  for (i in portfolio_length:portfolio_length) {
    result <- Return.excess(portfolio_bacon[1:i,], Rf=rf)
	  write.csv(result)
  }
}

for(geometric in c(TRUE, FALSE)) {
  print(geometric)
  for (i in 1:portfolio_length) {
    result <- Return.cumulative(portfolio_bacon[1:i,], geometric=geometric)
	  write.csv(result)
  }
}

for(mar in seq(0, 0.3, by=0.05)) {
  for (i in 1:portfolio_length) {
    result <- SortinoRatio(portfolio_bacon[1:i,], MAR=mar)
	write.csv(result)
  }
}

for (l in seq(0.0, 0.1, by=0.02)) {
  for(rf in c(0)) { # results doesn't change when rf is changed
	print(paste("L: ", l, "Risk-free rate: ", rf))
    for (i in 1:portfolio_length) {
      result <- Omega(portfolio_bacon[1:i, ], L=l, Rf=rf, method="simple", output="point")
	    write.csv(result)
    }
  }
}

for (l in c(1,2,3,4)) {
  for(mar in seq(0, 0.3, by=0.05)) {
	print(paste("L: ", l, "MAR: ", mar))
    for (i in 1:portfolio_length) {
      result <- Kappa(portfolio_bacon[1:i, ], MAR=mar, l=l)
	    write.csv(result)
    }
  }
}

for (i in 1:portfolio_length) {
  result <- InformationRatio(portfolio_bacon[1:i,1], Rb=portfolio_bacon[1:i,2])
	write.csv(result)
}
for (i in 1:portfolio_length) {
  result <- InformationRatio(portfolio_bacon[1:i,2], Rb=portfolio_bacon[1:i,1])
	write.csv(result)
}

for (method in c("excess", "moment", "fisher", "sample", "sample_excess")) {
	print(paste("method: ", method))
  for (i in 1:portfolio_length) {
    result <- kurtosis(portfolio_bacon[1:i, ], method=method)
	  write.csv(result)
  }
}

for (method in c("full","subset")) {
  for(mar in seq(0, 0.1, by=0.02)) {
	  print(paste("method: ", method, "MAR: ", mar))
    for (i in 1:portfolio_length) {
      result <- UpsidePotentialRatio(portfolio_bacon[1:i, ], MAR=mar, method=method)
	    write.csv(result)
    }
  }
}

result <- Drawdowns(portfolio_bacon[,], geometric = TRUE)
write.csv(result)
result <- maxDrawdown(portfolio_bacon[,], geometric = TRUE)
write.csv(result)

for (i in 1:portfolio_length) {
    result <- CalmarRatio(portfolio_bacon[1:i, ], scale=1)
	  write.csv(result)
}

for (excess in seq(0.0, 0.1, by=0.02)) {
  for (i in 1:portfolio_length) {
    result <- SterlingRatio(portfolio_bacon[1:i, ], excess=excess, scale=1)
	  write.csv(result)
  }
}

################################################################
# make daily returns from monthly ones
################################################################
dates <- as.Date(index(portfolio_bacon))
#print(paste("dates",dates))

# Generate a sequence of daily dates starting from the first date in the dataset
start_date <- dates[1]
end_date <- dates[length(dates)]
#print(paste("start=",start_date,"end=", end_date))
daily_dates <- seq.Date(from = start_date, to = end_date, by = "day")

# Ensure the number of daily dates matches the number of rows in the dataset
if (length(daily_dates) >= nrow(portfolio_bacon)) {
  daily_dates <- daily_dates[1:nrow(portfolio_bacon)]
} else {
  stop("The generated sequence of daily dates is shorter than the number of rows in the dataset.")
}

index(portfolio_bacon) <- daily_dates
head(portfolio_bacon, 100)
###################################################################

################################################################
# make yearly return dates from the monthly ones,
# because R code annualizes the returns
################################################################
dates <- as.Date(index(portfolio_bacon))
#print(paste("dates",dates))

# Generate a sequence of yearly dates starting from the first date in the dataset
start_date <- dates[1]
end_date <- dates[length(dates)]
#print(paste("start=",start_date,"end=", end_date))
yearly_dates <- seq.Date(from = start_date, by = "year", length.out = nrow(portfolio_bacon))

# Ensure the number of yearly dates matches the number of rows in the dataset
if (length(yearly_dates) > nrow(portfolio_bacon)) {
  yearly_dates <- yearly_dates[1:nrow(portfolio_bacon)]
} else if (length(yearly_dates) < nrow(portfolio_bacon)) {
  stop("Not enough yearly dates to match the number of rows in the dataset.")
}

index(portfolio_bacon) <- yearly_dates
portfolio_length <- nrow(portfolio_bacon)
#head(portfolio_bacon, 100)
###################################################################

for (modified in c(FALSE,TRUE)) {
  for (rf in seq(0.0, 0.1, by=0.02)) {
    print(paste("modified: ", modified, "Rf: ", rf))
	# first value (i=1) is always None
    for (i in 2:portfolio_length) {
      result <- BurkeRatio(portfolio_bacon[1:i, ], Rf=rf, modified=modified)
	    write.csv(result)
    }
  }
}

for (rf in seq(0.0, 0.1, by=0.02)) {
  print(paste("Rf: ", rf))
  # first value (i=1) is always None because R code calculates periodicity
  for (i in 2:portfolio_length) {
    result <- PainRatio(portfolio_bacon[1:i, ], Rf=rf)
    write.csv(result)
  }
}

for (rf in seq(0.0, 0.1, by=0.02)) {
  print(paste("Rf: ", rf))
  # first value (i=1) is always None because R code calculates periodicity
  for (i in 2:portfolio_length) {
    result <- MartinRatio(portfolio_bacon[1:i, ], Rf=rf)
    write.csv(result)
  }
}

for (i in 1:portfolio_length) {
  result <- PainIndex(portfolio_bacon[1:i, ])
  write.csv(result)
}

for (i in 1:portfolio_length) {
  result <- UlcerIndex(portfolio_bacon[1:i, ])
  write.csv(result)
}

# https://www.datacamp.com/datalab/w/28c21593-21e6-47d9-8e72-acebdd3be32c/edit

