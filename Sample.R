#####datathon#####

data<-read.csv('datathon_propattributes.csv',stringsAsFactors = FALSE)
str(data)
library(DataExplorer)
plot_str(data[1:10])
summary(data$IsTraining)

library(dplyr)
data2<-filter(data,IsTraining==1)
View(data2)

sum(is.na(data))
library(DataExplorer)
plot_missing(data2, title = NULL,
             theme_config = list(legend.position = c("bottom")))
plot_missing(data, title = NULL,
             theme_config = list(legend.position = c("bottom")))

##selecting variables




