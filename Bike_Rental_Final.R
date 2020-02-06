# removing previously loaded objects
rm(list =ls())

## Setting Working directory
setwd("D:/#/Projects/Projects/Bike Prediction/R")
getwd()

bike_data = read.csv("day.csv")

names(bike_data)
head(bike_data)

summary(bike_data)
dim(bike_data)


is.integer(bike_data)
is.null(bike_data)

bike_data = data.frame(bike_data)
str(bike_data)

## EXPLORATORY  DATA_ANALYSIS
library(dplyr)
library(corrplot)
library(ggplot2)
library(stats)
# formating of dataset column Binning the categorical variable 
bike_data$season = factor(format(bike_data$season,format="%A"),levels = c("1", "2","3","4") , labels = c("Spring","Summer","Fall","Winter"))
table(bike_data$season)
bike_data$holiday = factor(format(bike_data$holiday, format="%A"),levels = c("0", "1") , labels = c("generic","Holiday"))
table(bike_data$holiday)
bike_data$weathersit = factor(format(bike_data$weathersit, format="%A"),levels = c("1", "2","3","4"),labels = c("Pleasant","Moderate","Bad","Extreme"))
table(bike_data$weathersit)
bike_data$yr = factor(format(bike_data$yr, format="%A"),levels = c("0", "1") , labels = c("2011","2012"))
table(bike_data$yr)

# from Data set we need to calculate the colum according to the given value and put in to it
bike_data$actual_windspeed = bike_data$windspeed*67
bike_data$actual_feel_temp = bike_data$atemp*50
bike_data$actual_humidity <- bike_data$hum*100
bike_data$actual_temp = bike_data$temp*41
bike_data$mean_acttemp_feeltemp <- (bike_data$actual_temp+bike_data$actual_feel_temp)/2
str(bike_data)
summary(bike_data)
nrow(bike_data)
ncol(bike_data)
dim(bike_data)
names(bike_data)
max(bike_data$casual)
max(bike_data$registered)
lookup <- data.frame("numbers"=c("1","2","3","4"), "weather"=c("nice","cloudy", "wet", "lousy"))
head(bike_data)
lookup.month<- data.frame("mnth" = c(1:12),"mnth.name" = c("01Jan", "02Feb", "03March", "04April", "05May", "06June", "07July", "08Aug", "09Sept", "10Oct", "11Nov", "12Dec"), stringsAsFactors = FALSE)
bike_data <- merge(x = bike_data, y= lookup.month, by = 'mnth')
# Convert the nomalized windspeed and humidity
bike_data$raw.windspeed <- (bike_data$windspeed*67)
bike_data$raw.hum <- (bike_data$hum * 100)
head(bike_data)


#visualization
# Setting the margins to fit the plot
par(mar = rep(2, 4))
# Distributio of the target variable
hist_bikedata = hist(bike_data$cnt, breaks = 25, ylab = 'Rental Freqancy of Bike', xlab = 'Total Count of Bike Rental ', main = 'Total Count of Bike Rental ',col = 'brown',border="blue")
xfit = seq(min(bike_data$cnt),max(bike_data$cnt), length = 75)
yfit = dnorm(xfit, mean =mean(bike_data$cnt),sd=sd(bike_data$cnt))
yfit = yfit*diff(hist_bikedata$mids[1:2])*length(bike_data$cnt)
lines(xfit,yfit, col='black', lwd= 7)

#plot for bike rental and date 
plot(bike_data$dteday, bike_data$cnt,
     main = "Bike Rentals Vs DateDay",
     xlab = "Year",
     ylab = "Bike Rentals",
     col  = "red",
     pch  = 19)

#plot for Bike Rentals and Weather
boxplot(bike_data$cnt ~ bike_data$weathersit, data = bike_data, 
        main = "Bike Rentals Vs Weather",
        xlab = "Weather",
        ylab = "Bike Rentals",   
        col = c("green","blue","red"))
#plot for Bike Rentals and Holiday
boxplot(bike_data$cnt ~ bike_data$holiday, data = bike_data, 
        main = "Bike Rentals Vs Holiday",
        xlab = "Holiday",
        ylab = "Bike Rentals",   
        col = c("green","blue"))
# plot for Bike Rentals and season
boxplot(bike_data$cnt ~ bike_data$season, data = bike_data, 
        main = "Bike Rentals Vs season",
        xlab = "season",
        ylab = "Bike Rentals",   
        col = c("red","blue","green","blue"))
#plot for Bike Rentals Vs Year
boxplot(bike_data$cnt ~ bike_data$yr, data = bike_data, 
        main = "Bike Rentals Vs Year",
        xlab = "Year",
        ylab = "Bike Rentals",   
        col = c("red","green"))


#spliting the dataset into train and test so we can apply our model 
library(caTools)
split = sample.split(bike_data$cnt,SplitRatio = 0.75)
train_data = subset(bike_data, split == TRUE)
test_data = subset(bike_data, split == FALSE)
train_data
test_data


#ggplots for count with sesason, temprature, and day

ggplot(train_data,aes(temp,cnt)) +
  geom_point(aes(color=temp),alpha=0.2)+
  theme_bw()

ggplot(train_data,aes(dteday,cnt)) + 
  geom_point(aes(color=temp),alpha=0.2) + 
  scale_color_gradient(high='red',low='blue') + 
  theme_bw() 

train_data$season = factor(train_data$season)
ggplot(train_data,aes(season,cnt)) +
  geom_boxplot(aes(color=season),alpha=0.5) + 
  theme_bw()



#ggplot with working day 1
ggplot(filter(train_data,workingday==1),aes(dteday,cnt)) +
  geom_point(aes(color=temp),alpha=0.5,position=position_jitter(w=1, h=0)) + #position_jitter adds random noise in order to read the plot easier
  scale_color_gradientn(colors=c('dark green','green','light green','yellow','orange','red','dark blue','blue','light blue')) +
  theme_bw()

#ggplot with working day 0
ggplot(filter(train_data,workingday==0),aes(dteday,cnt)) +
  geom_point(aes(color=temp),alpha=0.5,position=position_jitter(w=1, h=0)) + 
  scale_color_gradientn(colors=c('dark green','green','light green','yellow','orange','red','dark blue','blue','light blue')) +
  theme_bw()




#model Selection 
# i try SVM but it give me nagetive result logestic regression is also negative impact on these data set 
#i choose random forest because its easly to predict the user and situation and no. of bike propasanl to dependent variable
library(randomForest) 
regressor = randomForest(cnt ~ season + holiday+ workingday+ weathersit+ temp+ hum+ windspeed+casual+registered ,data= train_data)

# Predicting a new result with Random Forest Regression
y_pred = predict(regressor, test_data)

# Save the results
results <- data.frame(dteday = test_data$dteday, cnt = y_pred)

# Write the results to a csv file
write.csv(results, file = 'BikeSharingDemand_RandomForest.csv', row.names = FALSE, quote=FALSE)



