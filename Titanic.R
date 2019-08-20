################################################################################################
#### This code is for Exploring the Titanic Data Set
## Created: July 25, 2019
## Edited:
################################################################################################

rm(list = ls())

# set working directory
setwd("/Users/m/Desktop/Kaggle/Titanic")


library(ggplot2)
library(ggthemes)
library(scales)
library(dplyr)
library(mice)
library(randomForest)



################################################################################################
#### Load the data
################################################################################################
train <- read.csv("train.csv", head=T, sep=",")
test <- read.csv("test.csv", head=T, sep=",")

# combine the test and train data into one data set
full <- bind_rows(train, test)

# compactly display the structure of the full data set
str(full)



################################################################################################
#### Feature Engineering
################################################################################################
# take the title from the passenger name
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

# show title counts by sex
table(full$Sex, full$Title)

# combine titles with low counts to "rare" level
rare_title <- c('Capt', 'Col', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'the Countess')

# reassign mlle, mme, and ms accordingly
full$Title[full$Title =='Mlle'] <- 'Miss'
full$Title[full$Title =='Mme'] <- 'Mrs'
full$Title[full$Title =='Ms'] <- 'Miss'

# reassign those deemed rare as a new title, "Rare Title"
full$Title[full$Title %in% rare_title] <- 'Rare Title'

# repeat title countrs by sex with updated data
table(full$Sex, full$Title)

# grab the surnames from passengers
full$Surname <- sapply(full$Name, function(x) strsplit(x, split = '[,.]')[[1]][1])

# explicitly state the number of unique surnames
cat(paste('We have', nlevels(factor(full$Surname)), 'unique surnames.'))



##################################
#### Do families sink or swim ####
##################################
# create a varaibles based on the number of memebers of families
full$Fsize <- full$SibSp + full$Parch + 1

# create a family varaible
full$Family <- paste(full$Surname, full$Fsize, sep='_')

# use ggplot2 to visualize the relationship between family size and survival
ggplot(full[1:891,], aes(x=Fsize, fill=factor(Survived))) + geom_bar(stat='count', position='dodge') + scale_x_continuous(breaks=c(1:11)) + labs(x='Family Size') + theme_few()

# we can see those with family size > 4 are less likely to survive
# we can also see those that are alone are less likely to survive
# split this variable into 3 levels since there are significantly less large families
# discretize family size
full$FsizeD[full$Fsize == 1] <- 'singles'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

# show family survival using a mosaic plit
mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=T)
# this confirms our previous assumptions about those that traveled alone as well as those with large families



##############################
#### Treat More Variables ####
##############################
# can see that the cabin variable has nunmerous missing values
# the first characted in the cabin variable is the deck
# create a deck variable to get passenger decks A - F
full$Deck <- factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))



################################################################################################
#### Missing Data
################################################################################################
# rely on the distribution of the data as well as prediction to fill in missing values
# do this rather than delete entire rows since this is a relatively small data set
# passengers 62 and 830 are missing Embarkment
# will infer their Embarkment based on their passenger class and fare:
  # they each paid $80
  # and each are from class 1

# remove these two from the data set so as to accurately describe those with Embarkment info
embark_fare <- full %>% filter(PassengerId !=62 & PassengerId !=830)

# visualize Embarkment, Passenger Class, and Median Fare
ggplot(embark_fare, aes(x=Embarked, y=Fare, fill=factor(Pclass))) + 
  geom_boxplot() + 
  geom_hline(aes(yintercept=80), colour='red', linetype='dashed', lwd=2) + 
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

# we will infer that each passenger Embarked at Charbourg(C) since the median price for 1st class passengers was ~$80
full$Embarked[c(62, 830)] <- 'C'

# passemnger 1044 has and N/A fare value
full[1044,]
# 3rd class passenger that Embarked at Southampton(S)
# visualize fares for those sharing class and Embarkment with 1044
ggplot(full[full$Pclass == '3' & full$Embarked == 'S',],
       aes(x=Fare)) + 
  geom_density(fill='#99d6ff', alpha=0.4) +
  geom_vline(aes(xintercept=median(Fare, na.rm=T)), colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous(labels=dollar_format()) +
  theme_few()
# seems reasonable to assign the fare value with the median for their Class/Embarkment = $8.05
# replace the missing fare value
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S',]$Fare, na.rm=T)



####################################
#### Predict Missing Age Values ####
####################################
# show number of missing age values
sum(is.na(full$Age))

# Multiple Imputations using Chained Equations (MICE) 
# factorize the factor variables and then perform MICE imputation
factor_vars <- c('PassengerId', 'Pclass', 'Sex', 'Embarked', 'Title', 'Surname', 'Family', 'FsizeD')
full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

# set the seed for reproducibility
set.seed(1234)

# perform MICE imputation, excluding less-than-useful variables
mice_mod <- mice(full[, !names(full) %in% c('PassengerId', 'Names', 'Ticket', 'Cabin', 'Family', 'Surname', 'Survived')], method='rf')

# save the compelte output 
mice_output <- complete(mice_mod)

# compare the mice_output with original passenger age distribution to confirm nothing is obviously wrong
par(mfrow=c(1,2))
hist(full$Age, freq=F, main="Original Age Data", col='darkgreen', ylim=c(0, 0.04))
hist(mice_output$Age, freq=F, main="Mice Output Age Data", col='lightgreen', ylim=c(0, 0.04))

# these look pretty spot-on 
# replace Age with the MICE model
full$Age <- mice_output$Age

# show the new number of missing Age values
sum(is.na(full$Age))



##################################
#### More Feature Engineering ####
##################################
# create age-dependent variables now -> child and mother
  # child is someone under 18 years old
  # mother is someone over 18 years olf, female, and has more than 0 children, does not have the title 'Miss'

# look at the relationship between age and survivial
ggplot(full[1:891,], aes(Age, fill=factor(Survived))) +
  geom_histogram() +
  # omc;ide sex since we know it's a significant predictor
  facet_grid(.~Sex) + 
  theme_few()

# create the column child and indicate whether 'Child' or 'Adult'
full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

# show counts
table(full$Child, full$Survived)
# pretty even amongst children who survived and those that did not

# create the Mother variable
full$Mother <- 'Not Mother'
full$Mother[full$Sex =='female' & full$Parch > 0 & full$Age >18 & full$Title != 'Miss'] <- 'Mother'

# show counts
table(full$Mother, full$Survived)

# factorize our new variables
full$Child <- factor(full$Child)
full$Mother <- factor(full$Mother)

# display missing data patterns
md.pattern(full)



################################################################################################
#### Prediction
################################################################################################
# predict who survives the Titanic by using the randomForest

# split into training and test sets
train <- full[1:891,]
test <- full[892:1309,]

# set seed
set.seed(1235)

# build our model using randomForest
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, data=train)

# show model error
plot(rf_model, y_lim=c(0, 0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)
# black line shows overall error rate which falls below 20%
# red and green lines show the error rate for 'died' and 'survived', respectively
  # can see higher success rate of predicting death than survival



#############################
#### Variable Importance ####
#############################
# look at the relative variable importance by plotting the mean decrease in Gini calculated across all trees
importance <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance),
                            Importance = round(importance[ ,'MeanDecreaseGini'], 2))

# create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# plot to visualize the relative importance of variables
ggplot(rankImportance, aes(x=reorder(Variables, Importance),
                           y=Importance, fill=Importance)) +
  geom_bar(stat='identity') +
  geom_text(aes(x=Variables, y=0.5, label=Rank),
            hjust=0, vjust=0.55, size=4, colour='red') +
  labs(x='Variables') +
  coord_flip() +
  theme_few()



####################
#### Prediction ####
####################
# predict using the test set
prediction <- predict(rf_model, test)

# save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)

# write the solution to a file
write.csv(solution, file='rf_model_Solution.csv', row.names=F)






########################
#### Model Accuracy ####
########################
acc <- predict(rf_model, train)
accuracy <- data.frame(PssID = train$PassengerId, survival=acc)
accuracy$real = if_else(train$Survived == accuracy$survival, 1, 0)
mean(accuracy$real)  
# 0.9169473



















