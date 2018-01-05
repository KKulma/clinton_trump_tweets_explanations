library(packrat)
packrat::init()

#####################

library(readr)
library(lime)
library(xgboost) # the classifier
library(caret)
library(dplyr)
#library(text2vec)


tweets <- read_csv('tweets.csv')
str(tweets)

table(tweets$handle)

all_tweets <- tweets %>% 
  rename(author = handle) %>% 
  select(author, text)


set.seed(1234)
trainIndex <- createDataPartition(all_tweets$author, p = .8, 
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)

train_tweets <- all_tweets[ trainIndex,]
test_tweets <- all_tweets[ -trainIndex,]

str(train_tweets)
str(test_tweets)

# Tokenize data


# BoW matrix generation
dtm_train = get_matrix(train_sentences$text)
dtm_test = get_matrix(test_sentences$text)

# Create boosting model for binary classification (-> logistic loss)
# Other parameters are quite standard
param <- list(max_depth = 7, 
              eta = 0.1, 
              objective = "binary:logistic", 
              eval_metric = "error", 
              nthread = 1)

xgb_model <- xgb.train(
  param, 
  xgb.DMatrix(dtm_train, label = train_sentences$class.text == "OWNX"),
  nrounds = 50
)


# We use a (standard) threshold of 0.5
predictions <- predict(xgb_model, dtm_test) > 0.5
test_labels <- test_sentences$class.text == "OWNX"

# Accuracy
print(mean(predictions == test_labels))


# We select 10 sentences from the label OWNX
sentence_to_explain <- head(test_sentences[test_labels,]$text, 5)
explainer <- lime(sentence_to_explain, model = xgb_model, 
                  preprocess = get_matrix)
explanation <- explain(sentence_to_explain, explainer, n_labels = 1, 
                       n_features = 2)

# Most of the words choosen by Lime
# are related to the team (we, our)
# or part of the paper (Section, in)
explanation[, 2:9]


plot_features(explanation)


plot_text_explanations(explanation)


# Launching the application is done in one command
interactive_text_explanations(explainer)
