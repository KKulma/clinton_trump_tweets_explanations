#library(packrat)
#packrat::init()

#####################

library(readr)
library(lime)
library(xgboost) # the classifier
library(caret)
library(dplyr)
library(tibbe)
library(text2vec)
library(qdapRegex) # remove urls from text


tweets <- read_csv('tweets.csv')
str(tweets)

table(tweets$handle)

all_tweets <- tweets %>% 
  rename(author = handle) %>% 
  select(author, text) %>% 
  mutate(text = rm_url(text)) %>% #removes URLs from text
  na.omit()

head(as.data.frame(all_tweets))

set.seed(1234)
trainIndex <- createDataPartition(all_tweets$author, p = .8, 
                                  list = FALSE, 
                                  times = 1)

train_tweets <- all_tweets[ trainIndex,]
test_tweets <- all_tweets[ -trainIndex,]

str(train_tweets)
str(test_tweets)

# Tokenize data

get_matrix2 <- function(text) {
  it <- itoken(text, progressbar = TRUE)
  create_dtm(it, vectorizer = hash_vectorizer(ngram = c(1L, 2L)))
}

# BoW matrix generation
dtm_train2 = get_matrix2(train_tweets$text)
dtm_test2 = get_matrix2(test_tweets$text)

dim(dtm_train2)
dim(dtm_test2)

# Create boosting model for binary classification (-> logistic loss)
# Other parameters are quite standard
param <- list(max_depth = 7, 
              eta = 0.1, 
              objective = "binary:logistic", 
              eval_metric = "error", 
              nthread = 1)

set.seed(1234)
xgb_model2 <- xgb.train(
  param, 
  xgb.DMatrix(dtm_train2, label = train_tweets$author == "realDonaldTrump"),
  nrounds = 50
)




# We use a (standard) threshold of 0.5
predictions <- predict(xgb_model2, dtm_test2) > 0.5
test_labels <- test_tweets$author == "realDonaldTrump"

# Accuracy
print(mean(predictions == test_labels))


# select only correct predictions
predictions_tb = predictions %>% as_tibble() %>% 
  rename_(predict_label = names(.)[1]) %>%
  tibble::rownames_to_column()

library(dplyr)

correct_pred = test_tweets %>%
  tibble::rownames_to_column() %>% 
  mutate(test_label = author == "realDonaldTrump") %>%
  left_join(predictions_tb) %>%
  filter(test_label == predict_label) %>% 
  pull(text) %>% 
  head(5)

sentence_to_explain <- head(test_tweets[test_labels,]$text, 5)    


# We select 10 sentences from the label OWNX

detach("package:dplyr", unload=TRUE)


explainer <- lime(sentence_to_explain, model = xgb_model2, 
                  preprocess = get_matrix2)

all_explanation <- lime::explain(sentence_to_explain, explainer, n_labels = 1, 
                       n_features = 5)

corr_explanation <- lime::explain(correct_pred, explainer, n_labels = 1, 
                       n_features = 5)


str(sentence_to_explain)
str(correct_pred)



# Most of the words choosen by Lime
# are related to the team (we, our)
# or part of the paper (Section, in)
corr_explanation[, 1:10]

plot_features(corr_explanation)

plot_text_explanations(corr_explanation)


# Launching the application is done in one command
interactive_text_explanations(explainer)




