library("skimr")
library("tidyverse")
library("ggplot2")
library("rnaturalearth")
library("rnaturalearthdata")
library("rgeos")
library("ggforce")
library("dplyr")
library("rsample")
library("data.table")
library("mlr3verse")
library(mlr3)
library(mlr3learners)
library(mlr3viz)
library(precrec)
library(mlr3proba)

#Read Data#
hearts <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv")
skim(hearts)
hearts<-hearts[order(hearts$fatal_mi),]
summary(hearts)
#plot#
DataExplorer::plot_bar(hearts, ncol = 2)
DataExplorer::plot_boxplot(hearts, by = "fatal_mi", ncol = 3)

hearts2<-hearts
hearts2$fatal_mi<-as.factor(unlist(hearts$fatal_mi))

hearts$anaemia<-as.factor(unlist(hearts$anaemia))
hearts$diabetes<-as.factor(unlist(hearts$diabetes))
hearts$high_blood_pressure<-as.factor(unlist(hearts$high_blood_pressure))
hearts$sex<-as.factor(unlist(hearts$sex))
hearts$smoking<-as.factor(unlist(hearts$smoking))
hearts$fatal_mi<-as.factor(unlist(hearts$fatal_mi))
hearts$age<-as.integer(unlist(hearts$age))
#k-fold#
set.seed(200)
hearts_task <- TaskClassif$new(id = "Hearts",
                              backend = hearts, 
                              target = "fatal_mi",
                              positive = "1")
hearts_task2 <- TaskClassif$new(id = "Hearts",
                               backend = hearts2, 
                               target = "fatal_mi",
                               positive = "1")

cv <- rsmp("cv",folds = 10)
cv$instantiate(hearts_task)
bs<-rsmp("bootstrap",ratio=0.8)
bs$instantiate(hearts_task)
train_set = sample(hearts_task$row_ids, 0.8 * hearts_task$nrow)
test_set = setdiff(hearts_task$row_ids, train_set)
train_set2 = sample(hearts_task2$row_ids, 0.8 * hearts_task2$nrow)
test_set2 = setdiff(hearts_task2$row_ids, train_set)

#models#
library(xgboost)
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.01)
lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")
pl_xgb <- po("encode") %>>%
  po(lrn_xgboost)
#traning#
lrn_baseline$train(hearts_task,row_ids = train_set)
lrn_cart$train(hearts_task,row_ids = train_set)
lrn_cart_cp$train(hearts_task,row_ids = train_set)
lrn_xgboost$train(hearts_task2,row_ids = train_set2)
summary(lrn_cart$model)
#predict#
pb=lrn_baseline$predict(hearts_task,row_ids = test_set)
autoplot(pb, type = "roc")
pb$score(msr("classif.auc"))

pc=lrn_cart$predict(hearts_task,row_ids = test_set)
autoplot(pc, type = "roc")
pcp$score(msr("classif.auc"))

px=lrn_xgboost$predict(hearts_task2,row_ids = test_set2)
autoplot(px, type = "roc")
px$score(msr("classif.auc"))

#evaluation#
res <- benchmark(data.table(
  task       = list(hearts_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    pl_xgb),
  resampling = list(cv)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
autoplot(res)
#choose tree with prun#
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)
res_cart_cv <- resample(hearts_task, lrn_cart_cv, cv, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[10]]$model)
#evaluation#
res_cv <- benchmark(data.table(
  task       = list(hearts_task),
  learner    = list(lrn_cart_cv),
  resampling = list(cv)
), store_models = TRUE)

res_cv$aggregate(list(msr("classif.ce"),
                        msr("classif.acc"),
                        msr("classif.fpr"),
                        msr("classif.fnr")))
plot(res_cv$resample_result(1)$learners[[10]]$model, compress = TRUE, margin = 0.1)
text(res_cv$resample_result(1)$learners[[10]]$model, use.n = TRUE, cex = 0.8)


#what if I use Random forest#
learner_rf = lrn("classif.ranger", predict_type = "prob", importance = "permutation")
#training#
learner_rf$train(hearts_task, row_ids = train_set)
importance = as.data.table(learner_rf$importance(), keep.rownames = TRUE)
colnames(importance) = c("Feature", "Importance")
ggplot(data=importance,
       aes(x = reorder(Feature, Importance), y = Importance)) + 
  geom_col() + coord_flip() + xlab("")
#predcition#
pred_cart = lrn_cart$predict(hearts_task, row_ids = test_set)
pred_rf = learner_rf$predict(hearts_task, row_ids = test_set)

pred_cart
pred_cart$confusion
pred_rf$confusion
pred_cart$score(msr("classif.acc"))
pred_rf$score(msr("classif.acc"))

?autoplot

#evaluation#
res_rf <- benchmark(data.table(
  task       = list(hearts_task),
  learner    = list(learner_rf),
  resampling = list(bs)
), store_models = TRUE)

res_rf$aggregate(list(msr("classif.ce"),
                      msr("classif.acc"),
                      msr("classif.fpr"),
                      msr("classif.fnr")))


#hyperparameter selection#
learner_rf$param_set
as.data.table(learner_rf$param_set)[,.(id, class, lower, upper)]
search_space = ps(
  mtry = p_int(lower = 1, upper =12),
  num.trees = p_int(lower = 1, upper = 40),
  min.node.size=p_int(lower = 1, upper = 10)
)
search_space
measures = msrs(c("classif.ce","classif.acc","classif.fpr","classif.fnr"))
evals20 = trm("evals", n_evals = 20)

instance = TuningInstanceMultiCrit$new(
  task = hearts_task,
  learner = learner_rf,
  resampling = bs,
  measure = measures,
  search_space = search_space,
  terminator = evals20
)
instance
tuner = tnr("random_search")
tuner$optimize(instance)
instance$result_learner_param_vals
instance$result_y

plot(classif.fnr ~ classif.fpr, instance$archive$data)
points(classif.fnr ~ classif.fpr, instance$result, col = "red")
points(classif.fnr ~ classif.fpr, instance$result[1], col = "green")
points(classif.fnr ~ classif.fpr, instance$result[3], col = "blue")

plot(classif.acc ~ classif.ce, instance$archive$data)
points(classif.acc ~ classif.ce, instance$result, col = "red")
points(classif.acc ~ classif.ce, instance$result[1], col = "green")
points(classif.acc ~ classif.ce, instance$result[3], col = "blue")
#train it on the dataset#
learner_rf4 = lrn("classif.ranger", predict_type = "prob", importance = "permutation")
learner_rf4$param_set$values <- instance$result_learner_param_vals[[1]]
learner_rf4$train(hearts_task,row_ids = train_set)

#prediction#
predictions = learner_rf4$predict(hearts_task, row_ids = test_set)

##evaluation##
res_rf4<- benchmark(data.table(
  task       = list(hearts_task),
  learner    = list(learner_rf4),
  resampling = list(bs)
), store_models = TRUE)

res_rf4$aggregate(list(msr("classif.ce"),
                      msr("classif.acc"),
                      msr("classif.fpr"),
                      msr("classif.fnr"),
                      msr("classif.tpr"),
                      msr("classif.tnr")))

autoplot(res_rf4,type="prc")
autoplot(res_rf4,type="roc")

importance = as.data.table(learner_rf4$importance(), keep.rownames = TRUE)
colnames(importance) = c("Feature", "Importance")
ggplot(data=importance,
       aes(x = reorder(Feature, Importance), y = Importance)) + 
  geom_col() + coord_flip() + xlab("")
#excutive summary#
ce = c(0.1603789,0.8396211)
barplot(ce,main="Predict Accurancy",col=c("#ED1C24","#22B14C"),names.arg=c("wrong","correct"))

pr = c(0.0750329,0.9249671)
barplot(pr,main="Patients Have the Infarction ",col=c("#ED1C24","#22B14C"),names.arg=c("wrong","correct"))

pr = c(0.3617056,0.6382944)
barplot(pr,main="Patients Don't Have the Infarction ",col=c("#ED1C24","#22B14C"),names.arg=c("wrong","correct"))
