# ============================================================
# Analise: Gallstone - https://archive.ics.uci.edu/dataset/1150/gallstone-1
# Autores: - Luis Henrique Rodrigues de Oliveira, 
#           - Rodrigo Simão Guimarães
#           - Luan Henrique Campos Soares

############################################################
# 0. INSTALAÇÃO DE PACOTES (EXECUTAR UMA VEZ)
############################################################
# Função auxiliar para instalar pacotes caso não estejam instalados
# (Está comentada pois deve ser executada apenas na primeira vez)

# install_if_missing <- function(p) {
#   if (!require(p, character.only = TRUE)) {
#     install.packages(p)
#     library(p, character.only = TRUE)
#   }
# }
#
# lapply(c(
#   "dplyr","ggplot2","readxl","caret",
#   "randomForest","e1071","xgboost","pROC",
#   "tidyr","MASS"
# ), install_if_missing)

############################################################
# 1. PACOTES
############################################################
# Carregamento das bibliotecas necessárias para análise,
# modelagem, métricas e visualizações

library(dplyr)
library(readxl)
library(caret)
library(randomForest)
library(e1071)
library(xgboost)
library(pROC)
library(ggplot2)
library(tidyr)
library(MASS)

############################################################
# 2. CARREGAMENTO DOS DADOS
############################################################
# Leitura do dataset e padronização dos nomes das colunas

df <- read_excel("dataset-uci.xlsx")
names(df) <- make.names(names(df))

# Definição da variável alvo (target)
target <- "Gallstone.Status"

############################################################
# 3. PRÉ-PROCESSAMENTO
############################################################
# Conversão da variável alvo para fator (classificação binária)
# Renomeação dos níveis para "No" e "Yes"

df[[target]] <- as.factor(df[[target]])
levels(df[[target]]) <- c("No", "Yes")

############################################################
# 4. SPLIT TREINO / TESTE
############################################################
# Separação estratificada dos dados (80% treino / 20% teste)

set.seed(123)
train_index <- createDataPartition(df[[target]], p = 0.8, list = FALSE)

train_raw <- df[train_index, ]
test_raw  <- df[-train_index, ]

############################################################
# 5. LIMPEZA DE FEATURES
############################################################
# Remoção de variáveis irrelevantes ou redundantes

predictor_names <- setdiff(names(train_raw), target)

# 5.1 Remover variáveis quase constantes (near zero variance)
nzv <- nearZeroVar(train_raw[, predictor_names])
if (length(nzv) > 0) predictor_names <- predictor_names[-nzv]

# 5.2 Remover variáveis altamente correlacionadas (> 0.9)
num_cols <- sapply(train_raw[, predictor_names], is.numeric)

if (sum(num_cols) > 1) {
  cor_mat <- cor(train_raw[, predictor_names[num_cols]], use = "pairwise.complete.obs")
  high_cor <- findCorrelation(cor_mat, cutoff = 0.9)

  if (length(high_cor) > 0) {
    drop <- predictor_names[num_cols][high_cor]
    predictor_names <- setdiff(predictor_names, drop)
  }
}

# Aplicar seleção de variáveis ao treino e teste
train <- train_raw[, c(target, predictor_names)]
test  <- test_raw[, c(target, predictor_names)]

############################################################
# 6. NORMALIZAÇÃO + IMPUTAÇÃO (KNN)
############################################################
# Padronização dos dados + imputação de valores faltantes via KNN

pre_proc <- preProcess(
  train[, predictor_names],
  method = c("center", "scale", "knnImpute")
)

# Aplicação da transformação
train_x <- predict(pre_proc, train[, predictor_names])
test_x  <- predict(pre_proc, test[, predictor_names])

# Reanexar variável alvo
train <- cbind(train_x, Gallstone.Status = train[[target]])
test  <- cbind(test_x, Gallstone.Status = test[[target]])

names(train)[ncol(train)] <- target
names(test)[ncol(test)] <- target

predictor_names <- setdiff(names(train), target)

############################################################
# 7. CONTROLE DE TREINAMENTO (CV)
############################################################
# Configuração de validação cruzada repetida
# Métrica principal: AUC (ROC)

train_control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

############################################################
# 8. SELEÇÃO DE VARIÁVEIS (BACKWARD - REGRESSÃO LOGÍSTICA)
############################################################
# Seleção automática de variáveis usando StepAIC (backward)

get_backward_vars <- function(train_df, target_name) {

  # Modelo completo
  full_model <- glm(
    as.formula(paste(target_name, "~ .")),
    data = train_df,
    family = binomial()
  )

  # Stepwise backward
  step_model <- stepAIC(full_model, direction = "backward", trace = FALSE)

  # Variáveis finais selecionadas
  vars <- setdiff(names(coef(step_model)), "(Intercept)")
  unique(vars)
}

selected_vars <- tryCatch(
  get_backward_vars(train, target),
  error = function(e) predictor_names
)

# Garantir robustez
selected_vars <- intersect(selected_vars, predictor_names)
if (length(selected_vars) < 2) selected_vars <- predictor_names

############################################################
# 9. MODELO 1 — XGBOOST (BOOSTING)
############################################################
# Função para treinar XGBoost com validação cruzada interna

run_xgb_gbtree <- function(train_df, target_name, vars, seed = 123) {

  set.seed(seed)

  x_train <- as.matrix(train_df[, vars])
  y_train <- ifelse(train_df[[target_name]] == "Yes", 1, 0)

  dtrain <- xgb.DMatrix(data = x_train, label = y_train)

  # Parâmetros principais do modelo
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.05,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8
  )

  # Cross-validation interna do XGBoost
  cv_fit <- tryCatch(
    xgb.cv(
      data = dtrain,
      params = params,
      nrounds = 300,
      nfold = 10,
      early_stopping_rounds = 20,
      verbose = 0
    ),
    error = function(e) NULL
  )

  if (is.null(cv_fit)) {
    stop("Falha no xgb.cv para o modelo gbtree.")
  }

  # Seleção automática do número ideal de árvores
  eval_df <- as.data.frame(cv_fit$evaluation_log)
  best_nrounds <- suppressWarnings(as.integer(cv_fit$best_iteration[1]))

  if (is.na(best_nrounds) || length(best_nrounds) == 0 || best_nrounds < 1) {
    auc_col <- if ("test_auc_mean" %in% names(eval_df)) "test_auc_mean" else if ("test_auc" %in% names(eval_df)) "test_auc" else NA_character_
    if (!is.na(auc_col) && nrow(eval_df) > 0) {
      best_nrounds <- which.max(eval_df[[auc_col]])
    }
  }

  # Fallback de segurança
  if (is.null(best_nrounds) || length(best_nrounds) == 0 || is.na(best_nrounds) || best_nrounds < 1) {
    best_nrounds <- 100L
  }

  # Treinamento final
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = best_nrounds,
    verbose = 0
  )

  list(model = model, vars = vars, nrounds = best_nrounds)
}

xgb_model <- run_xgb_gbtree(train, target, selected_vars)

############################################################
# 10. MODELO 2 — RANDOM FOREST (BAGGING)
############################################################
model_rf <- train(
  as.formula(paste(target, "~ .")),
  data = train[, c(target, selected_vars)],
  method = "rf",
  trControl = train_control,
  metric = "ROC",
  tuneLength = 10,
  ntree = 500
)

############################################################
# 11. MODELO 3 — SVM (KERNEL RADIAL)
############################################################
model_svm <- train(
  as.formula(paste(target, "~ .")),
  data = train[, c(target, selected_vars)],
  method = "svmRadial",
  trControl = train_control,
  metric = "ROC",
  tuneLength = 10
)

############################################################
# 11.1 MELHORIAS — OTIMIZAÇÃO DE THRESHOLD E BLEND
############################################################
# Função para encontrar melhor threshold via índice de Youden

optimize_threshold <- function(obs, prob) {
  roc_obj <- roc(obs, prob, quiet = TRUE)
  thr <- coords(roc_obj, x = "best", best.method = "youden", ret = "threshold")
  thr <- as.numeric(thr)[1]
  if (is.na(thr)) thr <- 0.5
  thr
}

# Avaliação usando threshold customizado
avaliar_prob_threshold <- function(prob, obs, threshold = 0.5) {
  pred <- factor(ifelse(prob > threshold, "Yes", "No"), levels = c("No", "Yes"))
  cm <- confusionMatrix(pred, obs)
  roc_obj <- roc(obs, prob, quiet = TRUE)
  list(
    cm = cm,
    auc = as.numeric(auc(roc_obj)),
    threshold = threshold
  )
}

############################################################
# 12. FUNÇÕES DE AVALIAÇÃO
############################################################
# Avaliação padrão de modelos (Confusion Matrix + ROC/AUC)

avaliar_modelo <- function(model, test_df, target_name, nome, xgb = FALSE, vars = NULL) {

  if (xgb) {
    prob <- predict(model, as.matrix(test_df[, vars]))
    pred <- factor(ifelse(prob > 0.5, "Yes", "No"), levels = c("No","Yes"))
  } else {
    prob <- predict(model, test_df, type = "prob")[, "Yes"]
    pred <- predict(model, test_df)
  }

  cm <- confusionMatrix(pred, test_df[[target_name]])
  roc_obj <- roc(test_df[[target_name]], prob)

  cat("\n====================\n", nome, "\n====================\n")
  print(cm)
  cat("AUC:", auc(roc_obj), "\n")

  list(cm = cm, auc = auc(roc_obj), prob = prob)
}

# Função para cálculo do F1-Score
f1_score <- function(cm) {
  p <- as.numeric(cm$byClass["Pos Pred Value"])
  r <- as.numeric(cm$byClass["Sensitivity"])
  if ((p + r) == 0) return(0)
  2 * (p * r) / (p + r)
}

############################################################
# 13. AVALIAÇÃO DOS MODELOS
############################################################
test_best <- test[, c(target, selected_vars)]

res_rf  <- avaliar_modelo(model_rf, test_best, target, "Random Forest")
res_svm <- avaliar_modelo(model_svm, test_best, target, "SVM")
res_xgb <- avaliar_modelo(xgb_model$model, test_best, target, "XGBoost", TRUE, selected_vars)

############################################################
# 13.1 EXECUÇÃO DAS MELHORIAS (THRESHOLD + BLEND)
############################################################
# Otimização de threshold com base no conjunto de treino

prob_rf_train <- predict(model_rf, train[, c(target, selected_vars)], type = "prob")[, "Yes"]
prob_xgb_train <- predict(xgb_model$model, as.matrix(train[, selected_vars]))

thr_rf <- optimize_threshold(train[[target]], prob_rf_train)
thr_xgb <- optimize_threshold(train[[target]], prob_xgb_train)

# Probabilidades no teste
prob_rf_test <- res_rf$prob
prob_xgb_test <- res_xgb$prob

# Ensemble simples (média das probabilidades)
prob_blend_train <- (prob_rf_train + prob_xgb_train) / 2
prob_blend_test <- (prob_rf_test + prob_xgb_test) / 2
thr_blend <- optimize_threshold(train[[target]], prob_blend_train)

# Avaliação dos modelos com threshold otimizado
res_rf_thr <- avaliar_prob_threshold(prob_rf_test, test_best[[target]], thr_rf)
res_xgb_thr <- avaliar_prob_threshold(prob_xgb_test, test_best[[target]], thr_xgb)
res_blend <- avaliar_prob_threshold(prob_blend_test, test_best[[target]], thr_blend)

# Tabela consolidada das melhorias aplicadas
melhorias <- data.frame(
  Modelo = c(
    "RF (threshold otimizado)",
    "XGBoost (threshold otimizado)",
    "Blend RF+XGB (threshold otimizado)"
  ),
  Threshold = c(res_rf_thr$threshold, res_xgb_thr$threshold, res_blend$threshold),
  Accuracy = c(
    as.numeric(res_rf_thr$cm$overall["Accuracy"]),
    as.numeric(res_xgb_thr$cm$overall["Accuracy"]),
    as.numeric(res_blend$cm$overall["Accuracy"])
  ),
  Sensitivity = c(
    as.numeric(res_rf_thr$cm$byClass["Sensitivity"]),
    as.numeric(res_xgb_thr$cm$byClass["Sensitivity"]),
    as.numeric(res_blend$cm$byClass["Sensitivity"])
  ),
  Specificity = c(
    as.numeric(res_rf_thr$cm$byClass["Specificity"]),
    as.numeric(res_xgb_thr$cm$byClass["Specificity"]),
    as.numeric(res_blend$cm$byClass["Specificity"])
  ),
  AUC = c(res_rf_thr$auc, res_xgb_thr$auc, res_blend$auc),
  F1 = c(
    f1_score(res_rf_thr$cm),
    f1_score(res_xgb_thr$cm),
    f1_score(res_blend$cm)
  )
) %>%
  # Arredondamento para melhor leitura
  mutate(across(where(is.numeric), \(x) round(x, 3))) %>%
  arrange(desc(Accuracy), desc(AUC), desc(F1))

############################################################
# 14. COMPARAÇÃO FINAL DOS ALGORITMOS
############################################################
# Comparação direta dos modelos base (sem threshold otimizado)

comparacao <- data.frame(
  Modelo = c("Random Forest", "SVM", "XGBoost"),

  Accuracy = c(
    res_rf$cm$overall["Accuracy"],
    res_svm$cm$overall["Accuracy"],
    res_xgb$cm$overall["Accuracy"]
  ),

  AUC = c(res_rf$auc, res_svm$auc, res_xgb$auc),

  Sensitivity = c(
    res_rf$cm$byClass["Sensitivity"],
    res_svm$cm$byClass["Sensitivity"],
    res_xgb$cm$byClass["Sensitivity"]
  ),

  Specificity = c(
    res_rf$cm$byClass["Specificity"],
    res_svm$cm$byClass["Specificity"],
    res_xgb$cm$byClass["Specificity"]
  ),

  Precision = c(
    res_rf$cm$byClass["Pos Pred Value"],
    res_svm$cm$byClass["Pos Pred Value"],
    res_xgb$cm$byClass["Pos Pred Value"]
  ),

  F1 = c(
    f1_score(res_rf$cm),
    f1_score(res_svm$cm),
    f1_score(res_xgb$cm)
  )
)

# Ajuste final da tabela
comparacao <- comparacao %>%
  mutate(across(-Modelo, \(x) round(as.numeric(x), 3))) %>%
  arrange(desc(AUC), desc(Accuracy))

############################################################
# 15. GRÁFICOS (COMPARAÇÃO + MELHOR MODELO)
############################################################

# Preparação dos dados para gráfico de barras
metricas_plot <- comparacao %>%
  dplyr::select(Modelo, Accuracy, AUC, F1) %>%
  pivot_longer(
    cols = c(Accuracy, AUC, F1),
    names_to = "Metrica",
    values_to = "Valor"
  )

# Gráfico comparativo entre algoritmos (base)
p_base <- ggplot(metricas_plot, aes(x = Modelo, y = Valor, fill = Metrica)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  labs(
    title = "Comparação dos Algoritmos Base (Accuracy, AUC e F1)",
    x = "Algoritmo",
    y = "Score"
  )

# Preparação dos dados para gráfico de barras com threshold otimizado
metricas_plot_threshold <- melhorias %>%
  mutate(Modelo = gsub(" \\(threshold otimizado\\)", "", Modelo)) %>%
  dplyr::select(Modelo, Accuracy, AUC, F1) %>%
  pivot_longer(
    cols = c(Accuracy, AUC, F1),
    names_to = "Metrica",
    values_to = "Valor"
  )

# Gráfico comparativo entre algoritmos (threshold otimizado)
p_threshold <- ggplot(metricas_plot_threshold, aes(x = Modelo, y = Valor, fill = Metrica)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  labs(
    title = "Comparação dos Algoritmos com Threshold Otimizado (Accuracy, AUC e F1)",
    x = "Algoritmo",
    y = "Score"
  )

print(p_base)
print(p_threshold)


############################################################
# Identificação automática do melhor modelo após melhorias
############################################################
best_row <- melhorias[1, , drop = FALSE]
best_model_name <- best_row$Modelo[[1]]
best_threshold <- as.numeric(best_row$Threshold[[1]])
obs_best <- test_best[[target]]

# Seleção da probabilidade correta do melhor modelo
if (grepl("^RF", best_model_name)) {
  best_prob <- prob_rf_test
} else if (grepl("^XGBoost", best_model_name)) {
  best_prob <- prob_xgb_test
} else {
  best_prob <- prob_blend_test
}

# Predição final usando threshold otimizado
best_pred <- factor(
  ifelse(best_prob > best_threshold, "Yes", "No"),
  levels = c("No", "Yes")
)

# Matriz de confusão e ROC do melhor modelo
cm_best <- confusionMatrix(best_pred, obs_best, positive = "Yes")
roc_best <- roc(obs_best, best_prob, quiet = TRUE)

############################################################
# Curva ROC do melhor modelo
############################################################
roc_df <- data.frame(
  FPR = 1 - roc_best$specificities,
  TPR = roc_best$sensitivities
)

ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "firebrick", linewidth = 1.1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  theme_minimal() +
  labs(
    title = paste0("ROC - Melhor Modelo: ", best_model_name),
    subtitle = paste0("AUC = ", round(as.numeric(auc(roc_best)), 3)),
    x = "1 - Especificidade (FPR)",
    y = "Sensibilidade (TPR)"
  )

############################################################
# Heatmap da matriz de confusão
############################################################
cm_df <- as.data.frame(cm_best$table)
colnames(cm_df) <- c("Real", "Predito", "Freq")

ggplot(cm_df, aes(x = Predito, y = Real, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "#5DA5DA", high = "#0B3C5D") +
  theme_minimal() +
  labs(
    title = paste0("Matriz de Confusão - ", best_model_name),
    subtitle = paste0("Threshold = ", round(best_threshold, 3)),
    x = "Classe Predita",
    y = "Classe Real"
  )

############################################################
# Distribuição das probabilidades previstas
############################################################
prob_df <- data.frame(Prob = best_prob, Classe = obs_best)

ggplot(prob_df, aes(x = Prob, fill = Classe)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 20) +
  geom_vline(xintercept = best_threshold, linetype = "dashed", color = "red", linewidth = 1) +
  theme_minimal() +
  labs(
    title = paste0("Distribuição de Probabilidades - ", best_model_name),
    subtitle = "Linha tracejada = threshold otimizado",
    x = "Probabilidade prevista para classe Yes",
    y = "Contagem"
  )

############################################################
# 15.1 IMPACTO DO LIMIAR DE CLASSIFICAÇÃO
############################################################
threshold_grid <- seq(0.05, 0.95, by = 0.01)

threshold_impact <- lapply(threshold_grid, function(thr) {
  pred_thr <- factor(
    ifelse(best_prob > thr, "Yes", "No"),
    levels = c("No", "Yes")
  )

  cm_thr <- confusionMatrix(pred_thr, obs_best, positive = "Yes")

  data.frame(
    Threshold = thr,
    Accuracy = as.numeric(cm_thr$overall["Accuracy"]),
    Sensitivity = as.numeric(cm_thr$byClass["Sensitivity"]),
    Specificity = as.numeric(cm_thr$byClass["Specificity"]),
    Precision = as.numeric(cm_thr$byClass["Pos Pred Value"]),
    F1 = f1_score(cm_thr)
  )
}) %>%
  bind_rows()

threshold_impact_long <- threshold_impact %>%
  dplyr::select(Threshold, Accuracy, Sensitivity, Specificity, F1) %>%
  pivot_longer(
    cols = c(Accuracy, Sensitivity, Specificity, F1),
    names_to = "Metrica",
    values_to = "Valor"
  )

ggplot(threshold_impact_long, aes(x = Threshold, y = Valor, color = Metrica)) +
  geom_line(linewidth = 1) +
  geom_vline(
    xintercept = best_threshold,
    linetype = "dashed",
    color = "black",
    linewidth = 0.8
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  labs(
    title = paste0("Impacto do Limiar de Classificação - ", best_model_name),
    subtitle = paste0("Linha tracejada = limiar otimizado (", round(best_threshold, 3), ")"),
    x = "Limiar de classificação",
    y = "Score",
    color = "Métrica"
  )

############################################################
# 15.2 DISTRIBUIÇÃO DAS VARIÁVEIS SELECIONADAS
############################################################
if (length(selected_vars) > 0) {
  imp_df <- varImp(model_rf, scale = FALSE)$importance %>%
    tibble::rownames_to_column("Variavel")

  imp_col <- if ("Overall" %in% names(imp_df)) {
    "Overall"
  } else {
    setdiff(names(imp_df), "Variavel")[1]
  }

  imp_df <- imp_df %>%
    dplyr::select(Variavel, Importancia = all_of(imp_col)) %>%
    filter(is.finite(Importancia)) %>%
    arrange(desc(Importancia)) %>%
    slice_head(n = 15)

  ggplot(imp_df, aes(x = reorder(Variavel, Importancia), y = Importancia)) +
    geom_col(fill = "#33A02C", width = 0.85) +
    coord_flip() +
    theme_minimal() +
    labs(
      title = "Importância das Variáveis Selecionadas - Random Forest",
      subtitle = "Top 15 por importância média",
      x = "Variável",
      y = "Importância"
    )
} else {
  cat("Aviso: não há variáveis selecionadas para o gráfico de importância.\n")
}

############################################################
# 16. RESULTADO FINAL
############################################################

cat("COMPARAÇÃO FINAL DOS ALGORITMOS\n")
cat("=========================================\n")
print(comparacao)

cat("MELHORIAS (THRESHOLD OTIMIZADO)\n")
cat("=========================================\n")
print(melhorias)
