# Predição de Doença da Vesícula Biliar

Este repositório contém o pipeline de machine learning em R para classificação de `Gallstone.Status`, com comparação entre modelos base e versões com threshold otimizado.

## Métricas finais (modelos base)

| Modelo | Accuracy | AUC | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.873 | 0.951 | 0.906 | 0.839 | 0.853 | 0.879 |
| XGBoost | 0.857 | 0.927 | 0.906 | 0.806 | 0.829 | 0.866 |
| SVM | 0.730 | 0.802 | 0.812 | 0.645 | 0.703 | 0.754 |

## Melhorias (threshold otimizado)

| Modelo | Threshold | Accuracy | Sensitivity | Specificity | AUC | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Blend RF+XGB (threshold otimizado) | 0.439 | 0.889 | 0.906 | 0.871 | 0.938 | 0.892 |
| RF (threshold otimizado) | 0.477 | 0.873 | 0.906 | 0.839 | 0.951 | 0.879 |
| XGBoost (threshold otimizado) | 0.405 | 0.873 | 0.906 | 0.839 | 0.927 | 0.879 |

## Gráficos

### Comparação de algoritmos (base)
![Comparação dos algoritmos base](<img/Grafico - Comparacao Algoritmos Base.png>)

### Comparação de algoritmos (threshold otimizado)
![Comparação dos algoritmos com threshold](<img/Grafico - Comparacao Algoritmos com Threshold.png>)

### ROC do melhor modelo
![ROC melhor modelo](<img/Grafico - ROC Melhor Modelo.png>)

### Matriz de confusão do melhor modelo
![Matriz de confusão](<img/Grafico - Matriz de Confusao.png>)

### Distribuição de probabilidades
![Distribuição de probabilidades](<img/Grafico - Distribuicao De Probabilidades.png>)

### Impacto do limiar de classificação
![Impacto do limiar de classificação](<img/Grafico - Impacto Do Limiar De Classificação.png>)

### Importância das variáveis
![Importância das variáveis](<img/Grafico - Importancia das Variaveis Selecionadas.png>)
