# DemandWise — Backtesting temporal

Foram avaliadas três janelas consecutivas e não sobrepostas de 90 dias. Em
cada fold, o modelo foi treinado somente com dados anteriores à validação e
previu o horizonte recursivamente.

| Fold | Início | Fim | MAE | RMSE | MAPE | SMAPE |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 06/04/2017 | 04/07/2017 | 6.732 | 8.744 | 11.55% | 11.39% |
| 2 | 05/07/2017 | 02/10/2017 | 8.217 | 10.404 | 15.49% | 13.87% |
| 3 | 03/10/2017 | 31/12/2017 | 7.801 | 9.991 | 18.32% | 16.18% |

- MAE médio: **7.583**
- RMSE médio: **9.713**
- SMAPE médio: **13.81%**
- Variação relativa do MAE entre folds: **10.10%**
- Parâmetros aplicados: `{"n_estimators": 70, "max_depth": 16, "min_samples_leaf": 5, "max_features": 0.7}`

O resultado médio oferece uma visão mais robusta que um único corte temporal e
serve de base para calibrar os intervalos de previsão.
