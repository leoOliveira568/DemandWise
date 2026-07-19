# DemandWise — Modelos supervisionados

## Configuração

- Features de treino: 19
- Observações usadas: 300,000
- Janela de treino efetiva: 11/02/2016 a 02/10/2017
- Validação: 03/10/2017 a 31/12/2017
- Horizonte: 90 dias
- Estratégia: previsão recursiva por dia e por série loja-produto

Os lags, médias móveis e médias históricas da validação são atualizados apenas
com previsões anteriores. Nenhum valor real do horizonte futuro é usado.

## Ranking geral

| Tipo | Modelo | MAE | RMSE | MAPE | SMAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| Supervisionado | Random Forest | 7.801 | 9.991 | 18.32% | 16.18% |
| Supervisionado | Gradient Boosting | 7.976 | 10.139 | 19.14% | 16.74% |
| Supervisionado | HistGradient Boosting | 8.476 | 10.858 | 20.38% | 17.46% |
| Baseline | Média histórica loja-produto | 10.158 | 13.514 | 20.49% | 19.76% |
| Baseline | Média dos últimos 7 dias | 11.494 | 15.089 | 26.07% | 21.87% |
| Baseline | Média dos últimos 28 dias | 11.754 | 15.413 | 26.94% | 22.25% |
| Baseline | Média global histórica | 22.968 | 28.557 | 59.85% | 44.17% |

## Resultado

O melhor modelo supervisionado foi **Random Forest**, com MAE de
**7.801** e SMAPE de **16.18%**. Ele superou o melhor baseline em **23.20%** no MAE.

## Tempo de treinamento

| Modelo | Linhas | Tempo |
| --- | ---: | ---: |
| Random Forest | 300,000 | 70.8 s |
| Gradient Boosting | 300,000 | 135.7 s |
| HistGradient Boosting | 300,000 | 13.1 s |

O modelo escolhido para a previsão final deverá considerar desempenho,
estabilidade recursiva e custo computacional.
