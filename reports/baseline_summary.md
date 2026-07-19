# DemandWise — Avaliação dos baselines

## Estratégia de validação

- Treino: 01/01/2013 a 02/10/2017
- Validação: 03/10/2017 a 31/12/2017
- Horizonte: 90 dias
- Observações de validação: 45,000
- Origem das previsões: fechamento de 02/10/2017

Todas as previsões são calculadas exclusivamente com o período de treino e
permanecem fixas durante os 90 dias de validação.

## Resultados

| Baseline | MAE | RMSE | MAPE | SMAPE |
| --- | ---: | ---: | ---: | ---: |
| Média histórica loja-produto | 10.158 | 13.514 | 20.49% | 19.76% |
| Média dos últimos 7 dias | 11.494 | 15.089 | 26.07% | 21.87% |
| Média dos últimos 28 dias | 11.754 | 15.413 | 26.94% | 22.25% |
| Média global histórica | 22.968 | 28.557 | 59.85% | 44.17% |

## Melhor baseline

**Média histórica loja-produto**, com MAE de **10.158** unidades e SMAPE de
**19.76%**.

Esses resultados formam o patamar mínimo que os modelos de machine learning
deverão superar usando o mesmo corte temporal.
