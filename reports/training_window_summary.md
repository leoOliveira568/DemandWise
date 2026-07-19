# DemandWise — Janela de treinamento

As janelas foram comparadas no holdout de desenvolvimento de
**04/08/2017 a 02/10/2017**,
anterior à validação oficial. Os hiperparâmetros já selecionados foram mantidos.

| Linhas mais recentes | MAE | RMSE | MAPE | SMAPE |
| ---: | ---: | ---: | ---: | ---: |
| 300,000 | 7.414 | 9.412 | 14.76% | 13.24% |
| 220,000 | 7.432 | 9.438 | 14.81% | 13.28% |
| 180,000 | 8.280 | 10.443 | 16.48% | 14.52% |

A janela selecionada foi **300,000 linhas**, com MAE
de **7.414**. O corte é temporal e mantém todas as 500 séries em cada
dia; nenhuma amostragem aleatória é usada. Os últimos 90 dias de 2017 não
participam desta escolha e permanecem independentes para o resultado final.
