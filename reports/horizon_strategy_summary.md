# DemandWise — Comparação por horizonte

As três estratégias foram avaliadas nos mesmos folds do backtesting:

- Random Forest com previsão recursiva;
- média histórica fixa por loja-produto;
- sazonal ingênuo usando o mesmo dia do ano anterior.

| Horizonte | Estratégia | MAE | RMSE | SMAPE |
| --- | --- | ---: | ---: | ---: |
| 01-07 | Random Forest recursivo | 6.641 | 8.544 | 11.75% |
| 01-07 | Sazonal ingênuo — mesmo dia do ano anterior | 12.225 | 16.305 | 20.80% |
| 01-07 | Média histórica loja-produto | 15.461 | 19.646 | 25.48% |
| 08-28 | Random Forest recursivo | 6.657 | 8.580 | 11.73% |
| 08-28 | Sazonal ingênuo — mesmo dia do ano anterior | 12.402 | 16.479 | 21.06% |
| 08-28 | Média histórica loja-produto | 15.592 | 19.878 | 25.63% |
| 29-60 | Random Forest recursivo | 7.257 | 9.345 | 12.55% |
| 29-60 | Sazonal ingênuo — mesmo dia do ano anterior | 12.164 | 16.138 | 20.73% |
| 29-60 | Média histórica loja-produto | 15.225 | 19.647 | 25.24% |
| 61-90 | Random Forest recursivo | 8.593 | 10.721 | 16.78% |
| 61-90 | Sazonal ingênuo — mesmo dia do ano anterior | 11.675 | 15.514 | 21.93% |
| 61-90 | Média histórica loja-produto | 14.215 | 18.242 | 25.36% |

Melhor estratégia por faixa: **01-07: Random Forest recursivo, 08-28: Random Forest recursivo, 29-60: Random Forest recursivo, 61-90: Random Forest recursivo**.

A decomposição evidencia como o erro evolui quando previsões anteriores passam
a alimentar os lags. Ela também evita concluir sobre um horizonte inteiro a
partir de uma única métrica agregada.
