# DemandWise — Otimização temporal do Random Forest

- Treino: 09/08/2016 a 03/08/2017
- Holdout: 04/08/2017 a 02/10/2017
- Horizonte recursivo: 60 dias

| Configuração | Árvores | Profundidade | Folha mínima | Features | MAE | RMSE | SMAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| responsive | 70 | 16 | 5 | 0.7 | 8.280 | 10.443 | 14.52% |
| current | 60 | 14 | 10 | 0.8 | 8.457 | 10.657 | 14.79% |
| regularized | 70 | 12 | 15 | 0.8 | 8.753 | 10.973 | 15.26% |

A configuração selecionada foi **responsive**, com MAE de
**8.280** no holdout anterior à validação oficial.
