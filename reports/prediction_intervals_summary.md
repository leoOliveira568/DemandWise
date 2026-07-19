# DemandWise — Intervalos de previsão

Os intervalos de 90% usam calibração conformal por faixa de horizonte e volume.
O primeiro fold define as margens, o segundo adiciona um buffer contra mudança
temporal e o terceiro funciona como teste fora da calibração.

- Cobertura-alvo: **90%**
- Cobertura observada no fold mais recente: **93.65%**
- Fator de ajuste temporal: **1.066 a 1.482**, condicionado por horizonte e volume
- Largura média observada: **34.15 unidades**
- Largura média no horizonte futuro: **36.18 unidades**

| Horizonte | Volume | Observações | Margem absoluta |
| --- | --- | ---: | ---: |
| 01-07 | high | 3,405 | 20.60 |
| 01-07 | low | 3,627 | 11.11 |
| 01-07 | medium | 3,468 | 14.37 |
| 08-28 | high | 10,438 | 21.49 |
| 08-28 | low | 10,634 | 11.77 |
| 08-28 | medium | 10,428 | 15.38 |
| 29-60 | high | 16,601 | 24.89 |
| 29-60 | low | 15,476 | 13.38 |
| 29-60 | medium | 15,923 | 20.24 |
| 61-90 | high | 14,556 | 27.90 |
| 61-90 | low | 15,263 | 17.58 |
| 61-90 | medium | 15,181 | 27.04 |

Os limites representam incerteza estatística estimada pelos resíduos históricos.
Eles não incorporam choques externos ausentes no dataset, como promoções,
rupturas, mudanças de preço ou eventos não observados.
