# DemandWise — Segmentação e política de estoque

## Segmentação ABC/XYZ

- ABC: contribuição acumulada de vendas nos últimos 365 dias.
- XYZ: tercis do coeficiente de variação das vendas semanais, adequados à
  dispersão observada no dataset.
- Séries classificadas: **500**

| Segmento | Séries | Participação da demanda |
| --- | ---: | ---: |
| AX | 137 | 36.48% |
| AY | 117 | 30.38% |
| AZ | 58 | 13.16% |
| BZ | 60 | 7.15% |
| BY | 41 | 5.23% |
| CZ | 49 | 3.70% |
| BX | 22 | 2.66% |
| CY | 8 | 0.62% |
| CX | 8 | 0.62% |

## Cenário de estoque

- Lead time assumido: **7 dias**
- Período de revisão: **7 dias**
- Nível de serviço: **95%**
- Séries de prioridade alta: **175**

O arquivo `inventory_policy.csv` apresenta estoque de segurança, ponto de
reposição e posição-alvo por loja e produto. Esses valores são parâmetros de
cenário, não ordens de compra. A quantidade a comprar exige saldo disponível,
pedidos em trânsito, lote mínimo, lead time real e restrições comerciais, que
não existem no dataset.
