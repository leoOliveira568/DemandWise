# DemandWise — Previsão futura

## Configuração

- Modelo: Random Forest
- Treino efetivo: 18/10/2016 a 31/12/2017
- Observações de treino: 220,000
- Features: 19
- Tempo de retreinamento: 35.1 segundos
- Horizonte previsto: 01/01/2018 a 31/03/2018
- Estratégia: previsão recursiva para 500 combinações loja-produto

## Resumo das previsões

- Demanda total prevista: **2,190,265 unidades**
- Média diária prevista: **24,336 unidades**
- Dia de maior demanda: **25/03/2018**, com **32,203 unidades**
- Loja com maior demanda prevista: **Loja 2**
- Produto com maior demanda prevista: **Produto 28**

## Demanda prevista por mês

| Mês | Unidades previstas |
| --- | ---: |
| 2018-01 | 677,915 |
| 2018-02 | 673,918 |
| 2018-03 | 838,432 |

As previsões são estimativas do modelo e devem apoiar decisões de estoque em
conjunto com restrições de capacidade, lead time, nível de serviço e contexto
comercial. O arquivo de entrega foi gerado em
`submissions/demandwise_submission.csv`.
