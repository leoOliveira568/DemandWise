# DemandWise — Relatório Executivo

## 1. Resumo executivo

O DemandWise foi desenvolvido para responder como uma rede varejista pode prever vendas futuras por loja e produto e transformar essa previsão em decisões de estoque e planejamento.

O projeto analisou 913.000 registros diários entre 2013 e 2017, abrangendo 10 lojas, 50 produtos e 500 séries loja-produto. Após comparação com quatro baselines e três modelos supervisionados, o Random Forest apresentou o melhor desempenho na validação temporal de 90 dias.

O modelo final prevê aproximadamente **2.190.265 unidades** entre 01/01/2018 e 31/03/2018, com média de **24.336 unidades por dia**. Março concentra a maior demanda mensal projetada e requer antecipação de compras e capacidade operacional.

## 2. Problema de negócio

Sem uma previsão estruturada, o planejamento de estoque tende a reagir ao histórico recente e pode produzir dois custos simultâneos:

- ruptura e perda de vendas em períodos de pico;
- excesso de mercadoria, capital imobilizado e risco de obsolescência em períodos de menor demanda.

O objetivo do DemandWise é criar um sinal quantitativo por loja e produto que apoie compras, reposição, alocação e definição de estoques de segurança.

## 3. Dados e abordagem

Fonte: Kaggle, competição *Store Item Demand Forecasting Challenge*.

| Dimensão | Cobertura |
| --- | ---: |
| Registros de treino | 913.000 |
| Período histórico | 01/01/2013 a 31/12/2017 |
| Lojas | 10 |
| Produtos | 50 |
| Séries loja-produto | 500 |
| Horizonte futuro | 90 dias |

As features incluem calendário, defasagens de 7, 14 e 28 dias, médias móveis, dispersão móvel e médias históricas por loja, produto e mês. Todas as features históricas são fechadas na data anterior, evitando vazamento de dados.

## 4. Principais padrões de demanda

### Crescimento

As vendas anuais cresceram **35,2%** entre 2013 e 2017. Esse aumento de nível indica que parâmetros fixos de estoque baseados em anos antigos tendem a subestimar a demanda atual.

### Sazonalidade anual

Julho apresentou a maior média diária histórica. A amplitude entre meses fortes e fracos justifica políticas de cobertura variáveis ao longo do ano, em vez de um único estoque de segurança permanente.

### Sazonalidade semanal

Domingo apresentou a maior média diária de vendas. Fins de semana ficaram aproximadamente **23,3% acima dos dias úteis**, sinalizando necessidade de cobertura adicional antes de sábado e domingo.

### Concentração por loja e produto

A Loja 2 liderou o volume histórico e também a demanda prevista. O Produto 15 foi o líder histórico, enquanto o Produto 28 aparece como o maior volume previsto no horizonte futuro. Essa diferença reforça que o ranking precisa ser atualizado conforme o horizonte e não tratado como estático.

## 5. Validação e escolha do modelo

Os últimos 90 dias de 2017 foram reservados integralmente para validação, reproduzindo a duração do `test.csv`. A previsão foi recursiva: cada novo dia usou apenas histórico anterior e previsões já geradas.

| Modelo | MAE | RMSE | MAPE | SMAPE |
| --- | ---: | ---: | ---: | ---: |
| **Random Forest otimizado** | **7,697** | **9,866** | **18,08%** | **15,98%** |
| Gradient Boosting | 7,976 | 10,139 | 19,14% | 16,74% |
| HistGradient Boosting | 8,476 | 10,858 | 20,38% | 17,46% |
| Melhor baseline | 10,158 | 13,514 | 20,49% | 19,76% |

O Random Forest reduziu o MAE em **24,22%** em relação ao melhor baseline. Os hiperparâmetros foram escolhidos em um holdout temporal anterior e a janela de 220 mil observações superou as alternativas de 180 mil e 300 mil linhas.

O backtesting em três janelas consecutivas de 90 dias produziu MAEs de 6,730, 8,116 e 7,697, com média de **7,514**. Esse resultado reduz a dependência de um único corte temporal. Os intervalos conformais de 90% atingiram **93,65% de cobertura** no fold final, que permaneceu fora da calibração.

## 6. Previsão futura

| Mês | Demanda prevista |
| --- | ---: |
| Janeiro de 2018 | 677.915 |
| Fevereiro de 2018 | 673.918 |
| Março de 2018 | 838.432 |
| **Total** | **2.190.265** |

O maior pico diário está previsto para **25/03/2018**, com aproximadamente **32.203 unidades**. Março representa cerca de 38% do horizonte e fica aproximadamente 21,9% acima de janeiro.

## 7. Recomendações para estoque e planejamento

### 1. Antecipar a cobertura de março

Revisar pedidos, capacidade de recebimento e alocação antes do início de março. O aumento projetado não deve ser tratado apenas no momento em que aparecer nas vendas realizadas.

### 2. Proteger a operação de fim de semana

Programar reposições e checagens de ruptura antes de sábado. Séries de alto giro devem ter gatilhos de acompanhamento mais frequentes entre sexta e domingo.

### 3. Priorizar Loja 2 e Produto 28

Usar esses grupos como primeira fila de revisão de estoque de segurança e abastecimento. A decisão final deve considerar margem, lead time, espaço, validade e nível de serviço.

### 4. Segmentar políticas de estoque

Evitar uma política única para todas as combinações. Lojas, produtos e períodos apresentam níveis e sazonalidades diferentes; os parâmetros devem acompanhar essa heterogeneidade.

### 5. Monitorar erro e desvio em produção

Comparar previsão e realizado semanalmente. Um aumento persistente de MAE ou SMAPE deve acionar revisão de features, janela de treino e hiperparâmetros.

## 8. Limitações

- O dataset não contém preço, promoção, feriados, estoque disponível, ruptura observada ou lead time.
- As previsões são de vendas observadas, que podem ser menores que a demanda real quando ocorre falta de produto.
- O modelo não estima diretamente custo, margem ou impacto financeiro.
- O horizonte futuro não possui valores reais no projeto; portanto, sua qualidade é estimada pela validação temporal.
- Mudanças estruturais após 2017 podem exigir retreinamento e novas variáveis externas.

## 9. Próximas evoluções recomendadas

1. incorporar calendário de feriados e eventos comerciais;
2. adicionar preço, promoções, estoque, ruptura e lead time;
3. implementar intervalos de previsão para quantificar incerteza;
4. definir políticas de estoque de segurança por nível de serviço;
5. automatizar monitoramento e retreinamento periódico;
6. avaliar modelos específicos de boosting e forecasting hierárquico.

## 10. Conclusão

O DemandWise demonstra que uma abordagem temporal disciplinada supera médias simples e gera um sinal acionável para planejamento. O principal valor do projeto não é apenas prever 45.000 linhas, mas organizar uma cadeia reproduzível entre dados, validação, previsão, priorização e decisão operacional.
