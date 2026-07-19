# DemandWise Dashboard

Dashboard executivo do projeto DemandWise, construído para apresentar histórico, sazonalidade, desempenho dos modelos, previsão futura e recomendações de estoque.

## Funções interativas

- filtros combináveis por loja, produto, mês e cenário de incerteza;
- indicadores e curva diária recalculados em tempo real;
- ranking dos cinco dias de maior pressão;
- exportação CSV do recorte filtrado;
- simulador de estoque com lead time, revisão e nível de serviço ajustáveis;
- cálculo de estoque de segurança, ponto de reposição e posição-alvo;
- carregamento sob demanda das 45 mil previsões granulares.

## Atualizar os dados

A partir da raiz do projeto:

```powershell
python src/export_dashboard_data.py
```

## Executar localmente

```powershell
cd dashboard
npm install
npm run dev
```

## Validar

```powershell
npm run build
npm test
```
