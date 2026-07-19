"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import dashboardData from "./dashboard-data.json";

type Point = { date: string; sales: number; prediction_random_forest?: number };
type Metric = "MAE" | "RMSE" | "MAPE" | "SMAPE";
type ForecastScenario = "lower" | "base" | "upper";
type ForecastRow = [number, number, number, number, number, number];
type InventoryPolicy = [number, number, string, number, number];
type DashboardData = {
  overview: {
    historicalSales: number;
    historicalGrowthPct: number;
    forecastSales: number;
    forecastDailyAverage: number;
    forecastStart: string;
    forecastEnd: string;
    bestModel: string;
    bestMae: number;
    bestSmape: number;
    improvementPct: number;
    peakDate: string;
    peakSales: number;
    weekendUpliftPct: number;
  };
  monthlyHistory: Point[];
  yearlyHistory: { year: number; sales: number }[];
  weekdayProfile: { label: string; sales: number }[];
  modelMetrics: Array<{
    rank: number;
    model_type: string;
    model: string;
    MAE: number;
    RMSE: number;
    MAPE: number;
    SMAPE: number;
  }>;
  validationDaily: Point[];
  futureDaily: Point[];
  futureMonthly: { month: string; sales: number }[];
  futureByStore: { store: number; sales: number }[];
  futureTopItems: { item: number; sales: number }[];
  featureImportance: { feature: string; importance: number }[];
};
type ForecastData = {
  forecastDates: string[];
  forecastRows: ForecastRow[];
  inventoryPolicies: InventoryPolicy[];
};

const data = dashboardData as unknown as DashboardData;

const compactNumber = new Intl.NumberFormat("pt-BR", {
  notation: "compact",
  maximumFractionDigits: 1,
});

const integerNumber = new Intl.NumberFormat("pt-BR", {
  maximumFractionDigits: 0,
});

const shortDate = new Intl.DateTimeFormat("pt-BR", {
  day: "2-digit",
  month: "short",
});

function parseDate(value: string) {
  return new Date(value.length === 10 ? `${value}T00:00:00` : value);
}

function formatDate(value: string) {
  return shortDate.format(parseDate(value)).replace(".", "");
}

function LineChart({
  points,
  series,
  ariaLabel,
}: {
  points: Point[];
  series: Array<{ key: "sales" | "prediction_random_forest"; label: string; color: string }>;
  ariaLabel: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !points.length) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const draw = () => {
      const width = Math.max(parent.clientWidth, 300);
      const height = 300;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      const context = canvas.getContext("2d");
      if (!context) return;
      context.scale(ratio, ratio);
      context.clearRect(0, 0, width, height);

      const style = getComputedStyle(document.documentElement);
      const grid = style.getPropertyValue("--line").trim() || "#dbe4ee";
      const muted = style.getPropertyValue("--muted").trim() || "#667085";
      const left = 54;
      const right = 18;
      const top = 14;
      const bottom = 36;
      const plotWidth = width - left - right;
      const plotHeight = height - top - bottom;
      const values = series.flatMap(({ key }) =>
        points.map((point) => Number(point[key] ?? 0)),
      );
      const maximum = Math.max(...values) * 1.08;

      context.font = "12px Arial, sans-serif";
      context.textBaseline = "middle";
      for (let tick = 0; tick <= 4; tick += 1) {
        const y = top + (plotHeight * tick) / 4;
        const value = maximum * (1 - tick / 4);
        context.strokeStyle = grid;
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(left, y);
        context.lineTo(width - right, y);
        context.stroke();
        context.fillStyle = muted;
        context.textAlign = "right";
        context.fillText(compactNumber.format(value), left - 9, y);
      }

      [0, Math.floor((points.length - 1) / 2), points.length - 1].forEach((index) => {
        const x = left + (plotWidth * index) / Math.max(points.length - 1, 1);
        context.fillStyle = muted;
        context.textAlign = index === 0 ? "left" : index === points.length - 1 ? "right" : "center";
        context.fillText(formatDate(points[index].date), x, height - 12);
      });

      series.forEach(({ key, color }) => {
        context.strokeStyle = color;
        context.lineWidth = 2.5;
        context.lineJoin = "round";
        context.lineCap = "round";
        context.beginPath();
        points.forEach((point, index) => {
          const x = left + (plotWidth * index) / Math.max(points.length - 1, 1);
          const value = Number(point[key] ?? 0);
          const y = top + plotHeight - (value / maximum) * plotHeight;
          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        context.stroke();
      });
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(parent);
    return () => observer.disconnect();
  }, [points, series]);

  return (
    <div className="line-chart">
      <div className="chart-legend" aria-hidden="true">
        {series.map((item) => (
          <span key={item.key}>
            <i style={{ background: item.color }} />
            {item.label}
          </span>
        ))}
      </div>
      <canvas ref={canvasRef} role="img" aria-label={ariaLabel} />
    </div>
  );
}

function RankedBars({
  rows,
  labelKey,
  prefix,
}: {
  rows: Array<Record<string, number>>;
  labelKey: string;
  prefix: string;
}) {
  const maximum = Math.max(...rows.map((row) => row.sales));
  return (
    <div className="ranked-bars">
      {rows.map((row, index) => (
        <div className="ranked-row" key={`${prefix}-${row[labelKey]}`}>
          <span className="rank-number">{String(index + 1).padStart(2, "0")}</span>
          <span className="rank-label">{prefix} {row[labelKey]}</span>
          <div className="bar-track" aria-hidden="true">
            <div className="bar-fill" style={{ width: `${(row.sales / maximum) * 100}%` }} />
          </div>
          <strong>{compactNumber.format(row.sales)}</strong>
        </div>
      ))}
    </div>
  );
}

function ModelComparison() {
  const [metric, setMetric] = useState<Metric>("MAE");
  const maximum = Math.max(...data.modelMetrics.map((model) => model[metric]));
  const suffix = metric === "MAPE" || metric === "SMAPE" ? "%" : "";

  return (
    <div>
      <div className="metric-switch" aria-label="Escolher métrica">
        {(["MAE", "RMSE", "MAPE", "SMAPE"] as Metric[]).map((item) => (
          <button
            type="button"
            key={item}
            className={metric === item ? "active" : ""}
            aria-pressed={metric === item}
            onClick={() => setMetric(item)}
          >
            {item}
          </button>
        ))}
      </div>
      <div className="model-bars">
        {data.modelMetrics.map((model) => (
          <div className="model-row" key={model.model}>
            <div className="model-label">
              <span>{model.model}</span>
              <small>{model.model_type}</small>
            </div>
            <div className="model-track" aria-hidden="true">
              <div
                className={model.model_type === "Supervisionado" ? "model-fill supervised" : "model-fill baseline"}
                style={{ width: `${Math.max((model[metric] / maximum) * 100, 2)}%` }}
              />
            </div>
            <strong>{model[metric].toFixed(metric === "MAE" || metric === "RMSE" ? 2 : 1)}{suffix}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

const scenarioDefinitions: Record<ForecastScenario, { label: string; index: 3 | 4 | 5; color: string }> = {
  lower: { label: "Cenário inferior (90%)", index: 4, color: "#6f8f8d" },
  base: { label: "Previsão central", index: 3, color: "#0f6b68" },
  upper: { label: "Cenário superior (90%)", index: 5, color: "#d97706" },
};

const serviceZScores: Record<string, number> = {
  "90": 1.282,
  "95": 1.645,
  "98": 2.054,
};

function ForecastExplorer() {
  const [forecastData, setForecastData] = useState<ForecastData | null>(null);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    let active = true;
    fetch("/forecast-data.json")
      .then((response) => {
        if (!response.ok) throw new Error("Falha ao carregar previsões");
        return response.json() as Promise<ForecastData>;
      })
      .then((payload) => {
        if (active) setForecastData(payload);
      })
      .catch(() => {
        if (active) setLoadError(true);
      });
    return () => { active = false; };
  }, []);

  if (loadError) {
    return <div className="explorer-loading error" role="alert">Não foi possível carregar o explorador. Atualize a página para tentar novamente.</div>;
  }
  if (!forecastData) {
    return <div className="explorer-loading" role="status">Carregando 45.000 previsões para análise interativa…</div>;
  }
  return <ForecastExplorerLoaded forecastData={forecastData} />;
}

function ForecastExplorerLoaded({ forecastData }: { forecastData: ForecastData }) {
  const [selectedStore, setSelectedStore] = useState("all");
  const [selectedItem, setSelectedItem] = useState("all");
  const [selectedMonth, setSelectedMonth] = useState("all");
  const [scenario, setScenario] = useState<ForecastScenario>("base");
  const [leadTime, setLeadTime] = useState(7);
  const [reviewPeriod, setReviewPeriod] = useState(7);
  const [serviceLevel, setServiceLevel] = useState("95");

  const filteredRows = useMemo(() => {
    return forecastData.forecastRows.filter((row) => {
      const date = forecastData.forecastDates[row[0]];
      return (
        (selectedStore === "all" || row[1] === Number(selectedStore)) &&
        (selectedItem === "all" || row[2] === Number(selectedItem)) &&
        (selectedMonth === "all" || date.startsWith(selectedMonth))
      );
    });
  }, [forecastData, selectedStore, selectedItem, selectedMonth]);

  const summary = useMemo(() => {
    const definition = scenarioDefinitions[scenario];
    const daily = new Map<number, { sales: number; lower: number; base: number; upper: number }>();
    let lowerTotal = 0;
    let baseTotal = 0;
    let upperTotal = 0;
    const series = new Set<string>();

    for (const row of filteredRows) {
      const current = daily.get(row[0]) ?? { sales: 0, lower: 0, base: 0, upper: 0 };
      current.sales += row[definition.index];
      current.base += row[3];
      current.lower += row[4];
      current.upper += row[5];
      daily.set(row[0], current);
      lowerTotal += row[4];
      baseTotal += row[3];
      upperTotal += row[5];
      series.add(`${row[1]}-${row[2]}`);
    }

    const points = [...daily.entries()]
      .sort(([left], [right]) => left - right)
      .map(([dateIndex, value]) => ({ date: forecastData.forecastDates[dateIndex], sales: value.sales }));
    const peak = points.reduce(
      (best, point) => (point.sales > best.sales ? point : best),
      points[0] ?? { date: forecastData.forecastDates[0], sales: 0 },
    );
    return {
      points,
      total: points.reduce((sum, point) => sum + point.sales, 0),
      average: points.length ? points.reduce((sum, point) => sum + point.sales, 0) / points.length : 0,
      peak,
      lowerTotal,
      baseTotal,
      upperTotal,
      seriesCount: series.size,
      selectedSeries: series,
    };
  }, [forecastData, filteredRows, scenario]);

  const inventoryScenario = useMemo(() => {
    const policies = forecastData.inventoryPolicies.filter((policy) =>
      summary.selectedSeries.has(`${policy[0]}-${policy[1]}`),
    );
    const aggregateResidualStd = Math.sqrt(
      policies.reduce((sum, policy) => sum + policy[3] ** 2, 0),
    );
    const zScore = serviceZScores[serviceLevel];
    const safetyStock = Math.ceil(zScore * aggregateResidualStd * Math.sqrt(leadTime));
    const reorderPoint = Math.ceil(summary.average * leadTime + safetyStock);
    const targetPosition = Math.ceil(
      summary.average * (leadTime + reviewPeriod) + safetyStock,
    );
    return {
      safetyStock,
      reorderPoint,
      targetPosition,
      coverageDays: summary.average ? targetPosition / summary.average : 0,
    };
  }, [forecastData, summary, serviceLevel, leadTime, reviewPeriod]);

  const topDays = useMemo(
    () => [...summary.points].sort((a, b) => b.sales - a.sales).slice(0, 5),
    [summary.points],
  );

  const chartSeries = useMemo(
    () => [{
      key: "sales" as const,
      label: scenarioDefinitions[scenario].label,
      color: scenarioDefinitions[scenario].color,
    }],
    [scenario],
  );

  const resetFilters = () => {
    setSelectedStore("all");
    setSelectedItem("all");
    setSelectedMonth("all");
    setScenario("base");
  };

  const downloadCsv = () => {
    const definition = scenarioDefinitions[scenario];
    const header = "date,store,item,scenario,forecast_base,lower_90,upper_90,selected_value";
    const lines = filteredRows.map((row) => [
      forecastData.forecastDates[row[0]], row[1], row[2], scenario,
      row[3].toFixed(3), row[4].toFixed(3), row[5].toFixed(3), row[definition.index].toFixed(3),
    ].join(","));
    const blob = new Blob([[header, ...lines].join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `demandwise-${selectedStore}-${selectedItem}-${selectedMonth}-${scenario}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="forecast-explorer">
      <div className="explorer-toolbar">
        <div className="filter-grid">
          <label>
            <span>Loja</span>
            <select value={selectedStore} onChange={(event) => setSelectedStore(event.target.value)}>
              <option value="all">Todas as lojas</option>
              {Array.from({ length: 10 }, (_, index) => index + 1).map((store) => (
                <option value={store} key={store}>Loja {store}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Produto</span>
            <select value={selectedItem} onChange={(event) => setSelectedItem(event.target.value)}>
              <option value="all">Todos os produtos</option>
              {Array.from({ length: 50 }, (_, index) => index + 1).map((item) => (
                <option value={item} key={item}>Produto {item}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Período</span>
            <select value={selectedMonth} onChange={(event) => setSelectedMonth(event.target.value)}>
              <option value="all">Jan — Mar 2018</option>
              <option value="2018-01">Janeiro</option>
              <option value="2018-02">Fevereiro</option>
              <option value="2018-03">Março</option>
            </select>
          </label>
          <label>
            <span>Cenário</span>
            <select value={scenario} onChange={(event) => setScenario(event.target.value as ForecastScenario)}>
              {Object.entries(scenarioDefinitions).map(([key, definition]) => (
                <option value={key} key={key}>{definition.label}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="toolbar-actions">
          <button type="button" className="secondary-button" onClick={resetFilters}>Limpar filtros</button>
          <button type="button" className="download-button" onClick={downloadCsv}>Exportar CSV</button>
        </div>
      </div>

      <div className="active-filter-note" aria-live="polite">
        <span>{summary.seriesCount} séries</span>
        <p>
          {selectedStore === "all" ? "Todas as lojas" : `Loja ${selectedStore}`} · {" "}
          {selectedItem === "all" ? "todos os produtos" : `Produto ${selectedItem}`} · {" "}
          {scenarioDefinitions[scenario].label}
        </p>
      </div>

      <div className="explorer-kpis">
        <article><span>Total do cenário</span><strong>{integerNumber.format(summary.total)}</strong><small>unidades no recorte</small></article>
        <article><span>Média diária</span><strong>{integerNumber.format(summary.average)}</strong><small>unidades por dia</small></article>
        <article><span>Maior pico</span><strong>{compactNumber.format(summary.peak.sales)}</strong><small>{formatDate(summary.peak.date)}</small></article>
        <article><span>Faixa de 90%</span><strong>{compactNumber.format(summary.lowerTotal)}–{compactNumber.format(summary.upperTotal)}</strong><small>limites agregados</small></article>
      </div>

      <div className="explorer-main-grid">
        <article className="chart-panel explorer-chart">
          <div className="panel-heading">
            <div><span>Previsão filtrada</span><strong>Evolução diária do cenário selecionado</strong></div>
            <span className="panel-tag">{summary.points.length} dias</span>
          </div>
          <LineChart points={summary.points} series={chartSeries} ariaLabel="Previsão diária filtrada por loja, produto, período e cenário" />
        </article>
        <aside className="peak-days-panel">
          <p className="micro-label">DIAS DE MAIOR PRESSÃO</p>
          {topDays.map((point, index) => (
            <div className="peak-day-row" key={point.date}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <p>{new Intl.DateTimeFormat("pt-BR", { day: "2-digit", month: "long" }).format(parseDate(point.date))}</p>
              <strong>{integerNumber.format(point.sales)}</strong>
            </div>
          ))}
          <div className="uncertainty-note">
            <span>Incerteza do recorte</span>
            <p>A amplitude total entre os limites é {integerNumber.format(summary.upperTotal - summary.lowerTotal)} unidades.</p>
          </div>
        </aside>
      </div>

      <article className="inventory-simulator" id="simulator">
        <div className="simulator-intro">
          <p className="micro-label">SIMULADOR DE COBERTURA</p>
          <h3>Transforme o cenário em parâmetros de estoque.</h3>
          <p>Use os controles para testar lead time, revisão e nível de serviço sobre o mesmo recorte filtrado.</p>
        </div>
        <div className="simulator-controls">
          <label>
            <span>Lead time <strong>{leadTime} dias</strong></span>
            <input type="range" min="1" max="30" value={leadTime} onChange={(event) => setLeadTime(Number(event.target.value))} />
          </label>
          <label>
            <span>Revisão <strong>{reviewPeriod} dias</strong></span>
            <input type="range" min="0" max="30" value={reviewPeriod} onChange={(event) => setReviewPeriod(Number(event.target.value))} />
          </label>
          <label>
            <span>Nível de serviço</span>
            <select value={serviceLevel} onChange={(event) => setServiceLevel(event.target.value)}>
              <option value="90">90%</option>
              <option value="95">95%</option>
              <option value="98">98%</option>
            </select>
          </label>
        </div>
        <div className="simulator-results">
          <div><span>Estoque de segurança</span><strong>{integerNumber.format(inventoryScenario.safetyStock)}</strong><small>proteção contra erro</small></div>
          <div><span>Ponto de reposição</span><strong>{integerNumber.format(inventoryScenario.reorderPoint)}</strong><small>demanda no lead time + segurança</small></div>
          <div><span>Posição-alvo</span><strong>{integerNumber.format(inventoryScenario.targetPosition)}</strong><small>lead time + revisão</small></div>
          <div><span>Cobertura estimada</span><strong>{inventoryScenario.coverageDays.toFixed(1).replace(".", ",")} d</strong><small>sobre a média filtrada</small></div>
        </div>
        <p className="simulator-disclaimer">Hipótese: erros das séries agregadas são independentes. Os valores não são ordens de compra; saldo disponível, pedidos em trânsito, lote mínimo e lead time real não existem no dataset.</p>
      </article>
    </div>
  );
}

export default function Dashboard() {
  const validationSeries = useMemo(
    () => [
      { key: "sales" as const, label: "Real", color: "#172b4d" },
      { key: "prediction_random_forest" as const, label: "Previsto", color: "#d97706" },
    ],
    [],
  );
  const march = data.futureMonthly.find((month) => month.month === "2018-03")!;
  const january = data.futureMonthly.find((month) => month.month === "2018-01")!;
  const marchIncrease = ((march.sales / january.sales) - 1) * 100;
  const maxWeekday = Math.max(...data.weekdayProfile.map((day) => day.sales));
  const maxImportance = Math.max(...data.featureImportance.map((feature) => feature.importance));

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="DemandWise — início">
          <span className="brand-mark">DW</span>
          <span>
            <strong>DemandWise</strong>
            <small>Retail demand intelligence</small>
          </span>
        </a>
        <nav aria-label="Navegação principal">
          <a href="#project">O projeto</a>
          <a href="#models">Modelos</a>
          <a href="#forecast">Previsão</a>
          <a href="#recommendations">Decisões</a>
        </nav>
        <span className="status-dot"><i /> Modelo validado</span>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">Ciência de Dados · Forecasting · Supply Chain</p>
          <h1>Previsão que vira<br /><em>decisão de estoque.</em></h1>
          <p className="hero-description">
            Um sistema de previsão para 500 combinações de loja e produto, validado em um
            horizonte realista de 90 dias e traduzido em prioridades operacionais.
          </p>
          <div className="hero-actions">
            <a className="primary-action" href="#forecast">Explorar previsão</a>
            <a className="text-action" href="#methodology">Ver metodologia <span>→</span></a>
          </div>
        </div>
        <aside className="hero-brief" aria-label="Resumo da previsão">
          <div className="brief-topline">
            <span>Próximo horizonte</span>
            <strong>JAN — MAR 2018</strong>
          </div>
          <p>Demanda total projetada</p>
          <strong className="hero-number">{compactNumber.format(data.overview.forecastSales)}</strong>
          <span className="hero-unit">unidades</span>
          <div className="brief-callout">
            <span className="callout-index">01</span>
            <p><strong>Março exige antecipação.</strong> A demanda projetada fica {marchIncrease.toFixed(1).replace(".", ",")}% acima de janeiro.</p>
          </div>
          <div className="brief-footer">
            <span>Maior pico</span>
            <strong>{formatDate(data.overview.peakDate)} · {compactNumber.format(data.overview.peakSales)}</strong>
          </div>
        </aside>
      </section>

      <section className="kpi-grid" id="overview" aria-label="Indicadores principais">
        <article className="kpi-card primary-kpi">
          <span>Demanda prevista</span>
          <strong>{compactNumber.format(data.overview.forecastSales)}</strong>
          <small>90 dias · 500 séries</small>
        </article>
        <article className="kpi-card">
          <span>MAE do melhor modelo</span>
          <strong>{data.overview.bestMae.toFixed(2).replace(".", ",")}</strong>
          <small>unidades por previsão</small>
        </article>
        <article className="kpi-card">
          <span>Ganho vs. baseline</span>
          <strong>{data.overview.improvementPct.toFixed(1).replace(".", ",")}%</strong>
          <small>redução no MAE</small>
        </article>
        <article className="kpi-card">
          <span>Pressão de fim de semana</span>
          <strong>+{data.overview.weekendUpliftPct.toFixed(1).replace(".", ",")}%</strong>
          <small>sobre dias úteis</small>
        </article>
      </section>

      <section className="section-shell project-story" id="project">
        <div className="project-story-header">
          <div>
            <p className="section-index">O PROJETO EM 60 SEGUNDOS</p>
            <h2>Da pergunta de negócio<br />à política de estoque.</h2>
          </div>
          <div className="project-summary">
            <p>Um case end-to-end que transforma cinco anos de vendas em previsões auditáveis e parâmetros operacionais para varejo.</p>
            <div className="stack-list" aria-label="Tecnologias do projeto">
              <span>Python</span><span>Pandas</span><span>scikit-learn</span><span>Plotly</span><span>React</span>
            </div>
          </div>
        </div>
        <div className="project-story-grid">
          <article>
            <span>01 / DESAFIO</span>
            <h3>Planejar 500 séries</h3>
            <p>Prever 90 dias por loja e produto sem consultar vendas futuras, apoiando cobertura e priorização de estoque.</p>
          </article>
          <article>
            <span>02 / ABORDAGEM</span>
            <h3>Validar como produção</h3>
            <p>Features históricas, seleção em holdout anterior, forecast recursivo, backtesting e intervalos conformais.</p>
          </article>
          <article>
            <span>03 / ENTREGA</span>
            <h3>Converter erro em ação</h3>
            <p>Dashboard filtrável, cenários de incerteza, ABC/XYZ, estoque de segurança e ponto de reposição.</p>
          </article>
        </div>
        <div className="project-flow" aria-label="Fluxo do projeto">
          <span>913 mil vendas</span><i>→</i><span>tratamento + EDA</span><i>→</i><span>features sem vazamento</span><i>→</i><span>previsão recursiva</span><i>→</i><span>decisão de estoque</span>
        </div>
        <p className="project-boundary"><strong>Escopo responsável:</strong> o dataset não contém preço, promoções, estoque disponível, rupturas ou lead time real. Por isso, os resultados apoiam decisões — não geram ordens de compra automáticas.</p>
      </section>

      <section className="section-shell" id="forecast">
        <div className="section-heading">
          <div>
            <p className="section-index">01 / PREVISÃO OPERACIONAL</p>
            <h2>O horizonte futuro em<br />ritmo diário.</h2>
          </div>
          <p>O modelo recursivo projeta cada dia usando apenas o histórico disponível e suas próprias previsões anteriores.</p>
        </div>
        <ForecastExplorer />
      </section>

      <section className="section-shell muted-section" id="models">
        <div className="section-heading compact-heading">
          <div>
            <p className="section-index">02 / QUALIDADE DO MODELO</p>
            <h2>O ganho é medido,<br />não presumido.</h2>
          </div>
          <p>Todos os modelos foram avaliados no mesmo corte de 90 dias. Menor erro representa melhor desempenho.</p>
        </div>
        <div className="model-layout">
          <article className="chart-panel model-panel">
            <div className="panel-heading">
              <div><span>Ranking de modelos</span><strong>Compare por métrica</strong></div>
              <span className="panel-tag success-tag">7 candidatos</span>
            </div>
            <ModelComparison />
          </article>
          <article className="winner-card">
            <p className="micro-label">MODELO ESCOLHIDO</p>
            <div className="winner-title"><span>01</span><h3>{data.overview.bestModel}</h3></div>
            <p>Melhor equilíbrio entre precisão, estabilidade recursiva e capacidade de capturar diferenças entre séries.</p>
            <dl>
              <div><dt>MAE</dt><dd>{data.overview.bestMae.toFixed(3).replace(".", ",")}</dd></div>
              <div><dt>SMAPE</dt><dd>{data.overview.bestSmape.toFixed(2).replace(".", ",")}%</dd></div>
              <div><dt>Ganho</dt><dd>{data.overview.improvementPct.toFixed(1).replace(".", ",")}%</dd></div>
            </dl>
          </article>
        </div>
        <article className="chart-panel validation-panel">
          <div className="panel-heading">
            <div><span>Validação temporal</span><strong>Real × previsto · últimos 90 dias de 2017</strong></div>
            <span className="panel-tag">45.000 previsões</span>
          </div>
          <LineChart points={data.validationDaily} series={validationSeries} ariaLabel="Comparação diária entre vendas reais e previstas na validação" />
        </article>
      </section>

      <section className="section-shell">
        <div className="section-heading compact-heading">
          <div>
            <p className="section-index">03 / PORTFÓLIO DE DEMANDA</p>
            <h2>Onde concentrar<br />atenção e capital.</h2>
          </div>
          <p>Rankings futuros orientam priorização de cobertura, sem substituir restrições de margem, lead time e capacidade.</p>
        </div>
        <div className="ranking-grid">
          <article className="chart-panel">
            <div className="panel-heading"><div><span>Previsão por loja</span><strong>Ranking do horizonte</strong></div></div>
            <RankedBars rows={data.futureByStore as unknown as Array<Record<string, number>>} labelKey="store" prefix="Loja" />
          </article>
          <article className="chart-panel">
            <div className="panel-heading"><div><span>Produtos prioritários</span><strong>Top 10 por demanda prevista</strong></div></div>
            <RankedBars rows={data.futureTopItems as unknown as Array<Record<string, number>>} labelKey="item" prefix="Item" />
          </article>
        </div>
      </section>

      <section className="section-shell pattern-section">
        <div className="pattern-copy">
          <p className="section-index">04 / PADRÕES QUE EXPLICAM</p>
          <h2>Tempo importa.<br />Contexto também.</h2>
          <p>O histórico cresceu {data.overview.historicalGrowthPct.toFixed(1).replace(".", ",")}% entre 2013 e 2017. A semana acelera progressivamente até o domingo.</p>
          <div className="feature-list">
            {data.featureImportance.slice(0, 5).map((feature, index) => (
              <div key={feature.feature}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <p>{feature.feature.replaceAll("_", " ")}</p>
                <div><i style={{ width: `${(feature.importance / maxImportance) * 100}%` }} /></div>
                <strong>{(feature.importance * 100).toFixed(1).replace(".", ",")}%</strong>
              </div>
            ))}
          </div>
        </div>
        <article className="weekday-card">
          <div className="panel-heading"><div><span>Sazonalidade semanal</span><strong>Média diária histórica</strong></div></div>
          <div className="weekday-bars" role="img" aria-label="Vendas médias aumentam ao longo da semana e atingem o pico no domingo">
            {data.weekdayProfile.map((day) => (
              <div key={day.label}>
                <span>{compactNumber.format(day.sales)}</span>
                <div><i style={{ height: `${(day.sales / maxWeekday) * 100}%` }} /></div>
                <strong>{day.label}</strong>
              </div>
            ))}
          </div>
          <p className="chart-caption">Domingo registra o maior volume médio; fins de semana ficam {data.overview.weekendUpliftPct.toFixed(1).replace(".", ",")}% acima dos dias úteis.</p>
        </article>
      </section>

      <section className="decision-section" id="recommendations">
        <div className="decision-header">
          <p className="section-index light-index">05 / DECISÕES RECOMENDADAS</p>
          <h2>Do forecast para<br />o plano de ação.</h2>
        </div>
        <div className="decision-grid">
          <article><span>01</span><h3>Antecipar cobertura de março</h3><p>Reposicionar compras e capacidade antes do mês de maior demanda prevista, preservando nível de serviço.</p></article>
          <article><span>02</span><h3>Proteger fins de semana</h3><p>Reforçar reposição de sexta a domingo e acompanhar ruptura nas séries de maior giro.</p></article>
          <article><span>03</span><h3>Priorizar Loja 2 e Item 28</h3><p>Usar o ranking como fila inicial de revisão para estoque de segurança, espaço e abastecimento.</p></article>
          <article><span>04</span><h3>Monitorar o erro em produção</h3><p>Comparar realizado e previsto semanalmente, recalibrando o modelo quando o padrão de demanda mudar.</p></article>
        </div>
        <div className="decision-footer">
          <p><strong>Nota de governança:</strong> previsão não é decisão automática. Margem, lead time, promoções e capacidade devem completar a recomendação.</p>
          <a href="#methodology">Entender o processo <span>↓</span></a>
        </div>
      </section>

      <section className="section-shell methodology" id="methodology">
        <div className="section-heading compact-heading">
          <div><p className="section-index">06 / METODOLOGIA</p><h2>Rastreável do dado<br />à previsão.</h2></div>
          <p>Uma sequência reproduzível, com corte temporal explícito e proteção contra vazamento de dados.</p>
        </div>
        <ol className="method-steps">
          <li><span>01</span><div><h3>Base tratada</h3><p>913 mil registros diários, 10 lojas e 50 produtos entre 2013 e 2017.</p></div></li>
          <li><span>02</span><div><h3>Features históricas</h3><p>Lags, janelas móveis e médias expansivas calculadas somente com datas anteriores.</p></div></li>
          <li><span>03</span><div><h3>Validação temporal</h3><p>Escolhas feitas em holdout anterior; últimos 90 dias de 2017 preservados para avaliação independente.</p></div></li>
          <li><span>04</span><div><h3>Forecast recursivo</h3><p>Cada previsão alimenta o próximo dia sem acesso às vendas reais futuras.</p></div></li>
        </ol>
      </section>

      <footer>
        <div className="brand footer-brand"><span className="brand-mark">DW</span><span><strong>DemandWise</strong><small>Retail demand intelligence</small></span></div>
        <p>Projeto de portfólio · Ciência de Dados aplicada a varejo e supply chain.</p>
        <a href="#top">Voltar ao topo ↑</a>
      </footer>
    </main>
  );
}
