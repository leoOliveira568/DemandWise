import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("renders the DemandWise dashboard", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /DemandWise/);
  assert.match(html, /Previsão que vira/);
  assert.match(html, /Random Forest/);
  assert.match(html, /2,2(?:\u00a0| )mi/);
  assert.match(html, /Do forecast para/);
  assert.match(html, /O PROJETO EM 60 SEGUNDOS/);
  assert.match(html, /Escopo responsável/);
  assert.match(html, /Carregando 45.000 previsões/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/);
});

test("ships the interactive forecast explorer", async () => {
  const [source, forecastPayload] = await Promise.all([
    readFile(new URL("../app/Dashboard.tsx", import.meta.url), "utf8"),
    readFile(new URL("../public/forecast-data.json", import.meta.url), "utf8"),
  ]);
  const forecast = JSON.parse(forecastPayload);
  assert.match(source, /Todas as lojas/);
  assert.match(source, /Todos os produtos/);
  assert.match(source, /Cenário inferior \(90%\)/);
  assert.match(source, /Exportar CSV/);
  assert.match(source, /SIMULADOR DE COBERTURA/);
  assert.match(source, /Ponto de reposição/);
  assert.equal(forecast.forecastDates.length, 90);
  assert.equal(forecast.forecastRows.length, 45_000);
  assert.equal(forecast.inventoryPolicies.length, 500);
});
