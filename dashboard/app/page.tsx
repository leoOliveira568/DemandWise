import type { Metadata } from "next";
import Dashboard from "./Dashboard";

export const metadata: Metadata = {
  title: "DemandWise | Retail Demand Intelligence",
  description:
    "Dashboard executivo de previsão de demanda por loja e produto, com validação temporal, comparação de modelos e projeções operacionais.",
};

export default function Home() {
  return <Dashboard />;
}
