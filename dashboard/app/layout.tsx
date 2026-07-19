import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DemandWise | Retail Demand Intelligence",
  description: "Previsão de demanda para decisões de estoque, planejamento e supply chain.",
  openGraph: {
    title: "DemandWise — Retail Demand Intelligence",
    description: "Da série temporal à decisão de estoque: 500 séries previstas em um horizonte recursivo de 90 dias.",
    type: "website",
    images: [{ url: "/og.png", width: 1792, height: 1024, alt: "DemandWise — Retail Demand Intelligence" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "DemandWise — Retail Demand Intelligence",
    description: "Forecasting de varejo com validação temporal e decisões operacionais.",
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="pt-BR">
      <body>{children}</body>
    </html>
  );
}
