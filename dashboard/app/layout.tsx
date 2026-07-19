import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.includes("localhost") ? "http" : "https");
  const base = new URL(`${protocol}://${host}`);

  return {
    metadataBase: base,
    title: "DemandWise | Retail Demand Intelligence",
    description: "Previsão de demanda para decisões de estoque, planejamento e supply chain.",
    openGraph: {
      title: "DemandWise — Retail Demand Intelligence",
      description: "Da série temporal à decisão de estoque: 500 séries previstas em um horizonte recursivo de 90 dias.",
      type: "website",
      images: [{ url: new URL("/og.png", base).toString(), width: 1792, height: 1024, alt: "DemandWise — Retail Demand Intelligence" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "DemandWise — Retail Demand Intelligence",
      description: "Forecasting de varejo com validação temporal e decisões operacionais.",
      images: [new URL("/og.png", base).toString()],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="pt-BR">
      <body>{children}</body>
    </html>
  );
}
