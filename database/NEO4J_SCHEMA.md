# Neo4j Financial Knowledge Graph Schema Documentation

This document describes the schema, entity labels, relationships, properties, constraints, and indexes for the Financial GraphRAG system.

---

## 1. Node Labels and Properties

### `Company`
Represents the corporation whose filings are ingested.
- `ticker` (String, Unique): Stock ticker symbol (e.g., `"LTIM"`, `"AAPL"`, `"MSFT"`).
- `name` (String): Full corporate name.
- `industry` (String, Optional): Industry sector.
- `isin` (String, Optional): International Securities Identification Number.

### `AnnualReport` / `SourceDocument`
Represents an ingested filing / annual report.
- `document_id` (String, Unique): Unique report identifier (e.g., `"LTIM_FY2025-26_Annual_Report"`).
- `company` (String): Company name.
- `ticker` (String): Ticker symbol.
- `fiscal_year` (String): Fiscal year (e.g., `"FY2025-26"`).
- `source_url` (String): Source URL of the document.
- `num_pages` (Integer): Total page count.

### `Section`
Represents a structural section within an annual report.
- `section_id` (String, Unique): ID in format `{document_id}_{section_name}`.
- `name` (String): Section title (e.g., `"Management Discussion & Analysis"`, `"Financial Statements"`).
- `document_id` (String): Parent document identifier.

### `Chunk`
Represents an atomic, searchable text or table chunk.
- `chunk_id` (String, Unique): Deterministic ID (e.g., `"LTIM_FY2025-26_Annual_Report_p23_s1"`).
- `page` (Integer): Source page number in the original PDF.
- `section` (String): Section name.
- `chunk_type` (String): `"prose"` or `"table"`.
- `text` (String): Full or excerpt text.

### `FinancialMetric`
Represents an abstract financial KPI concept.
- `name` (String, Unique): Metric name (e.g., `"revenue"`, `"ebitda"`, `"profit_after_tax"`, `"earnings_per_share"`).
- `category` (String): Category (e.g., `"profit_and_loss"`, `"balance_sheet"`, `"cash_flow"`).
- `description` (String): Human-readable metric description.

### Specialized Metric Labels
Nodes with `FinancialMetric` also carry specific labels for direct indexing:
- `Revenue`
- `EBITDA`
- `Profit`
- `EPS`
- `CashFlow`
- `Debt`
- `Asset`
- `Liability`

### `FinancialValue`
Represents a specific quantitative financial fact reported in a filing.
- `fact_id` (String, Unique): Unique identifier (e.g., `"LTIM_FY2025-26_Annual_Report_revenue_FY2025_26"`).
- `metric` (String): The metric name.
- `value` (Float / Double): Quantitative numeric value.
- `unit` (String): Unit of measurement (e.g., `"INR million"`, `"USD million"`, `"%"`, `"INR"`).
- `statement_type` (String): `"consolidated"` or `"standalone"`.
- `confidence` (Float): Extraction confidence score (`1.0` for structured tables).
- `extraction_method` (String): `"structured_financial"` or `"docling_table"`.

### `ReportingPeriod`
Represents the time horizon of reported numbers.
- `label` (String, Unique): Period label (e.g., `"FY2022-23"`, `"FY2023-24"`, `"FY2024-25"`, `"FY2025-26"`).
- `start` (Date/String): Start date (e.g., `"2025-04-01"`).
- `end` (Date/String): End date (e.g., `"2026-03-31"`).

### `SourcePage`
Represents the exact page of the document.
- `page_key` (String, Unique): `{document_id}:{page_number}`.
- `page` (Integer): Page number.
- `section` (String): Section heading on that page.

### Strategic & Qualitative Entities
- `BusinessSegment`: Segments (e.g., `"Banking, Financial Services & Insurance"`, `"Hi-Tech, Media & Entertainment"`, `"Manufacturing & Resources"`).
- `Geography`: Operating regions (e.g., `"North America"`, `"Europe"`, `"India"`, `"Rest of the World"`).
- `Risk`: Disclosed risk factors (e.g., `"Cybersecurity Risk"`, `"Currency Volatility"`, `"Talent Attrition"`).
- `Strategy` / `Initiative`: Strategic initiatives (e.g., `"Fit4Future"`, `"Canvas.ai"`, `"ESG Net Zero"`).
- `Executive`: Key management executives (e.g., `"CEO"`, `"CFO"`, `"Board Chairman"`).

---

## 2. Relationships

```
(:Company)-[:HAS_REPORT]->(:AnnualReport)
(:AnnualReport)-[:CONTAINS]->(:Section)
(:Section)-[:CONTAINS]->(:Chunk)

(:Company)-[:HAS_METRIC]->(:FinancialMetric)
(:FinancialMetric)-[:REPORTED_IN]->(:AnnualReport)
(:FinancialMetric)-[:HAS_VALUE]->(:FinancialValue)
(:FinancialValue)-[:FOR_PERIOD]->(:ReportingPeriod)
(:FinancialValue)-[:FOUND_ON]->(:SourcePage)
(:FinancialValue)-[:SUPPORTED_BY]->(:AnnualReport)

(:Company)-[:HAS_SEGMENT]->(:BusinessSegment)
(:Company)-[:OPERATES_IN]->(:Geography)
(:Company)-[:FACES]->(:Risk)
(:Company)-[:HAS_STRATEGY]->(:Strategy)
(:Company)-[:LED_BY]->(:Executive)

(:Entity)-[:DRIVES]->(:Entity | :FinancialMetric)
(:Entity)-[:IMPROVES]->(:Entity | :FinancialMetric)
(:Entity)-[:IMPACTS]->(:Entity | :FinancialMetric)
(:Entity)-[:SUPPORTS]->(:Entity | :FinancialMetric)
(:Entity)-[:MITIGATES]->(:Risk)
(:Entity)-[:CONTRIBUTES_TO]->(:Entity)
```

---

## 3. Constraints and Indexes

```cypher
CREATE CONSTRAINT company_ticker_unique IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE;
CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:AnnualReport) REQUIRE d.document_id IS UNIQUE;
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE;
CREATE CONSTRAINT metric_name_unique IF NOT EXISTS FOR (m:FinancialMetric) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT fact_id_unique IF NOT EXISTS FOR (v:FinancialValue) REQUIRE v.fact_id IS UNIQUE;
CREATE CONSTRAINT period_label_unique IF NOT EXISTS FOR (p:ReportingPeriod) REQUIRE p.label IS UNIQUE;
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;
CREATE CONSTRAINT page_key_unique IF NOT EXISTS FOR (p:SourcePage) REQUIRE p.page_key IS UNIQUE;

CREATE INDEX metric_category_idx IF NOT EXISTS FOR (m:FinancialMetric) ON (m.category);
CREATE INDEX chunk_page_idx IF NOT EXISTS FOR (ch:Chunk) ON (ch.page);
CREATE INDEX value_period_idx IF NOT EXISTS FOR (v:FinancialValue) ON (v.metric);
```
