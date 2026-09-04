# LTM Financial Knowledge Graph Schema

## 1. Company

Represents the company being analyzed.

### Properties

- company_id
- name
- ticker

Example:

Company {
    company_id: "LTM"
    name: "LTM Limited"
    ticker: "LTM"
}


## 2. FinancialMetric

Represents a financial KPI or financial concept.

### Properties

- metric_id
- name
- category

Example:

FinancialMetric {
    metric_id: "revenue"
    name: "Revenue"
    category: "profit_and_loss"
}


## 3. FinancialValue

Represents the actual reported value.

### Properties

- fact_id
- value
- unit
- confidence
- extraction_method

Example:

FinancialValue {
    fact_id: "LTM_FY26_AR_001_revenue_FY26"
    value: 423076
    unit: "INR million"
    confidence: 1.0
    extraction_method: "structured_kpi"
}


## 4. ReportingPeriod

Represents the reporting period.

### Properties

- period_id
- label
- start
- end

Example:

ReportingPeriod {
    period_id: "FY2025-26"
    label: "FY2025-26"
    start: "2025-04-01"
    end: "2026-03-31"
}


## 5. SourceDocument

Represents the source document.

### Properties

- document_id
- document_type
- title
- company
- reporting_period

Example:

SourceDocument {
    document_id: "LTM_FY26_AR_001"
    document_type: "annual_report"
    title: "LTM Limited Integrated Annual Report 2025-26"
    company: "LTM Limited"
    reporting_period: "FY2025-26"
}


## 6. SourcePage

Represents the exact page from which a fact was extracted.

### Properties

- page_id
- page_number
- section

Example:

SourcePage {
    page_id: "LTM_FY26_AR_001_p23"
    page_number: 23
    section: "Key Performance Indicators"
}


# Relationships

## Company → FinancialMetric

(:Company)-[:HAS_METRIC]->(:FinancialMetric)


## FinancialMetric → FinancialValue

(:FinancialMetric)-[:HAS_VALUE]->(:FinancialValue)


## FinancialValue → ReportingPeriod

(:FinancialValue)-[:FOR_PERIOD]->(:ReportingPeriod)


## FinancialValue → SourcePage

(:FinancialValue)-[:SUPPORTED_BY]->(:SourcePage)


## SourcePage → SourceDocument

(:SourcePage)-[:PART_OF]->(:SourceDocument)


## Company → SourceDocument

(:Company)-[:PUBLISHED]->(:SourceDocument)


# Complete Example

(:Company {
    name: "LTM Limited",
    ticker: "LTM"
})

-[:HAS_METRIC]->

(:FinancialMetric {
    name: "Revenue"
})

-[:HAS_VALUE]->

(:FinancialValue {
    value: 423076,
    unit: "INR million",
    confidence: 1.0
})

-[:FOR_PERIOD]->

(:ReportingPeriod {
    label: "FY2025-26"
})

(:FinancialValue)

-[:SUPPORTED_BY]->

(:SourcePage {
    page_number: 23,
    section: "Key Performance Indicators"
})

-[:PART_OF]->

(:SourceDocument {
    document_id: "LTM_FY26_AR_001"
})


# Design Principles

1. Every financial fact must retain source provenance.

2. Every financial fact must have a reporting period.

3. Values and units must remain separate.

4. Extraction confidence must be preserved.

5. The graph must be deterministic.

6. The LLM must not invent financial facts.

7. Every generated answer should eventually be traceable to graph evidence.

8. The graph should support both structured queries and semantic retrieval.