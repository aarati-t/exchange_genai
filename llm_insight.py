"""
Gemini-based Query Audit engine for the Red Flag Automator.

Requirements:
  pip install google-generativeai pydantic

Env:
  GOOGLE_API_KEY=xxxx
"""

import os
import json
from typing import Any, Dict, List, Optional

import google.generativeai as genai

# Configure Gemini
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
_API_KEY = os.getenv("GOOGLE_API_KEY")
_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or \
  


if not _API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set. Please set it in your environment.")

genai.configure(api_key=_API_KEY)

# Strict schema requested from the model (we instruct via system prompt + response_mime_type)
SCHEMA_EXAMPLE = {
    "summary": "One-paragraph executive summary for auditors.",
    "confidence": 0.72,
    "tables": [
        {
            "title": "Table title",
            "columns": ["Col1", "Col2", "Col3"],
            "rows": [["r1c1", "r1c2", "r1c3"], ["r2c1", "r2c2", "r2c3"]]
        }
    ],
    "charts": [
        {
            "title": "Chart title",
            "type": "timeseries",   # 'timeseries' | 'bar' | 'line'
            "labels": ["2025-07","2025-08","2025-09"],
            "series": [
                {"name": "Flags", "data": [12,18,21]}
            ]
        }
    ],
    "explanations": [
        "Reason 1 in plain language oriented to inspection.",
        "Reason 2…"
    ],
    "citations": [],
    "sql": "/* Optional diagnostic SQL or pseudo-SQL used to fetch */"
}

SYSTEM_PROMPT = """
You are the "Query Audit" assistant powering a government audit insights console.
Your role is to help auditors and inspection teams investigate anomalies without using words like 'corruption'.
Focus on: inspection readiness, anomaly patterns, deep-dive diagnostics, explainability.

Output must be STRICT JSON with keys:
- summary (string; <= 80 words, executive tone, inspection-focused)
- confidence (float 0..1; model confidence about usefulness of results)
- tables (array of tables with {title, columns:[], rows:[[]]})
- charts (array with {title, type: 'timeseries'|'bar'|'line', labels:[], series:[{name,data:[]}]})
- explanations (array of 1-5 short bullet strings explaining WHY flagged or prioritized)
- citations (array; optional strings)
- sql (string; optional, best-effort pseudo-SQL for reproducibility)

Safety/Style:
- Avoid the words 'corrupt', 'corruption', 'bribe'. Use 'inspection', 'deviation', 'anomaly', 'policy variance', 'deep dive'.
- Never include markdown. JSON only.
- Be conservative in claims. If data is missing, say so in explanations.
- Keep numbers realistic and aligned to the user's filters.

If the user asks for vendor similarity, cost deviation, overrides, or ESG, prefer these canonical column names:
- Vendor Relationship: Vendor, Tender ID, Linked Entities, Similarity %, Last Checked
- Cost Pattern: Project, Predicted (₹), Actual (₹), Deviation %, Historical Variance, Notes
- Override Monitor: Officer/Dept., Recs, Overrides %, Peer Avg., Remarks
- ESG Lens: Project ID, Contractor ESG, Site Sensitivity, Approval Date, Status
"""

def _build_user_payload(query: str, filters: Dict[str, Any]) -> str:
    """Compose a compact JSON instruction for the model with user query + filters."""
    payload = {
        "user_query": query,
        "filters": {
            "from": filters.get("from"),
            "to": filters.get("to"),
            "departments": filters.get("departments", []),
            "flag_types": filters.get("flag_types", []),
            "min_confidence": filters.get("min_confidence", 0.5),
            "explain": bool(filters.get("explain", True))
        },
        "expected_schema": SCHEMA_EXAMPLE
    }
    return json.dumps(payload, ensure_ascii=False)

def run_audit_query(user_query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls Gemini with a strict JSON contract. Returns a parsed dict with safe fallbacks.
    """
    model = genai.GenerativeModel(
        model_name=_GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT
    )
    prompt = _build_user_payload(user_query, filters)

    # Ask Gemini to return JSON only
    resp = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            max_output_tokens=1200,
            response_mime_type="application/json"
        )
    )

    # Parse
    text = resp.text or "{}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback envelope
        data = {
            "summary": "No structured result. Please refine your query or adjust filters.",
            "confidence": 0.0,
            "tables": [],
            "charts": [],
            "explanations": ["The model returned an unexpected format."],
            "citations": [],
            "sql": ""
        }

    # Minimal normalization
    data.setdefault("summary", "")
    data.setdefault("confidence", 0.5)
    data.setdefault("tables", [])
    data.setdefault("charts", [])
    data.setdefault("explanations", [])
    data.setdefault("citations", [])
    data.setdefault("sql", "")

    return data
