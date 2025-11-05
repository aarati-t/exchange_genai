import os
import json
import datetime as dt
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

# Optional libraries
from google.cloud import bigquery
import google.generativeai as genai
from google.oauth2 import service_account
from jinja2 import Environment, FileSystemLoader, select_autoescape
from dotenv import load_dotenv
from sample_data import SAMPLE_SCENARIOS, SAMPLE_LAYERS
from data_providers import try_bigquery_query, try_sheets_read
from llm import simulate_with_gemini
from llm_insight import run_audit_query
from typing import Any, Dict, List, Optional
import json, os, uuid
# ---------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------
load_dotenv()
app = FastAPI()
templates = Jinja2Templates(directory="templates")

class AuditSearchFilters(BaseModel):
    from_: Optional[str] = None  # 'from' is reserved in Python; map manually
    to: Optional[str] = None
    departments: List[str] = []
    flag_types: List[str] = []
    min_confidence: float = 0.5
    explain: bool = True

class AuditSearchPayload(BaseModel):
    query: str
    filters: Dict[str, Any] = {}

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
# Jinja
templates_env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(["html", "xml"])
)
# ---------------------------------------------------------------------
# BigQuery configuration (migis-prototype project)
# ---------------------------------------------------------------------
BQ_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "migis-prototype")
BQ_KEY_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    r"\migis-prototype-27d09631dfe4.json"
)
BQ_ENABLED = True
bq_client = None

try:
    credentials = service_account.Credentials.from_service_account_file(BQ_KEY_PATH)
    
    # Initialize BigQuery client explicitly using credentials
    bq_client = bigquery.Client(credentials=credentials, project=BQ_PROJECT_ID)
    # TODO - on google app engine use below code
    #bq_client = bigquery.Client(project=BQ_PROJECT_ID)
except Exception as e:
    print("⚠️ BigQuery client init failed:", e)
    BQ_ENABLED = False

# ---------------------------------------------------------------------
# Google Generative AI (Gemini) optional setup
# ---------------------------------------------------------------------
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
GENAI_ENABLED = bool(GENAI_API_KEY)
if GENAI_ENABLED:
    try:
        genai.configure(api_key=GENAI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        print("⚠️ Gemini init failed:", e)
        GENAI_ENABLED = False
        gemini_model = None
else:
    gemini_model = None

# ---------------------------------------------------------------------
# Routes for pages
# ---------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def main_menu(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("decision_dashboards.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# other placeholder pages
@app.get("/green-simulator", response_class=HTMLResponse)
def green_simulator(request: Request):
    return templates.TemplateResponse("green_simulator.html", {"request": request})

@app.get("/climate-risk", response_class=HTMLResponse)
def climate_risk(request: Request):
    return templates.TemplateResponse("climate_risk.html", {"request": request})

@app.get("/citizen-app", response_class=HTMLResponse)
def citizen_app(request: Request):
    return templates.TemplateResponse("citizen_app.html", {"request": request})

@app.get("/oversight", response_class=HTMLResponse)
def oversight(request: Request):
    return templates.TemplateResponse("oversight.html", {"request": request})

@app.get("/sandbox", response_class=HTMLResponse)
def sandbox(request: Request):
    return templates.TemplateResponse("sandbox.html", {"request": request})

@app.get("/insights", response_class=HTMLResponse)
def insights(request: Request):
    return templates.TemplateResponse("insights.html", {"request": request})

# ---------------------------------------------------------------------
# API: KPIs
# ---------------------------------------------------------------------
@app.get("/api/kpis")
def get_kpis():
    if not BQ_ENABLED or not bq_client:
        return {"critical_alerts": 12, "ongoing_projects": 37, "complaints_24h": 214, "avg_confidence": 76}

    sql = """
    SELECT critical_alerts, ongoing_projects, complaints_24h, avg_confidence
    FROM `migis-prototype.migis_demo.kpis_daily`
    WHERE day = CURRENT_DATE()
    ORDER BY day DESC
    LIMIT 1
    """
    try:
        df = bq_client.query(sql).result().to_dataframe()
        if df.empty:
            raise ValueError("No KPI data")
        row = df.iloc[0]
        return {
            "critical_alerts": int(row["critical_alerts"]),
            "ongoing_projects": int(row["ongoing_projects"]),
            "complaints_24h": int(row["complaints_24h"]),
            "avg_confidence": int(row["avg_confidence"]),
        }
    except Exception as e:
        print("⚠️ KPI fetch error:", e)
        return {"critical_alerts": 12, "ongoing_projects": 37, "complaints_24h": 214, "avg_confidence": 76}

# ---------------------------------------------------------------------
# API: Map points (Health + Infra)
# ---------------------------------------------------------------------
@app.get("/api/health-alerts")
def get_health_alerts():
    if not BQ_ENABLED or not bq_client:
        return {
            "points": [
                {"block": "Satara", "disease": "Dengue", "risk_score": 78, "lat": 17.68, "lng": 73.99},
                {"block": "Thane", "disease": "Leptospirosis", "risk_score": 66, "lat": 19.21, "lng": 72.97},
            ]
        }

    sql = """
    SELECT lat, lng, label AS block, note AS disease, risk_score
    FROM `migis-prototype.migis_demo.map_points`
    WHERE category = 'Health'
    ORDER BY risk_score DESC
    LIMIT 100
    """
    df = bq_client.query(sql).result().to_dataframe()
    points = [
        {"block": r["block"], "disease": r["disease"], "risk_score": int(r["risk_score"]),
         "lat": float(r["lat"]), "lng": float(r["lng"])}
        for _, r in df.iterrows()
    ]
    return {"points": points}

@app.get("/api/infra-events")
def get_infra_events():
    if not BQ_ENABLED or not bq_client:
        return {
            "points": [
                {"asset": "Water Pipe - Zone Y", "note": "65% fail prob", "lat": 18.52, "lng": 73.85},
                {"asset": "Bridge Y", "note": "80% fail in 6 months", "lat": 19.09, "lng": 74.74},
            ]
        }

    sql = """
    SELECT lat, lng, label AS asset, note
    FROM `migis-prototype.migis_demo.map_points`
    WHERE category IN ('Water','Transport')
    ORDER BY label
    LIMIT 100
    """
    df = bq_client.query(sql).result().to_dataframe()
    points = [
        {"asset": r["asset"], "note": r["note"], "lat": float(r["lat"]), "lng": float(r["lng"])}
        for _, r in df.iterrows()
    ]
    return {"points": points}

# ---------------------------------------------------------------------
# API: Alerts Log (table view)
# ---------------------------------------------------------------------
@app.get("/api/alerts-log")
def get_alerts_log():
    try:
        sql = """
        SELECT FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', ts) AS time, district, type, score, note
        FROM `migis-prototype.migis_demo.alerts_log`
        ORDER BY ts DESC
        LIMIT 50
        """
        df = bq_client.query(sql).result().to_dataframe()
        rows = [
            dict(time=r["time"], district=r["district"], type=r["type"],
                 score=int(r["score"]), note=r["note"])
            for _, r in df.iterrows()
        ]
        return {"rows": rows}
    except Exception as e:
        print("🚨 BigQuery alerts-log error:", e)
        return JSONResponse(status_code=200, content={"rows": [], "error": str(e)})

# ---------------------------------------------------------------------
# API: Time Series (Chart.js)
# ---------------------------------------------------------------------
@app.get("/api/time-series")
def get_time_series():
    if not BQ_ENABLED or not bq_client:
        return {"labels": ["W1","W2","W3","W4","W5","W6"],
                "complaints": [120,160,140,210,190,230],
                "confidence": [60,64,67,72,71,76]}

    sql = """
    SELECT bucket, complaints, confidence
    FROM `migis-prototype.migis_demo.complaints_timeseries`
    ORDER BY bucket
    """
    df = bq_client.query(sql).result().to_dataframe()
    labels = df["bucket"].tolist()
    complaints = [int(v) for v in df["complaints"].tolist()]
    confidence = [int(v) for v in df["confidence"].tolist()]
    return {"labels": labels, "complaints": complaints, "confidence": confidence}

# ---------------------------------------------------------------------
# API: Chat (Gemini LLM)
# ---------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
def chat_with_ai(req: ChatRequest):
    default_answer = "I suggest prioritizing Satara (outbreak 78%) and scheduling a bridge audit in Nashik within 7 days."
    if not GENAI_ENABLED or not gemini_model:
        return {"answer": default_answer}

    try:
        prompt = (
            "You are an executive assistant in Maharashtra Command Center. "
            "Provide concise, actionable insights for government officers.\n\n"
            f"User: {req.query}"
        )
        resp = gemini_model.generate_content(prompt)
        answer = getattr(resp, "text", "") or default_answer
        return {"answer": answer.strip()}
    except Exception as e:
        print("⚠️ Gemini chat error:", e)
        return {"answer": default_answer}

# ---------------------------------------------------------------------
# API: green simulatro 
#

@app.get("/api/scenarios")
def get_scenarios():
    # Could enrich with BQ/Sheets KPIs if needed
    return JSONResponse(SAMPLE_SCENARIOS)

@app.get("/api/layers")
def get_layers():
    # Could fetch from BQ (e.g., flood rasters converted to polygons) or Sheets catalog
    return JSONResponse(SAMPLE_LAYERS)

@app.get("/api/cost-benefit")
def get_cost_benefit():
    """
    Comparative cost-benefit: traditional vs sustainable over a timeline.
    In real use, read from BigQuery project tables.
    """
    sql = """
    -- Example placeholder SQL
    SELECT 2025 AS year, 100 AS traditional_cost, 92 AS sustainable_cost, 12000 AS co2_tons
    UNION ALL SELECT 2030, 140, 120, 14000
    UNION ALL SELECT 2040, 220, 170, 16000
    """
    rows = try_bigquery_query(sql)
    return JSONResponse({"rows": rows})

@app.get("/api/kpis-green")
def get_kpis():
    data = try_sheets_read(SHEETS_ID, "KPIs") if SHEETS_ID else try_sheets_read("NO_SHEETS")
    return JSONResponse({"rows": data})

@app.post("/api/what-if")
async def what_if(request: Request):
    payload = await request.json()
    # Expect: { location, asset_type, materials, constraints, notes }
    user_prompt = f"""
Location: {payload.get('location')}
Asset Type: {payload.get('asset_type')}
Materials: {payload.get('materials')}
Constraints: {payload.get('constraints')}
Notes: {payload.get('notes')}
"""
    text = simulate_with_gemini(user_prompt)
    return JSONResponse({"result": text})

# audit
@app.post("/audit-search")
async def audit_search(payload: AuditSearchPayload):
    """
    POST body:
    {
      "query": "Show vendor similarity > 80% in the past 6 months",
      "filters": {
        "from": "2025-06-01",
        "to": "2025-11-05",
        "departments": ["Public Works"],
        "flag_types": ["vendor_relationship","cost_pattern"],
        "min_confidence": 0.6,
        "explain": true
      }
    }
    """
    # Normalize 'from' field name that conflicts with Python keyword
    filters = payload.filters.copy()
    if "from_" in filters and not filters.get("from"):
        filters["from"] = filters.pop("from_")

    data = run_audit_query(payload.query, filters)
    return data

## citizen

@app.get("/api/complaints")
def get_complaints():
    BQ_DATASET = "migp_dataset"
    BQ_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.citizen_complaints"
    try:
        credentials = service_account.Credentials.from_service_account_file(BQ_KEY_PATH)
        client = bigquery.Client(credentials=credentials, project=BQ_PROJECT_ID)
        rows = client.query(f"SELECT * FROM `{BQ_TABLE}` ORDER BY created_at DESC LIMIT 50").result()
        data = [dict(row) for row in rows]

        # 🔧 Convert datetime objects to strings for JSON
        for row in data:
            for k, v in row.items():
                if isinstance(v, datetime):
                    row[k] = v.isoformat()

    except Exception as e:
        print("⚠️ BigQuery unavailable, using local citizen.json fallback:", e)
        file_path = os.path.join(os.getcwd(), "citizen.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

    return JSONResponse(content=data)

@app.post("/api/complaints")
async def add_complaint(req: Request):
    data = await req.json()
    data["id"] = str(uuid.uuid4())[:8]
    data["status"] = "Submitted"
    data["prediction"] = "Likely to resolve in 7 days"
    data["created_at"] = "CURRENT_TIMESTAMP"
    BQ_DATASET = "migp_dataset"
    BQ_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.citizen_complaints"
    try:
        credentials = service_account.Credentials.from_service_account_file(BQ_KEY_PATH)
        client = bigquery.Client(credentials=credentials, project=BQ_PROJECT_ID)
        #client = bigquery.Client()
        table = client.get_table(BQ_TABLE)
        client.insert_rows_json(table, [data])
        msg = "Complaint added to BigQuery!"
    except Exception:
        if os.path.exists("citizen.json"):
            store = json.load(open("citizen.json"))
        else:
            store = []
        store.append(data)
        json.dump(store, open("citizen.json", "w"), indent=2)
        msg = "BigQuery not available, saved to citizen.json."

    return JSONResponse({"message": "✅ Data saved successfully!"})

@app.get("/api/dashboard")
def dashboard_data():
    try:
        credentials = service_account.Credentials.from_service_account_file(BQ_KEY_PATH)
    
        # Initialize BigQuery client explicitly using credentials
        client = bigquery.Client(credentials=credentials, project=BQ_PROJECT_ID)
        # TODO - on google app engine use below code
        
        #client = bigquery.Client()
        query = """
        SELECT category, COUNT(*) as count
        FROM `migp_dataset.citizen_complaints`
        GROUP BY category
        """
        rows = client.query(query).result()
        data = [dict(r) for r in rows]
    except Exception:
        if os.path.exists("citizen.json"):
            with open("citizen.json") as f:
                items = json.load(f)
            counts = {}
            for i in items:
                counts[i["category"]] = counts.get(i["category"], 0) + 1
            data = [{"category": k, "count": v} for k, v in counts.items()]
        else:
            data = []
    return data
### field DPS
@app.get("/field-agent", response_class=HTMLResponse)
def field_agent(request: Request):
    return templates.TemplateResponse("field_agent.html", {"request": request})         
# ---------------------------------------------------------------------
# Run locally
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
