import os, json, uuid, math, datetime as dt
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jinja2 import Environment, FileSystemLoader, select_autoescape
from dotenv import load_dotenv

# Optional libraries
from google.cloud import bigquery
import google.generativeai as genai
from google.oauth2 import service_account

# local imports
from sample_data import SAMPLE_SCENARIOS, SAMPLE_LAYERS
from data_providers import try_bigquery_query, try_sheets_read
from llm import simulate_with_gemini
from llm_insight import run_audit_query

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

# Static + CORS
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Jinja env
templates_env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(["html", "xml"])
)

# ---------------------------------------------------------------------
# BigQuery configuration
# ---------------------------------------------------------------------
BQ_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "migis-prototype")
BQ_DATASET = "migp_dataset"
BQ_KEY_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    r"\migis-prototype-27d09631dfe4.json"
)
BQ_ENABLED = True
bq_client = None

try:
    credentials = service_account.Credentials.from_service_account_file(BQ_KEY_PATH)
    bq_client = bigquery.Client(credentials=credentials, project=BQ_PROJECT_ID)
except Exception as e:
    print("⚠️ BigQuery init failed:", e)
    BQ_ENABLED = False

# ---------------------------------------------------------------------
# Gemini (Generative AI)
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

@app.get("/green-simulator", response_class=HTMLResponse)
def green_simulator(request: Request):
    return templates.TemplateResponse("green_simulator.html", {"request": request})
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


@app.get("/citizen-app", response_class=HTMLResponse)
def citizen_app(request: Request):
    return templates.TemplateResponse("citizen_app.html", {"request": request})

@app.get("/oversight", response_class=HTMLResponse)
def oversight(request: Request):
    return templates.TemplateResponse("oversight.html", {"request": request})

@app.get("/insights", response_class=HTMLResponse)
def insights(request: Request):
    return templates.TemplateResponse("insights.html", {"request": request})

# ---------------------------------------------------------------------
# Citizen Complaints (existing logic)
# ---------------------------------------------------------------------
@app.get("/api/complaints")
def get_complaints():
    BQ_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.citizen_complaints"
    try:
        credentials = service_account.Credentials.from_service_account_file(BQ_KEY_PATH)
        client = bigquery.Client(credentials=credentials, project=BQ_PROJECT_ID)
        rows = client.query(f"SELECT * FROM `{BQ_TABLE}` ORDER BY created_at DESC LIMIT 50").result()
        data = [dict(row) for row in rows]
        for row in data:
            for k, v in row.items():
                if isinstance(v, datetime):
                    row[k] = v.isoformat()
    except Exception as e:
        print("⚠️ BigQuery unavailable, fallback to citizen.json:", e)
        if os.path.exists("citizen.json"):
            data = json.load(open("citizen.json"))
        else:
            data = []
    return JSONResponse(content=data)

@app.post("/api/complaints")
async def add_complaint(req: Request):
    data = await req.json()
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # -------------------------------------------------------------
    # 1️⃣ Basic Complaint Data
    # -------------------------------------------------------------
    data["id"] = str(uuid.uuid4())[:8]
    data["status"] = "Submitted"
    data["prediction"] = "Likely to resolve in 7 days"
    data["created_at"] = now

    # -------------------------------------------------------------
    # 2️⃣ Field Normalization (for Citizen Form)
    # -------------------------------------------------------------
    # Convert citizen form fields to DPS-compatible format
    desc = data.get("description", "")
    cat = (data.get("category") or "general").lower()
    loc = data.get("location", "Mumbai")

    # Auto-generate missing fields for DPS screen
    data["title"] = data.get("title") or desc.title() or f"{cat.title()} Issue"
    data["department"] = data.get("department", "Municipal")
    data["category"] = cat
    data["priority"] = data.get("priority", "P2")
    data["location_name"] = data.get("location_name", loc)

    # -------------------------------------------------------------
    # 3️⃣ SLA, Timestamps, and Location Defaults
    # -------------------------------------------------------------
    if cat in ["pipe_leak", "sanitation"]:
        sla_hours = 8
    elif cat in ["pothole", "bridge", "infra_road"]:
        sla_hours = 12
    else:
        sla_hours = 10
    data["sla_hours"] = sla_hours
    data["due_at"] = (dt.datetime.fromisoformat(now.replace("Z","")) + dt.timedelta(hours=sla_hours)).isoformat() + "Z"

    # Provide default lat/lon if not given
    if not data.get("lat"):
        data["lat"], data["lon"] = 19.07, 72.87

    # -------------------------------------------------------------
    # 4️⃣ Dynamic Service Prioritization (DPS) Scoring
    # -------------------------------------------------------------
    base_urgency = 60
    citizen_impact = 65
    predictive_risk = 0.3

    text = (desc + " " + loc).lower()

    if "hospital" in text or "school" in text:
        citizen_impact += 15
    if "bridge" in text or "pothole" in text:
        base_urgency += 15
    if "ambulance" in text:
        predictive_risk = 0.8
    elif "bridge" in text or "pipe" in text:
        predictive_risk = 0.6
    elif "road" in text:
        predictive_risk = 0.4

    data["base_urgency"] = base_urgency
    data["citizen_impact"] = citizen_impact
    data["predictive_risk"] = predictive_risk

    # DPS = weighted sum
    data["dps_score"] = round(
        0.4 * base_urgency + 0.3 * (predictive_risk * 100) + 0.2 * citizen_impact, 1
    )

    # Reason text for display in Field Agent app
    data["priority_reason"] = data.get(
        "priority_reason",
        f"{cat.replace('_',' ').title()} complaint near {loc} (auto-prioritized)"
    )

    # -------------------------------------------------------------
    # 5️⃣ Save to BigQuery or JSON Fallback
    # -------------------------------------------------------------
    BQ_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.citizen_complaints"

    try:
        credentials = service_account.Credentials.from_service_account_file(BQ_KEY_PATH)
        client = bigquery.Client(credentials=credentials, project=BQ_PROJECT_ID)
        table = client.get_table(BQ_TABLE)
        client.insert_rows_json(table, [data])
        msg = "Saved to BigQuery"
    except Exception as e:
        print("⚠️ BigQuery unavailable, saving locally:", e)
        file_path = "citizen.json"

        # Safe load (handles dict or list)
        if os.path.exists(file_path):
            raw = json.load(open(file_path))
            if isinstance(raw, dict) and "tasks" in raw:
                store = raw["tasks"]
            elif isinstance(raw, list):
                store = raw
            else:
                store = []
        else:
            store = []

        store.append(data)
        json.dump(store, open(file_path, "w"), indent=2)
        msg = "Saved to citizen.json"

    return JSONResponse({"message": f"✅ Complaint logged and enriched for DPS. {msg}"})


# ---------------------------------------------------------------------
# ---------------------- FIELD AGENT DPS SECTION -----------------------
# ---------------------------------------------------------------------
CITIZEN_JSON = "citizen.json"
def _now_iso(): return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# simple helpers
#def read_local(): return json.load(open(CITIZEN_JSON)) if os.path.exists(CITIZEN_JSON) else []
def read_local():
    """Always return a list of task dicts from citizen.json"""
    if not os.path.exists("citizen.json"):
        return []
    try:
        data = json.load(open("citizen.json"))
        # handle both formats: {"tasks": [...]} or just [...]
        if isinstance(data, dict) and "tasks" in data:
            return data["tasks"]
        elif isinstance(data, list):
            return data
        else:
            return []
    except Exception as e:
        print("⚠️ Error reading citizen.json:", e)
        return []


def write_local(rows): json.dump(rows, open(CITIZEN_JSON, "w"), indent=2)
def _compute_due(created, sla): 
    if not sla: return ""
    base = dt.datetime.fromisoformat(created.replace("Z", ""))
    return (base + dt.timedelta(hours=sla)).isoformat() + "Z"

class Task(BaseModel):
    id: str = str(uuid.uuid4())[:8]
    title: str
    department: Optional[str] = None
    category: Optional[str] = None
    priority: str = "P2"
    base_urgency: int = 50
    predictive_risk: float = 0.0
    citizen_impact: int = 50
    sla_hours: Optional[int] = 24
    created_at: Optional[str] = None
    due_at: Optional[str] = None
    status: str = "Submitted"
    lat: Optional[float] = None
    lon: Optional[float] = None
    location_name: Optional[str] = None
    assignee: Optional[str] = None
    priority_reason: Optional[str] = None
    dps_score: float = 0.0

@app.get("/field-agent", response_class=HTMLResponse)
def field_agent_page(request: Request):
    return templates.TemplateResponse("field_agent.html", {"request": request})

# Gemini/Heuristic Risk
def heuristic_risk(t):
    risk = 0.2
    if "ambulance" in (t.get("priority_reason") or "").lower(): risk += 0.4
    if (t.get("category") or "").lower() in ["bridge","pipe_leak","patient_visit"]: risk += 0.2
    if (t.get("department") or "").lower() in ["health","pwd"]: risk += 0.1
    if t.get("citizen_impact", 50) >= 70: risk += 0.1
    return min(1.0, risk)

def gemini_risk(t):
    if not gemini_model: return heuristic_risk(t), "Heuristic fallback"
    try:
        prompt = f"Give a JSON with risk_0_to_1 and reason for this task:\n{json.dumps(t)}"
        resp = gemini_model.generate_content(prompt)
        text = resp.text.strip()
        data = json.loads(text) if text.startswith("{") else {}
        r = float(data.get("risk_0_to_1", 0.5))
        return min(max(r, 0), 1), data.get("reason", "Gemini analysis")
    except Exception:
        return heuristic_risk(t), "Heuristic fallback"

def dps_calc(t):
    base = t.get("base_urgency", 50)
    risk = t.get("predictive_risk", 0.3)*100
    impact = t.get("citizen_impact", 50)
    pressure = 0
    try:
        created = dt.datetime.fromisoformat(t["created_at"].replace("Z",""))
        due = dt.datetime.fromisoformat(t["due_at"].replace("Z",""))
        total = (due - created).total_seconds()
        spent = (dt.datetime.utcnow() - created).total_seconds()
        pressure = min(100, max(0, (spent/total)*100))
    except: pass
    return round(0.4*base + 0.3*risk + 0.2*impact + 0.1*pressure,1)

@app.get("/api/tasks")
def list_tasks():
    rows = read_local()
    # ensure list format
    if isinstance(rows, dict):
        rows = rows.get("tasks", [])

    # Sort by created_at (newest first), then by DPS score
    def sort_key(x):
        try:
            created = dt.datetime.fromisoformat(x.get("created_at", "").replace("Z", ""))
        except Exception:
            created = dt.datetime.min
        return (created, x.get("dps_score", 0))

    # sort newest first, higher DPS first
    rows.sort(key=sort_key, reverse=True)

    return {"tasks": rows}


@app.post("/api/tasks/{task_id}")
def update_task(task_id:str, req:Request):
    rows = read_local()
    data = [r for r in rows if r["id"]==task_id]
    if not data: raise HTTPException(404)
    task = data[0]
    task["status"]="Completed"
    write_local(rows)
    return {"ok":True}

@app.post("/api/dps/recompute")
def recompute_all():
    rows = read_local()
    for t in rows:
        r,_ = gemini_risk(t)
        t["predictive_risk"]=r
        t["dps_score"]=dps_calc(t)
    write_local(rows)
    return {"ok":True,"updated":len(rows)}

@app.get("/climate-risk", response_class=HTMLResponse)
def climate_risk(request: Request):
    return templates.TemplateResponse("climate_risk.html", {"request": request})

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

# ---------------------------------------------------------------------
# Run locally
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
