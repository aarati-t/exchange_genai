import os
from typing import Any, Dict, List, Optional

from google.cloud import bigquery
import gspread
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2 import service_account

BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID")
BQ_KEY_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    r"C:\Users\Indian\Downloads\migis-prototype-27d09631dfe4.json"
)

def try_bigquery_query(sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Query BigQuery if available; otherwise return a small demo stub."""
    try:
        credentials = service_account.Credentials.from_service_account_file(BQ_KEY_PATH)
    
        # Initialize BigQuery client explicitly using credentials
        client = bigquery.Client(credentials=credentials, project=BQ_PROJECT_ID)
        # TODO - on google app engine use below code
        
        #client = bigquery.Client(project=BQ_PROJECT_ID)
        job = client.query(sql)
        rows = [dict(row) for row in job.result()]
        return rows
    except Exception:
        # Demo fallback
        return [
            {"year": 2025, "traditional_cost": 100, "sustainable_cost": 92, "co2_tons": 12000},
            {"year": 2030, "traditional_cost": 140, "sustainable_cost": 120, "co2_tons": 14000},
            {"year": 2040, "traditional_cost": 220, "sustainable_cost": 170, "co2_tons": 16000},
        ]

def try_sheets_read(sheet_id: str, tab_name: str = "Sheet1") -> List[Dict[str, Any]]:
    """Read a Google Sheet if available; otherwise return a demo stub."""
    try:
        gc = gspread.service_account()
        sh = gc.open_by_key(sheet_id)
        ws = sh.worksheet(tab_name)
        rows = ws.get_all_records()
        return rows
    except (DefaultCredentialsError, Exception):
        # Demo fallback
        return [
            {"kpi": "Resilience Score", "value": 72},
            {"kpi": "Lifecycle Savings (₹Cr)", "value": 48},
            {"kpi": "Water Savings (%)", "value": 60},
        ]
