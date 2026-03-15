"""
generate_demo_scenarios.py
Creates 5 realistic demo scenario JSON files for the dashboard.
Each file matches the DashboardData schema used by the React frontend.

Output: dashboard/public/scenarios/scenario_{1-5}.json
        dashboard/public/scenarios/index.json  (metadata list)

Run from: early_warning_pipeline/
    python generate_demo_scenarios.py
"""

import json, os, math, random
from datetime import datetime, timezone

random.seed(42)

BASE     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.normpath(os.path.join(BASE, '..', 'dashboard', 'public', 'scenarios'))
os.makedirs(OUT_DIR, exist_ok=True)

NOW = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ── Botswana / Zimbabwe study-area anchors ────────────────────────────────────
# Chobe / Hwange region
PARK_CENTER   = (-18.60, 26.20)   # deep park, safe
CHOBE_RIVER   = (-17.80, 25.10)   # Chobe river corridor
HWANGE_EAST   = (-19.20, 27.00)   # Hwange park eastern edge
FARM_BELT     = (-19.80, 26.50)   # farmland south of park
VILLAGE_KASANE= (-17.82, 25.14)   # Kasane town
VILLAGE_NATA  = (-20.21, 26.17)   # Nata village
VILLAGE_MAUN  = (-19.98, 23.42)   # Maun
CONCESSION_N  = (-18.10, 26.60)   # NG concession north
CONCESSION_S  = (-18.85, 26.80)   # NG concession south
BOUNDARY_LINE = (-18.50, 27.20)   # park-concession boundary

RANK_COLORS = ["#FF3B30", "#FF9500", "#FFCC00", "#34C759", "#007AFF"]

def fmt(v: float, nd=4) -> float:
    return round(v, nd)

def jitter(lat, lon, km_radius=5.0):
    """Random offset within km_radius km."""
    deg = km_radius / 111.0
    return (fmt(lat + random.uniform(-deg, deg)),
            fmt(lon + random.uniform(-deg, deg)))

def grid_id(lat, lon):
    row = int((lat - (-17.0)) / -0.04488)
    col = int((lon - 23.0) / 0.04488)
    return f"R{max(row,0):04d}_C{max(col,0):04d}"

def movement_trace(lat, lon, steps=24, max_step_km=1.5, label_prefix=""):
    """Generate synthetic hourly movement data."""
    trace = []
    cur_lat, cur_lon = lat, lon
    heading = random.uniform(0, 2 * math.pi)
    for i in range(steps):
        step_km = random.uniform(0, max_step_km)
        heading += random.uniform(-0.5, 0.5)
        dlat = step_km * math.cos(heading) / 111.0
        dlon = step_km * math.sin(heading) / (111.0 * max(math.cos(math.radians(cur_lat)), 0.01))
        cur_lat += dlat
        cur_lon += dlon
        hour = f"{(i % 24):02d}:00"
        trace.append({"hour": hour, "distance": fmt(step_km, 3)})
    return trace

def history_path(lat, lon, steps=20, max_step_km=2.0):
    """Dead-reckoned path list [[lat, lon], ...]."""
    coords = [[fmt(lat), fmt(lon)]]
    heading = random.uniform(0, 2 * math.pi)
    for _ in range(steps):
        step_km = random.uniform(0, max_step_km)
        heading += random.uniform(-0.6, 0.6)
        dlat = step_km * math.cos(heading) / 111.0
        dlon = step_km * math.sin(heading) / (111.0 * max(math.cos(math.radians(coords[-1][0])), 0.01))
        coords.append([fmt(coords[-1][0] + dlat), fmt(coords[-1][1] + dlon)])
    return coords

def predictions(cur_lat, cur_lon, n=5, max_dist_km=80, high_conf_idx=None):
    """Generate n ranked prediction entries."""
    preds = []
    confs = sorted([random.uniform(5, 40) for _ in range(n)], reverse=True)
    if high_conf_idx == 0:
        confs[0] = random.uniform(70, 99)
    total = sum(confs)
    confs = [round(c / total * 100, 2) for c in confs]
    for rank in range(1, n + 1):
        dist_km = random.uniform(5, max_dist_km)
        angle   = random.uniform(0, 2 * math.pi)
        plat = fmt(cur_lat + (dist_km / 111.0) * math.cos(angle))
        plon = fmt(cur_lon + (dist_km / 111.0) * math.sin(angle) / max(math.cos(math.radians(cur_lat)), 0.01))
        preds.append({
            "rank":        rank,
            "gridCell":    grid_id(plat, plon),
            "confidence":  confs[rank - 1],
            "location":    {"lat": plat, "lng": plon},
            "distanceKm":  fmt(dist_km, 1),
        })
    return preds

def make_alert(eid, eid_str, typ, sev, msg):
    return {"id": f"a_{eid_str}", "timestamp": NOW, "type": typ, "severity": sev, "message": msg, "elephantId": eid_str}

def build_scenario(scenario_id: int, title: str, description: str, eleph_defs: list) -> dict:
    """
    eleph_defs: list of dicts with keys:
        id, name, lat, lon, status, alert_type, alert_severity, alert_msg,
        villages (list of village dicts), high_conf (bool), max_step_km
    """
    elephants, preds_map, villages_map, movement_map, history_map, alerts = [], {}, {}, {}, {}, []

    for e in eleph_defs:
        eid_str = f"E-{e['id']}"
        lat, lon = e["lat"], e["lon"]
        gid = grid_id(lat, lon)

        elephants.append({
            "id":          eid_str,
            "name":        e["name"],
            "gridCell":    gid,
            "position":    {"lat": lat, "lng": lon},
            "horizon":     48,
            "model":       "LSTM (PyTorch)",
            "lastUpdated": NOW,
            "status":      e["status"],
        })
        preds_map[eid_str]    = predictions(lat, lon, n=5, max_dist_km=e.get("max_dist_km", 60),
                                             high_conf_idx=0 if e.get("high_conf") else None)
        villages_map[eid_str] = e.get("villages", [])
        movement_map[eid_str] = movement_trace(lat, lon, max_step_km=e.get("max_step_km", 1.2))
        history_map[eid_str]  = history_path(lat, lon, max_step_km=e.get("max_step_km", 1.2))

        if e.get("alert_msg"):
            alerts.append(make_alert(e["id"], eid_str, e.get("alert_type","prediction"),
                                     e.get("alert_severity","low"), e["alert_msg"]))

    return {
        "scenarioId":    scenario_id,
        "title":         title,
        "description":   description,
        "generatedAt":   NOW,
        "elephants":     elephants,
        "predictionsMap": preds_map,
        "villagesMap":   villages_map,
        "alertEvents":   alerts,
        "movementMap":   movement_map,
        "historyMap":    history_map,
    }


# ════════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — All Clear (8 elephants, all safe, deep park)
# ════════════════════════════════════════════════════════════════════════════════
SC1_ELEPHANTS = [
    {"id": 1,  "name": "Kibo",    "lat": -18.42, "lon": 26.15, "status": "safe",  "high_conf": True,  "max_step_km": 1.0,
     "alert_msg": "Kibo — all clear, predicted to stay within park core"},
    {"id": 2,  "name": "Tembo",   "lat": -18.60, "lon": 25.90, "status": "safe",  "high_conf": False, "max_step_km": 0.8,
     "alert_msg": "Tembo — resting near Chobe river, low movement"},
    {"id": 3,  "name": "Zuri",    "lat": -18.80, "lon": 26.40, "status": "safe",  "high_conf": True,  "max_step_km": 1.2,
     "alert_msg": "Zuri — moving northeast within park"},
    {"id": 4,  "name": "Mara",    "lat": -18.20, "lon": 26.70, "status": "safe",  "high_conf": False, "max_step_km": 0.7,
     "alert_msg": "Mara — grazing, stationary for 3h"},
    {"id": 5,  "name": "Jabali",  "lat": -19.10, "lon": 26.00, "status": "safe",  "high_conf": True,  "max_step_km": 1.5,
     "alert_msg": "Jabali — heading towards water source"},
    {"id": 6,  "name": "Amani",   "lat": -18.55, "lon": 27.10, "status": "safe",  "high_conf": False, "max_step_km": 0.9,
     "alert_msg": "Amani — part of herd grouping, safe zone"},
    {"id": 7,  "name": "Duma",    "lat": -19.30, "lon": 26.80, "status": "safe",  "high_conf": True,  "max_step_km": 1.1,
     "alert_msg": "Duma — predicted to continue east, no threats"},
    {"id": 8,  "name": "Nyota",   "lat": -18.95, "lon": 25.60, "status": "safe",  "high_conf": False, "max_step_km": 0.6,
     "alert_msg": "Nyota — juvenile, tracked with herd"},
]

# ════════════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — Crop Raid Alert (7 elephants, 3 danger heading to farms)
# ════════════════════════════════════════════════════════════════════════════════
SC2_ELEPHANTS = [
    {"id": 1,  "name": "Kibo",    "lat": -19.52, "lon": 26.30, "status": "danger",  "high_conf": True,  "max_step_km": 3.5, "max_dist_km": 30,
     "alert_type": "proximity", "alert_severity": "high",
     "alert_msg": "⚠️ Kibo crossing into maize field near Nata — immediate action needed",
     "villages": [{"name": "Nata Farmlands", "distanceKm": 1.2, "population": 320, "atRisk": True},
                  {"name": "Nata Village",   "distanceKm": 3.8, "population": 870, "atRisk": False}]},
    {"id": 2,  "name": "Tembo",   "lat": -19.65, "lon": 26.10, "status": "danger",  "high_conf": True,  "max_step_km": 4.0, "max_dist_km": 25,
     "alert_type": "boundary", "alert_severity": "high",
     "alert_msg": "⚠️ Tembo inside crop zone — sorghum fields at risk",
     "villages": [{"name": "Gweta Farms",  "distanceKm": 0.8, "population": 210, "atRisk": True}]},
    {"id": 3,  "name": "Zuri",    "lat": -19.40, "lon": 26.55, "status": "warning", "high_conf": True,  "max_step_km": 2.8, "max_dist_km": 40,
     "alert_type": "movement", "alert_severity": "medium",
     "alert_msg": "Zuri approaching farmland corridor — velocity spike 8.2 km/h",
     "villages": [{"name": "Mmathata Settlement", "distanceKm": 4.5, "population": 440, "atRisk": False}]},
    {"id": 4,  "name": "Mara",    "lat": -19.75, "lon": 26.80, "status": "warning", "high_conf": False, "max_step_km": 2.2, "max_dist_km": 50,
     "alert_type": "movement", "alert_severity": "medium",
     "alert_msg": "Mara bearing toward sorghum belt, predicted arrival <6h"},
    {"id": 5,  "name": "Jabali",  "lat": -18.80, "lon": 26.20, "status": "safe",    "high_conf": False, "max_step_km": 1.0,
     "alert_msg": "Jabali — stationary in park interior, no threat"},
    {"id": 6,  "name": "Amani",   "lat": -18.60, "lon": 26.70, "status": "safe",    "high_conf": True,  "max_step_km": 0.9,
     "alert_msg": "Amani — safe, moving away from farm belt"},
    {"id": 7,  "name": "Duma",    "lat": -19.20, "lon": 27.20, "status": "warning", "high_conf": False, "max_step_km": 2.0,
     "alert_type": "prediction", "alert_severity": "medium",
     "alert_msg": "Duma — 3rd predicted zone overlaps cotton farm boundary"},
]

# ════════════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — Village Proximity (6 elephants, 2 dangerously close to settlements)
# ════════════════════════════════════════════════════════════════════════════════
SC3_ELEPHANTS = [
    {"id": 1,  "name": "Kibo",    "lat": -17.87, "lon": 25.18, "status": "danger",  "high_conf": True,  "max_step_km": 2.5, "max_dist_km": 15,
     "alert_type": "proximity", "alert_severity": "high",
     "alert_msg": "🚨 Kibo 0.9 km from Kasane residential — human-elephant conflict imminent",
     "villages": [{"name": "Kasane Town",    "distanceKm": 0.9, "population": 12000, "atRisk": True},
                  {"name": "Kasane Ext.",    "distanceKm": 1.4, "population": 3400,  "atRisk": True}]},
    {"id": 2,  "name": "Tembo",   "lat": -19.95, "lon": 23.38, "status": "danger",  "high_conf": True,  "max_step_km": 2.0, "max_dist_km": 20,
     "alert_type": "proximity", "alert_severity": "high",
     "alert_msg": "🚨 Tembo 1.2 km from Maun outskirts — children's school route at risk",
     "villages": [{"name": "Maun Outskirts", "distanceKm": 1.2, "population": 8500, "atRisk": True}]},
    {"id": 3,  "name": "Zuri",    "lat": -20.18, "lon": 26.12, "status": "warning", "high_conf": False, "max_step_km": 1.8, "max_dist_km": 35,
     "alert_type": "proximity", "alert_severity": "medium",
     "alert_msg": "Zuri 3.1 km from Nata Village, monitoring",
     "villages": [{"name": "Nata Village", "distanceKm": 3.1, "population": 870, "atRisk": False}]},
    {"id": 4,  "name": "Mara",    "lat": -18.70, "lon": 26.50, "status": "safe",    "high_conf": False, "max_step_km": 0.8,
     "alert_msg": "Mara — deep park, no proximity concern"},
    {"id": 5,  "name": "Jabali",  "lat": -18.30, "lon": 25.80, "status": "safe",    "high_conf": True,  "max_step_km": 1.1,
     "alert_msg": "Jabali — well within Chobe park, safe"},
    {"id": 6,  "name": "Amani",   "lat": -19.45, "lon": 26.30, "status": "warning", "high_conf": False, "max_step_km": 1.6,
     "alert_type": "prediction", "alert_severity": "medium",
     "alert_msg": "Amani — predicted path passes 4km from Makalamabedi"},
]

# ════════════════════════════════════════════════════════════════════════════════
# SCENARIO 4 — Poaching Threat / Under Attack (5 elephants, erratic movement, all danger)
# ════════════════════════════════════════════════════════════════════════════════
SC4_ELEPHANTS = [
    {"id": 1,  "name": "Kibo",    "lat": -19.05, "lon": 27.35, "status": "danger",  "high_conf": False, "max_step_km": 5.0, "max_dist_km": 80,
     "alert_type": "movement", "alert_severity": "high",
     "alert_msg": "🚨 Kibo — erratic movement detected, 14.7 km/h burst, possible disturbance",
     "villages": []},
    {"id": 2,  "name": "Tembo",   "lat": -19.12, "lon": 27.42, "status": "danger",  "high_conf": False, "max_step_km": 6.0, "max_dist_km": 90,
     "alert_type": "movement", "alert_severity": "high",
     "alert_msg": "🚨 Tembo — rapid directional change, stampede pattern detected",
     "villages": []},
    {"id": 3,  "name": "Zuri",    "lat": -18.98, "lon": 27.28, "status": "danger",  "high_conf": False, "max_step_km": 5.5, "max_dist_km": 85,
     "alert_type": "boundary", "alert_severity": "high",
     "alert_msg": "🚨 Zuri — fleeing northeast, crossed 3 grid boundaries in 2h"},
    {"id": 4,  "name": "Mara",    "lat": -19.18, "lon": 27.55, "status": "danger",  "high_conf": False, "max_step_km": 7.0, "max_dist_km": 100,
     "alert_type": "movement", "alert_severity": "high",
     "alert_msg": "🚨 Mara — highest velocity spike ever recorded (19 km/h), possible snare stress"},
    {"id": 5,  "name": "Jabali",  "lat": -19.08, "lon": 27.48, "status": "danger",  "high_conf": False, "max_step_km": 4.5, "max_dist_km": 70,
     "alert_type": "movement", "alert_severity": "high",
     "alert_msg": "🚨 Jabali — erratic spin-back movement, anti-poaching rangers alerted"},
]

# ════════════════════════════════════════════════════════════════════════════════
# SCENARIO 5 — Boundary / Concession Crossing (9 elephants, 4 crossing)
# ════════════════════════════════════════════════════════════════════════════════
SC5_ELEPHANTS = [
    {"id": 1,  "name": "Kibo",    "lat": -18.48, "lon": 27.18, "status": "danger",  "high_conf": True,  "max_step_km": 2.5, "max_dist_km": 30,
     "alert_type": "boundary", "alert_severity": "high",
     "alert_msg": "🚨 Kibo crossed into private concession NG/32 — ranger dispatch required"},
    {"id": 2,  "name": "Tembo",   "lat": -18.52, "lon": 27.24, "status": "danger",  "high_conf": True,  "max_step_km": 3.0, "max_dist_km": 35,
     "alert_type": "boundary", "alert_severity": "high",
     "alert_msg": "🚨 Tembo — inside NG/32 concession, trophy hunting area risk"},
    {"id": 3,  "name": "Zuri",    "lat": -18.45, "lon": 27.30, "status": "warning", "high_conf": True,  "max_step_km": 2.0, "max_dist_km": 40,
     "alert_type": "boundary", "alert_severity": "medium",
     "alert_msg": "Zuri — 0.8 km from NG/32 boundary, monitoring"},
    {"id": 4,  "name": "Mara",    "lat": -18.40, "lon": 27.22, "status": "warning", "high_conf": False, "max_step_km": 1.5,
     "alert_type": "boundary", "alert_severity": "medium",
     "alert_msg": "Mara — predicted to reach concession boundary within 12h"},
    {"id": 5,  "name": "Jabali",  "lat": -18.88, "lon": 27.05, "status": "safe",    "high_conf": False, "max_step_km": 1.0,
     "alert_msg": "Jabali — well inside park, no boundary risk"},
    {"id": 6,  "name": "Amani",   "lat": -19.15, "lon": 26.80, "status": "safe",    "high_conf": True,  "max_step_km": 0.8,
     "alert_msg": "Amani — moving away from boundary, safe"},
    {"id": 7,  "name": "Duma",    "lat": -18.20, "lon": 26.90, "status": "safe",    "high_conf": False, "max_step_km": 1.2,
     "alert_msg": "Duma — well within park, low prediction confidence"},
    {"id": 8,  "name": "Nyota",   "lat": -18.62, "lon": 26.45, "status": "safe",    "high_conf": True,  "max_step_km": 0.7,
     "alert_msg": "Nyota — juvenile, tracked with herd, safe"},
    {"id": 9,  "name": "Simba",   "lat": -18.58, "lon": 27.08, "status": "danger",  "high_conf": True,  "max_step_km": 2.8, "max_dist_km": 30,
     "alert_type": "boundary", "alert_severity": "high",
     "alert_msg": "🚨 Simba — second border crossing in 24h, possible territorial shift"},
]

# ── Generate all scenarios ─────────────────────────────────────────────────────
scenarios = [
    build_scenario(1, "🟢 All Clear",           "All elephants safe within park core. Normal movement patterns, no threats detected.", SC1_ELEPHANTS),
    build_scenario(2, "🌾 Crop Raid Alert",     "3 elephants confirmed inside farm zones. Immediate human-elephant conflict risk near Nata.", SC2_ELEPHANTS),
    build_scenario(3, "🏘️ Village Proximity",   "2 elephants within 1.5 km of settlements. Kasane and Maun outskirts at risk.", SC3_ELEPHANTS),
    build_scenario(4, "🚨 Poaching Threat",     "5 elephants showing extreme erratic movement. Possible snare/poaching activity detected in Hwange east.", SC4_ELEPHANTS),
    build_scenario(5, "⚠️ Boundary Crossing",   "4 elephants have crossed or approaching the NG/32 concession boundary. Ranger dispatch in progress.", SC5_ELEPHANTS),
]

# Write individual files
for sc in scenarios:
    path = os.path.join(OUT_DIR, f"scenario_{sc['scenarioId']}.json")
    with open(path, 'w') as f:
        json.dump(sc, f, indent=2)
    print(f"[OK] {path}")

# Write index
index = [{"id": sc["scenarioId"], "title": sc["title"], "description": sc["description"], "file": f"scenario_{sc['scenarioId']}.json"} for sc in scenarios]
idx_path = os.path.join(OUT_DIR, "index.json")
with open(idx_path, 'w') as f:
    json.dump(index, f, indent=2)
print(f"[OK] {idx_path}")
print(f"\nDone. {len(scenarios)} demo scenarios written to {OUT_DIR}")
