SAMPLE_SCENARIOS = {
    "scenarios": [
        {
            "id": "mumbai_coastal_road",
            "title": "Scenario 1: Mumbai Coastal Road Extension",
            "district": "Mumbai",
            "risk_label": "High-Risk Zone",
            "risk_score": 0.85,  # 85%
            "flags": [
                "2050-level storm surges",
                "Sea-level rise inundation"
            ],
            "proposed": {
                "summary": "Coastal road alignment along western shoreline.",
                "materials": ["Traditional reinforced concrete & steel"],
            },
            "recommended": {
                "summary": "Shift 200m inland, titanium-reinforced concrete for pillars, wave-dissipation structures.",
                "benefits": {
                    "risk_reduction_pct": 70,
                    "extra_lifespan_years": 25
                }
            }
        },
        {
            "id": "pune_metro_phase_3",
            "title": "Scenario 2: Pune Metro Phase 3 Construction",
            "district": "Pune",
            "risk_label": "High Embodied Carbon & Heat",
            "risk_score": 0.78,
            "flags": [
                "Urban heat island effect",
                "High embodied carbon (12,000 tCO₂)"
            ],
            "proposed": {
                "summary": "Traditional concrete & steel for elevated sections.",
                "materials": ["Conventional concrete", "Conventional steel"]
            },
            "recommended": {
                "summary": "40% fly-ash concrete, recycled steel, solar-reflective coatings.",
                "benefits": {
                    "co2_reduction_pct": 45,
                    "temp_reduction_c": 3
                }
            }
        },
        {
            "id": "satara_school_complex",
            "title": "Scenario 9: Satara District School Complex",
            "district": "Satara",
            "risk_label": "Thermal Discomfort & Opex",
            "risk_score": 0.66,
            "flags": [
                "Poor thermal comfort during heatwaves",
                "High long-term maintenance costs"
            ],
            "proposed": {
                "summary": "Conventional construction for 50 new schools.",
                "materials": ["Standard brick & mortar"]
            },
            "recommended": {
                "summary": "Passive cooling, rainwater harvesting, local sustainable materials.",
                "benefits": {
                    "maintenance_saving_20y_pct": 40,
                    "water_use_reduction_pct": 60
                }
            }
        }
    ]
}

# Simple demo GeoJSON layers for flood/heat/biodiversity (2030/2050/2100)
SAMPLE_LAYERS = {
    "2030": {
        "flood": {"type": "FeatureCollection", "features": []},
        "heat": {"type": "FeatureCollection", "features": []},
        "biodiv": {"type": "FeatureCollection", "features": []}
    },
    "2050": {
        "flood": {"type": "FeatureCollection", "features": []},
        "heat": {"type": "FeatureCollection", "features": []},
        "biodiv": {"type": "FeatureCollection", "features": []}
    },
    "2100": {
        "flood": {"type": "FeatureCollection", "features": []},
        "heat": {"type": "FeatureCollection", "features": []},
        "biodiv": {"type": "FeatureCollection", "features": []}
    }
}
