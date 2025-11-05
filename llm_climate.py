import google.generativeai as genai
import os
from typing import Dict, List
import json

class ClimateRiskSimulator:
    def __init__(self, api_key: str = None):
        """Initialize the Climate Risk Simulator with Gemini"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        # Mock data for hyper-localized analysis
        self.regional_data = {
            "mumbai": {
                "nutritional_vulnerability": "High coastal population with 35% anemia rates in fishing communities",
                "environmental_triggers": "Coastal flooding, air pollution (PM2.5: 85μg/m³), water contamination",
                "cascading_risks": "Flood → Power outage → Hospital shutdown → Disease outbreak"
            },
            "pune": {
                "nutritional_vulnerability": "Urban poor with 28% malnutrition, migrant labor food insecurity",
                "environmental_triggers": "Urban heat island (+4°C), water scarcity, construction pollution",
                "cascading_risks": "Heatwave → Energy demand surge → Grid failure → Water pump shutdown"
            },
            "nagpur": {
                "nutritional_vulnerability": "Tribal areas with 42% child malnutrition, seasonal food shortages",
                "environmental_triggers": "Extreme heat (48°C), drought, agricultural distress",
                "cascading_risks": "Drought → Crop failure → Economic stress → Health system overload"
            },
            "nashik": {
                "nutritional_vulnerability": "Rural farming communities with variable food access",
                "environmental_triggers": "Erratic rainfall, pesticide runoff, groundwater depletion",
                "cascading_risks": "Flood → Road damage → Supply chain disruption → Price inflation"
            }
        }
    
    def analyze_climate_risk(self, location: str, project_type: str) -> str:
        """
        Analyze climate risk for a given location and project type
        
        Args:
            location: Area name (e.g., "Mumbai", "Pune coastal area")
            project_type: "health" or "infrastructure"
        
        Returns:
            Polished risk assessment with probabilities and recommendations
        """
        base_prompt = self._create_system_prompt()
        user_query = f"Location: {location}\nProject Type: {project_type}"
        
        if not self.api_key:
            return self._get_fallback_response(location, project_type)
        
        try:
            model = genai.GenerativeModel("gemini-pro")
            full_prompt = f"{base_prompt}\n\n{user_query}"
            response = model.generate_content(full_prompt)
            
            if response.text:
                return self._format_response(response.text)
            else:
                raise ValueError("Empty response from AI service")
                
        except Exception as e:
            print(f"Climate risk analysis error: {e}")
            return self._get_fallback_response(location, project_type)
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for climate risk analysis"""
        return """You are Climate Risk Simulator AI - an expert system for hyper-localized climate risk assessment in Maharashtra.

CRITICAL ANALYSIS FRAMEWORK - ALWAYS INCLUDE:

1. NUTRITIONAL VULNERABILITY MAPPING:
   - Analyze local food security, malnutrition rates, and climate impacts on nutrition
   - Consider seasonal variations and vulnerable populations

2. ENVIRONMENTAL HEALTH TRIGGERS:
   - Identify specific pollutants, disease vectors, and environmental stressors
   - Monitor air/water quality, temperature extremes, and disease outbreaks

3. CASCADING FAILURE SIMULATION:
   - Model chain reactions: Climate event → Infrastructure failure → Health impact
   - Identify critical system interdependencies and single points of failure

RESPONSE FORMAT - FOLLOW EXACTLY:

🌍 CLIMATE RISK ASSESSMENT: [Location] - [Project Type]

📍 RISK CATEGORY 1: [Category Name]
   📊 Probability: XX%
   ⚠️ Impact: [1-2 line explanation linking to nutritional/environmental/cascading risks]
   🛡️ Mitigation: • [Action 1] • [Action 2]

📍 RISK CATEGORY 2: [Category Name]  
   📊 Probability: XX%
   ⚠️ Impact: [1-2 line explanation]
   🛡️ Mitigation: • [Action 1] • [Action 2]

📍 RISK CATEGORY 3: [Category Name]
   📊 Probability: XX%
   ⚠️ Impact: [1-2 line explanation]
   🛡️ Mitigation: • [Action 1] • [Action 2]

💡 KEY VULNERABILITIES IDENTIFIED:
   • [Nutritional vulnerability insight]
   • [Environmental trigger concern] 
   • [Cascading failure pathway]

Always provide 3 risk categories. Use Maharashtra-specific context. Be precise with probabilities based on realistic climate models."""
    
    def _format_response(self, response_text: str) -> str:
        """Format the AI response for consistency"""
        # Clean up any markdown and ensure consistent formatting
        lines = response_text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.replace('**', '').strip()
            if line and not line.startswith('```'):
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _get_fallback_response(self, location: str, project_type: str) -> str:
        """Provide fallback response when API is unavailable"""
        location_key = location.lower().split()[0]  # Take first word for matching
        
        regional_info = self.regional_data.get(location_key, {
            "nutritional_vulnerability": "Moderate nutritional vulnerability in mixed population",
            "environmental_triggers": "Seasonal climate extremes and pollution concerns",
            "cascading_risks": "Infrastructure strain during climate events"
        })
        
        if project_type.lower() == "health":
            return f"""🌍 CLIMATE RISK ASSESSMENT: {location.title()} - Health Infrastructure

📍 RISK CATEGORY 1: Disease Outbreak Amplification
   📊 Probability: 65%
   ⚠️ Impact: Climate events strain healthcare while increasing vector-borne diseases
   🛡️ Mitigation: • Strengthen disease surveillance • Pre-position medical supplies

📍 RISK CATEGORY 2: Healthcare Access Disruption  
   📊 Probability: 58%
   ⚠️ Impact: Extreme weather blocks transport routes to health facilities
   🛡️ Mitigation: • Mobile medical units • Telemedicine infrastructure

📍 RISK CATEGORY 3: Nutritional Stress on Health System
   📊 Probability: 52%
   ⚠️ Impact: Climate affects food security, increasing malnutrition cases
   🛡️ Mitigation: • Integrated nutrition programs • Community kitchen planning

💡 KEY VULNERABILITIES IDENTIFIED:
   • {regional_info['nutritional_vulnerability']}
   • {regional_info['environmental_triggers']}
   • {regional_info['cascading_risks']}"""
        
        else:  # Infrastructure projects
            return f"""🌍 CLIMATE RISK ASSESSMENT: {location.title()} - Infrastructure

📍 RISK CATEGORY 1: Structural Resilience Failure
   📊 Probability: 72%
   ⚠️ Impact: Climate extremes exceed design limits of buildings and transport
   🛡️ Mitigation: • Climate-resilient materials • Elevated foundations

📍 RISK CATEGORY 2: Utility Service Disruption
   📊 Probability: 68%  
   ⚠️ Impact: Power, water, and communication networks vulnerable to climate shocks
   🛡️ Mitigation: • Redundant systems • Distributed energy resources

📍 RISK CATEGORY 3: Supply Chain Cascading Failure
   📊 Probability: 61%
   ⚠️ Impact: Climate events disrupt material supply and construction timelines
   🛡️ Mitigation: • Local material sourcing • Strategic stockpiling

💡 KEY VULNERABILITIES IDENTIFIED:
   • {regional_info['nutritional_vulnerability']}
   • {regional_info['environmental_triggers']}
   • {regional_info['cascading_risks']}"""

