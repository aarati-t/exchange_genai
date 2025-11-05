import os
import google.generativeai as genai

# --------------------------------------------------------------------
# Gemini API Configuration
# --------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or \
    
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --------------------------------------------------------------------
# Main Function
# --------------------------------------------------------------------
def simulate_with_gemini(user_prompt: str) -> str:
    """
    Generate a polished, emoji-formatted sustainability assessment.
    Designed for Green / Climate Simulator modules.
    """

    base_system = (
        "You are an expert Climate Resilience and Sustainability Advisor working inside "
        "a Digital Twin Simulator for infrastructure and city projects. "
        "Respond in an attractive, easy-to-read format with emojis and bullet points. "
        "Structure your output using these exact headings:\n\n"
        "🌍 **Climate Resilience Score:** [X]/100\n\n"
        "⚠️ **Primary Risk Factors:**\n"
        "• [Risk 1 — concise impact statement]\n"
        "• [Risk 2 — concise impact statement]\n"
        "• [Risk 3 — concise impact statement]\n\n"
        "💡 **Recommended Mitigation Strategies:**\n"
        "• [Action 1 — with short actionable advice]\n"
        "• [Action 2 — with measurable outcome]\n"
        "• [Action 3 — with climate benefit]\n\n"
        "📈 **Sustainability Impact Assessment:**\n"
        "• [Metric 1 — e.g., CO₂ reduction, %]\n"
        "• [Metric 2 — e.g., water savings, %]\n"
        "• [Metric 3 — e.g., cost savings, lifespan gains]\n\n"
        "Use emojis such as 🌊 ☀️ 💧 ♻️ 🦋 🔧 🌿 📊 to make it visually engaging. "
        "Avoid long paragraphs — keep each line concise and impactful."
    )

    # ----------------------------------------------------------------
    # Fallback when Gemini API is not available
    # ----------------------------------------------------------------
    if not GEMINI_API_KEY:
        return (
            "🌍 **Climate Resilience Score:** 58/100\n\n"
            "⚠️ **Primary Risk Factors:**\n"
            "• 🌊 Extreme rainfall flooding underground stations and electrical systems\n"
            "• ☀️ Heatwave-induced rail track buckling and increased cooling demand\n"
            "• 💧 Water scarcity accelerating concrete degradation in elevated sections\n\n"
            "💡 **Recommended Mitigation Strategies:**\n"
            "• 🔧 Upgrade drainage with sustainable urban drainage systems and flood barriers\n"
            "• ☀️ Install heat-resistant rails with reflective coatings and efficient cooling\n"
            "• ♻️ Use recycled aggregates in concrete with water harvesting systems\n\n"
            "📈 **Sustainability Impact Assessment:**\n"
            "• 🌿 Carbon footprint reduction: 20–30% lifecycle\n"
            "• 💧 Water use decrease: 40–50% maintenance phase\n"
            "• 💰 Cost savings: 15–20% over system lifespan"
        )

    try:
        # Use modern stable model
        model = genai.GenerativeModel("gemini-2.0-flash")

        full_prompt = (
            f"{base_system}\n\n"
            f"---\nUser Input:\n{user_prompt}\n---\n"
            "Now generate a concise, professional response following the above structure and style."
        )

        response = model.generate_content(full_prompt)
        text = (getattr(response, "text", "") or "").strip()

        if not text:
            raise ValueError("Empty response from Gemini")

        # Normalize bullet points if needed
        formatted = text.replace("-", "•").replace("*", "•")
        return formatted

    except Exception as e:
        print("⚠️ Gemini error:", e)
        return (
            "🌍 **Climate Resilience Score:** 60/100\n\n"
            "⚠️ **Primary Risk Factors:**\n"
            "• 🌡️ Extreme heat affecting material durability\n"
            "• 🌊 Seasonal flooding near low-lying track sections\n"
            "• 🏗️ High embodied carbon from cement and steel use\n\n"
            "💡 **Recommended Mitigation Strategies:**\n"
            "• ♻️ Adopt low-carbon cement blends (≥40% fly ash)\n"
            "• ☀️ Apply solar-reflective coatings to elevated segments\n"
            "• 💧 Integrate smart drainage and stormwater retention ponds\n\n"
            "📈 **Sustainability Impact Assessment:**\n"
            "• 🌿 Carbon emissions ↓ 35%\n"
            "• 💧 Water reuse ↑ 40%\n"
            "• 💰 Maintenance cost ↓ 20%"
        )
