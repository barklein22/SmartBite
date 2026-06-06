# SmartBite – Gemini AI Restaurant Agent

SmartBite is an AI restaurant agent for Israeli restaurants.

Main capabilities:
- Natural conversation with Gemini API.
- Restaurant-domain boundary: answers only restaurant/food-place questions.
- Internal restaurant data first.
- Google Places fallback when needed.
- TF-IDF for keyword-based matching.
- Lightweight semantic embeddings layer for phrase expansion.
- Cosine Similarity for similar restaurant recommendations.
- Isolation Forest anomaly detection for unusual restaurant patterns.
- RTL chat UI with long text wrapping.

Required Render environment variables:
- GEMINI_API_KEY
- GOOGLE_PLACES_API_KEY
