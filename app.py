from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

RESTAURANTS_CSV = os.path.join(DATA_DIR, "smartbite_12000_restaurants_hebrew.csv")
ORDERS_CSV = os.path.join(DATA_DIR, "orders.csv")
REVIEWS_CSV = os.path.join(DATA_DIR, "reviews.csv")
CUSTOMERS_CSV = os.path.join(DATA_DIR, "customers.csv")
MENU_ITEMS_CSV = os.path.join(DATA_DIR, "menu_items.csv")

app = Flask(__name__)

_cache: Dict[str, Any] = {}
_context: Dict[str, Any] = {
    "conversation_history": [],
    "last_restaurant_id": None,
    "last_restaurant_name": None,
    "last_params": {},
    "pending_question": None,
    "user_profile": {},
}

# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    value = str(value or "").lower().strip()
    value = value.replace("׳", "'").replace("’", "'").replace("`", "'")
    value = re.sub(r"[\"'״׳`.,:;!?()\[\]{}\-_/\\]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def detect_language(message: str) -> str:
    return "he" if re.search(r"[\u0590-\u05FF]", message or "") else "en"


def gemini_api_key() -> str:
    """Gemini API key from Google AI Studio. Keep it only in Render Environment."""
    return os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_AI_API_KEY", "")).strip()


def gemini_model_candidates() -> List[str]:
    configured = os.environ.get("GEMINI_MODEL", "").strip()
    models = [configured] if configured else []
    # Fast/free-tier friendly models first. If one is not available, the code tries the next.
    models += ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"]
    seen: List[str] = []
    for model in models:
        if model and model not in seen:
            seen.append(model)
    return seen


def technical_error_message(language: str = "he") -> str:
    if language == "he":
        return "מצטערת, יש כרגע תקלה טכנית בחיבור למערכת. נסי שוב בעוד רגע."
    return "Sorry, there is a temporary technical connection issue. Please try again in a moment."


def gemini_generate(system_prompt: str, user_payload: Any, temperature: float = 0.5, timeout: int = 12) -> str:
    """
    Calls Gemini through the REST API, so the project does not need a special SDK.
    Raises an exception only for a real technical problem; normal domain handling is done by Gemini.
    """
    api_key = gemini_api_key()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable")

    if isinstance(user_payload, str):
        user_text = user_payload
    else:
        user_text = json.dumps(user_payload, ensure_ascii=False)

    last_error = None
    for model in gemini_model_candidates():
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {
                "temperature": temperature,
                "topP": 0.9,
                "maxOutputTokens": 1200,
            },
        }
        try:
            response = requests.post(
                url,
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            if response.status_code >= 400:
                last_error = f"{response.status_code}: {response.text[:500]}"
                continue
            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                last_error = f"No candidates in Gemini response: {data}"
                continue
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(str(part.get("text", "")) for part in parts).strip()
            if text:
                return text
            last_error = f"Empty Gemini text response: {data}"
        except Exception as exc:
            last_error = str(exc)
            continue

    raise RuntimeError(f"Gemini request failed: {last_error}")


def remember_conversation(user_message: str, assistant_message: str) -> None:
    history = _context.setdefault("conversation_history", [])
    history.append({"user": user_message, "assistant": assistant_message})
    _context["conversation_history"] = history[-10:]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------


def load_csv_cached(key: str, path: str) -> pd.DataFrame:
    if key not in _cache:
        if not os.path.exists(path):
            _cache[key] = pd.DataFrame()
        else:
            df = pd.read_csv(path)
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].fillna("").astype(str)
            _cache[key] = df
    return _cache[key].copy()


def load_restaurants() -> pd.DataFrame:
    df = load_csv_cached("restaurants", RESTAURANTS_CSV)
    if not df.empty:
        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
        if "avg_price_per_person" in df.columns:
            df["avg_price_per_person"] = pd.to_numeric(df["avg_price_per_person"], errors="coerce").fillna(0)
    return df


def load_reviews() -> pd.DataFrame:
    df = load_csv_cached("reviews", REVIEWS_CSV)
    if not df.empty:
        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        if "review_date" in df.columns:
            df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    return df


def load_orders() -> pd.DataFrame:
    df = load_csv_cached("orders", ORDERS_CSV)
    if not df.empty and "order_datetime" in df.columns:
        df["order_datetime"] = pd.to_datetime(df["order_datetime"], errors="coerce")
    return df


def load_menu() -> pd.DataFrame:
    df = load_csv_cached("menu", MENU_ITEMS_CSV)
    if not df.empty and "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    if not df.empty and "popularity_score" in df.columns:
        df["popularity_score"] = pd.to_numeric(df["popularity_score"], errors="coerce").fillna(0)
    return df


# ---------------------------------------------------------------------
# Hebrew/English mappings for fallback extraction
# ---------------------------------------------------------------------

CITY_MAP = {
    "בתל אביב": "Tel Aviv", "תל אביב": "Tel Aviv", "תל אביב יפו": "Tel Aviv", "תא": "Tel Aviv", "ת א": "Tel Aviv", "בתא": "Tel Aviv", "בת א": "Tel Aviv",
    "בירושלים": "Jerusalem", "ירושלים": "Jerusalem",
    "בחיפה": "Haifa", "חיפה": "Haifa",
    "ברמת גן": "Ramat Gan", "רמת גן": "Ramat Gan",
    "בגבעתיים": "Givatayim", "גבעתיים": "Givatayim",
    "בחולון": "Holon", "חולון": "Holon",
    "בראשון לציון": "Rishon LeZion", "ראשון לציון": "Rishon LeZion",
    "בפתח תקווה": "Petah Tikva", "פתח תקווה": "Petah Tikva",
    "בהרצליה": "Herzliya", "הרצליה": "Herzliya",
    "בנתניה": "Netanya", "נתניה": "Netanya",
    "בכפר סבא": "Kfar Saba", "כפר סבא": "Kfar Saba",
    "ברעננה": "Ra'anana", "רעננה": "Ra'anana",
    "ברחובות": "Rehovot", "רחובות": "Rehovot",
    "במודיעין": "Modiin", "מודיעין": "Modiin",
    "באשדוד": "Ashdod", "אשדוד": "Ashdod",
    "באשקלון": "Ashkelon", "אשקלון": "Ashkelon",
    "בבאר שבע": "Beer Sheva", "באר שבע": "Beer Sheva",
    "באילת": "Eilat", "אילת": "Eilat",
    "בנצרת": "Nazareth", "נצרת": "Nazareth",
    "בעכו": "Acre", "עכו": "Acre",
    "בקיסריה": "Caesarea", "קיסריה": "Caesarea",
    "בזכרון יעקב": "Zichron Yaakov", "זכרון יעקב": "Zichron Yaakov",
    "בטבריה": "Tiberias", "טבריה": "Tiberias",
    "בראש פינה": "Rosh Pina", "ראש פינה": "Rosh Pina",
}

REGION_MAP = {
    "צפון": "North", "בצפון": "North",
    "מרכז": "Center", "במרכז": "Center",
    "דרום": "South", "בדרום": "South",
    "ירושלים": "Jerusalem", "באזור ירושלים": "Jerusalem",
}

CUISINE_MAP = {
    "איטלקי": "Italian", "איטלקית": "Italian", "פסטה": "Italian",
    "פיצה": "Pizza", "יפני": "Japanese", "יפנית": "Japanese", "סושי": "Japanese",
    "אסייתי": "Asian", "אסייתית": "Asian", "נודלס": "Asian",
    "המבורגר": "Burgers", "בורגר": "Burgers",
    "טבעוני": "Vegan", "טבעונית": "Vegan", "צמחוני": "Vegetarian", "צמחונית": "Vegetarian",
    "ישראלי": "Israeli", "ישראלית": "Israeli", "ים תיכוני": "Mediterranean",
    "ערבי": "Arab", "ערבית": "Arab", "יווני": "Greek", "יוונית": "Greek",
    "תאילנדי": "Thai", "דגים": "Seafood", "פירות ים": "Seafood", "סטייק": "Steakhouse",
    "קפה": "Cafe", "בית קפה": "Cafe", "ארוחת בוקר": "Breakfast",
}

RESTAURANT_ALIASES = {
    "מקדונלדס": ["מקדונלדס", "מקדונלד'ס", "מקדונלד׳ס", "mcdonalds", "mcdonald's"],
    "ארומה": ["ארומה", "aroma"],
    "קפה קפה": ["קפה קפה", "cafe cafe"],
    "גרג": ["גרג", "greg"],
    "לנדוור": ["לנדוור", "landwer"],
    "גפניקה": ["גפניקה", "ג'פניקה", "ג׳פניקה", "japanika"],
    "מחניודה": ["מחניודה", "machneyuda", "machneyuda restaurant"],
    "אבו חסן": ["אבו חסן", "abu hassan"],
    "טאיזו": ["טאיזו", "taizu"],
    "שילה": ["שילה", "shila"],
    "קלארו": ["קלארו", "claro"],
    "אורי בורי": ["אורי בורי", "uri buri"],
}

CITY_TO_REGION = {
    "Tel Aviv": "Center", "Ramat Gan": "Center", "Givatayim": "Center", "Holon": "Center", "Rishon LeZion": "Center",
    "Petah Tikva": "Center", "Herzliya": "Center", "Netanya": "Center", "Kfar Saba": "Center", "Ra'anana": "Center",
    "Rehovot": "Center", "Modiin": "Center",
    "Haifa": "North", "Nazareth": "North", "Acre": "North", "Caesarea": "North", "Zichron Yaakov": "North", "Tiberias": "North", "Rosh Pina": "North",
    "Ashdod": "South", "Ashkelon": "South", "Beer Sheva": "South", "Eilat": "South",
    "Jerusalem": "Jerusalem",
}

PRICE_TEXT = {"Cheap": "זולה", "Medium": "במחיר בינוני", "Expensive": "יקרה / יוקרתית"}


# ---------------------------------------------------------------------
# Data extraction and local DB tools
# ---------------------------------------------------------------------


def clean_name(name: Any) -> str:
    name = str(name or "")
    name = re.sub(r"\s*-\s*.*?Branch\s*\d+", "", name, flags=re.I)
    name = re.sub(r"\s*-\s*.*?סניף\s*\d+", "", name)
    return name.strip()


def row_to_dict(row: pd.Series) -> Dict[str, Any]:
    return {
        "restaurant_id": safe_int(row.get("restaurant_id")),
        "name": str(row.get("name_he") or row.get("name")),
        "name_en": str(row.get("name") or ""),
        "city": str(row.get("city_he") or row.get("city")),
        "city_en": str(row.get("city") or ""),
        "region": str(row.get("region_he") or row.get("region")),
        "region_en": str(row.get("region") or ""),
        "cuisine": str(row.get("cuisine_he") or row.get("cuisine")),
        "cuisine_en": str(row.get("cuisine") or ""),
        "restaurant_type": str(row.get("restaurant_type_he") or row.get("restaurant_type")),
        "restaurant_type_en": str(row.get("restaurant_type") or ""),
        "kosher": "כן" if str(row.get("kosher", "")).lower() == "yes" else "לא",
        "kosher_en": str(row.get("kosher") or ""),
        "avg_price_per_person": safe_float(row.get("avg_price_per_person")),
        "price_level": PRICE_TEXT.get(str(row.get("price_level") or ""), str(row.get("price_level") or "")),
        "rating": safe_float(row.get("rating")),
        "suitable_for": str(row.get("suitable_for_he") or row.get("suitable_for")),
        "recommended_dish": str(row.get("recommended_dish_he") or row.get("recommended_dish")),
        "known": str(row.get("is_known_recommended") or ""),
    }


def remember_restaurant(row: pd.Series) -> None:
    rid = safe_int(row.get("restaurant_id"))
    if rid:
        _context["last_restaurant_id"] = rid
        _context["last_restaurant_name"] = str(row.get("name_he") or row.get("name"))


def get_last_restaurant() -> Optional[pd.Series]:
    rid = _context.get("last_restaurant_id")
    if not rid:
        return None
    df = load_restaurants()
    found = df[df["restaurant_id"].astype(str) == str(rid)] if not df.empty else pd.DataFrame()
    if found.empty:
        return None
    return found.iloc[0]


def extract_local_params(message: str) -> Dict[str, Any]:
    msg = normalize_text(message)
    params: Dict[str, Any] = {
        "city": None,
        "region": None,
        "cuisine": None,
        "restaurant_type": None,
        "kosher": None,
        "price_level": None,
        "budget": None,
        "rating_min": None,
        "dish": None,
        "suitable_for": None,
    }

    for heb, eng in sorted(CITY_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        if normalize_text(heb) in msg:
            params["city"] = eng
            params["region"] = CITY_TO_REGION.get(eng)
            break

    for heb, eng in REGION_MAP.items():
        if normalize_text(heb) in msg:
            params["region"] = eng

    for heb, eng in CUISINE_MAP.items():
        if normalize_text(heb) in msg:
            params["cuisine"] = eng

    if any(w in msg for w in ["כשר", "כשרה", "כשרות", "kosher"]):
        if any(w in msg for w in ["לא כשר", "בלי כשרות", "non kosher"]):
            params["kosher"] = "No"
        else:
            params["kosher"] = "Yes"

    if any(w in msg for w in ["זול", "זולה", "cheap"]):
        params["price_level"] = "Cheap"
    elif any(w in msg for w in ["בינוני", "סביר", "medium"]):
        params["price_level"] = "Medium"
    elif any(w in msg for w in ["יקר", "יוקרתי", "expensive"]):
        params["price_level"] = "Expensive"

    nums = [int(x) for x in re.findall(r"\d+", msg)]
    if nums and any(w in msg for w in ["עד", "תקציב", "שקל", "₪", "price", "budget"]):
        params["budget"] = max(nums)
    if nums and any(w in msg for w in ["דירוג", "rating", "כוכבים"]):
        params["rating_min"] = min(max(nums), 5)

    if any(w in msg for w in ["משפחה", "ילדים", "famil"]):
        params["suitable_for"] = "Families"
    elif any(w in msg for w in ["דייט", "זוג", "רומנט", "date", "couple"]):
        params["suitable_for"] = "Couples"
    elif any(w in msg for w in ["חברים", "סטודנטים", "friends"]):
        params["suitable_for"] = "Friends"
    elif any(w in msg for w in ["עסקי", "פגישה", "business"]):
        params["suitable_for"] = "Business"

    return params


def merge_params(ai_params: Dict[str, Any], local_params: Dict[str, Any]) -> Dict[str, Any]:
    merged = local_params.copy()
    mapping = {
        "city": "city", "region": "region", "cuisine": "cuisine", "restaurant_type": "restaurant_type",
        "kosher": "kosher", "price_level": "price_level", "budget": "budget", "rating_min": "rating_min",
        "dish": "dish", "suitable_for": "suitable_for",
    }
    for src, dst in mapping.items():
        val = ai_params.get(src)
        if val not in [None, "", False]:
            merged[dst] = val
    if not merged.get("region") and merged.get("city"):
        merged["region"] = CITY_TO_REGION.get(merged["city"])
    return merged


def has_any_param(params: Dict[str, Any]) -> bool:
    return any(v not in [None, "", False] for v in params.values())


def find_restaurant(message: str, explicit_name: Optional[str] = None) -> Optional[pd.Series]:
    df = load_restaurants()
    if df.empty:
        return None

    search_text = " ".join([message or "", explicit_name or "", str(_context.get("last_restaurant_name") or "")]).strip()
    msg_norm = normalize_text(search_text)
    msg_compact = msg_norm.replace(" ", "")

    # 1) Known aliases
    for canonical, aliases in RESTAURANT_ALIASES.items():
        all_aliases = aliases + [canonical]
        if not any(normalize_text(alias) in msg_norm or normalize_text(alias).replace(" ", "") in msg_compact for alias in all_aliases):
            continue
        for _, row in df.iterrows():
            names = [row.get("name", ""), row.get("name_he", ""), clean_name(row.get("name", "")), clean_name(row.get("name_he", ""))]
            row_norms = [normalize_text(n) for n in names if str(n).strip()]
            if any(any(normalize_text(alias) in rn or rn in normalize_text(alias) for alias in all_aliases) for rn in row_norms):
                remember_restaurant(row)
                return row

    # 2) Explicit name direct contains
    if explicit_name:
        explicit_norm = normalize_text(explicit_name)
        explicit_compact = explicit_norm.replace(" ", "")
        best = []
        for idx, row in df.iterrows():
            names = [row.get("name", ""), row.get("name_he", ""), clean_name(row.get("name", "")), clean_name(row.get("name_he", ""))]
            for n in names:
                rn = normalize_text(n)
                rc = rn.replace(" ", "")
                if rn and (explicit_norm in rn or rn in explicit_norm or explicit_compact in rc or rc in explicit_compact):
                    best.append(idx)
                    break
        if best:
            row = df.loc[best[0]]
            remember_restaurant(row)
            return row

    # 3) Any restaurant name in message
    for _, row in df.iterrows():
        for n in [row.get("name", ""), row.get("name_he", ""), clean_name(row.get("name", "")), clean_name(row.get("name_he", ""))]:
            rn = normalize_text(n)
            rc = rn.replace(" ", "")
            if len(rn) >= 3 and (rn in msg_norm or rc in msg_compact):
                remember_restaurant(row)
                return row

    return None


def filter_restaurants(params: Dict[str, Any]) -> pd.DataFrame:
    df = load_restaurants()
    if df.empty:
        return df
    result = df.copy()

    if params.get("city"):
        result = result[result["city"].astype(str).str.lower() == str(params["city"]).lower()]
    elif params.get("region"):
        result = result[result["region"].astype(str).str.lower() == str(params["region"]).lower()]

    if params.get("cuisine"):
        cuisine = str(params["cuisine"]).lower()
        result = result[result["cuisine"].astype(str).str.lower().str.contains(re.escape(cuisine), na=False)]

    if params.get("restaurant_type"):
        rt = str(params["restaurant_type"]).lower()
        result = result[result["restaurant_type"].astype(str).str.lower().str.contains(re.escape(rt), na=False)]

    if params.get("kosher"):
        result = result[result["kosher"].astype(str).str.lower() == str(params["kosher"]).lower()]

    if params.get("price_level"):
        result = result[result["price_level"].astype(str).str.lower() == str(params["price_level"]).lower()]

    if params.get("budget"):
        result = result[result["avg_price_per_person"] <= safe_float(params["budget"])]

    if params.get("rating_min"):
        result = result[result["rating"] >= safe_float(params["rating_min"])]

    if params.get("suitable_for"):
        sf = str(params["suitable_for"])
        result = result[
            result["suitable_for"].astype(str).str.contains(sf, case=False, na=False)
            | result["suitable_for_he"].astype(str).str.contains(sf, case=False, na=False)
        ]

    if params.get("dish"):
        dish = str(params["dish"])
        result = result[
            result["recommended_dish"].astype(str).str.contains(dish, case=False, na=False)
            | result["recommended_dish_he"].astype(str).str.contains(dish, case=False, na=False)
        ]

    if result.empty:
        return result

    result = result.copy()
    known_score = result.get("is_known_recommended", pd.Series([""] * len(result), index=result.index)).astype(str).str.lower().eq("yes").astype(int)
    result["smartbite_score"] = result["rating"] * 2 + known_score * 1.5 - result["avg_price_per_person"] / 1000
    return result.sort_values(["smartbite_score", "rating"], ascending=False)


def fallback_relaxed_search(params: Dict[str, Any]) -> pd.DataFrame:
    # Gradually relax constraints, but keep location first.
    relax_sets = [
        ["budget", "price_level", "suitable_for"],
        ["budget", "price_level", "suitable_for", "kosher"],
        ["budget", "price_level", "suitable_for", "kosher", "restaurant_type"],
    ]
    for keys in relax_sets:
        relaxed = params.copy()
        for k in keys:
            relaxed[k] = None
        df = filter_restaurants(relaxed)
        if not df.empty:
            return df

    if params.get("city") and CITY_TO_REGION.get(params["city"]):
        relaxed = params.copy()
        relaxed["city"] = None
        relaxed["region"] = CITY_TO_REGION.get(params["city"])
        for k in ["budget", "price_level", "suitable_for", "kosher"]:
            relaxed[k] = None
        df = filter_restaurants(relaxed)
        if not df.empty:
            return df

    relaxed = params.copy()
    for k in ["city", "region", "budget", "price_level", "suitable_for", "kosher", "restaurant_type"]:
        relaxed[k] = None
    return filter_restaurants(relaxed)


def get_restaurant_in_location(base: Optional[pd.Series], params: Dict[str, Any]) -> Optional[pd.Series]:
    if base is None:
        return None
    df = load_restaurants()
    base_name = clean_name(str(base.get("name_he") or base.get("name")))
    chain_key = re.split(r"\s*-\s*", base_name)[0].strip()
    chain_norm = normalize_text(chain_key)
    candidates = df[
        df["name_he"].astype(str).apply(lambda x: chain_norm in normalize_text(clean_name(x)))
        | df["name"].astype(str).apply(lambda x: chain_norm in normalize_text(clean_name(x)))
    ].copy()

    if params.get("city") and not candidates.empty:
        by_city = candidates[candidates["city"].astype(str).str.lower() == str(params["city"]).lower()]
        if not by_city.empty:
            remember_restaurant(by_city.iloc[0])
            return by_city.iloc[0]

    if params.get("region") and not candidates.empty:
        by_region = candidates[candidates["region"].astype(str).str.lower() == str(params["region"]).lower()]
        if not by_region.empty:
            remember_restaurant(by_region.iloc[0])
            return by_region.iloc[0]

    return None


def summarize_reviews(row: pd.Series) -> Dict[str, Any]:
    reviews = load_reviews()
    rid = str(safe_int(row.get("restaurant_id")))
    if reviews.empty or "restaurant_id" not in reviews.columns:
        return {"available": False, "reason": "reviews table unavailable"}
    subset = reviews[reviews["restaurant_id"].astype(str) == rid].copy()
    if subset.empty:
        return {"available": False, "reason": "no reviews for this restaurant"}
    result: Dict[str, Any] = {"available": True, "count": int(len(subset))}
    if "rating" in subset.columns:
        avg = subset["rating"].dropna().mean()
        result["average_review_rating"] = None if pd.isna(avg) else round(float(avg), 2)
    if "sentiment" in subset.columns:
        counts = subset["sentiment"].astype(str).str.lower().value_counts().to_dict()
        result["sentiment_counts"] = {str(k): int(v) for k, v in counts.items()}
        if counts:
            result["dominant_sentiment"] = max(counts.items(), key=lambda x: x[1])[0]
    sample_reviews = []
    if "review_text" in subset.columns:
        for txt in subset["review_text"].dropna().astype(str).head(3):
            sample_reviews.append(txt[:250])
    result["sample_reviews"] = sample_reviews
    return result


def summarize_peak_hours(row: pd.Series) -> Dict[str, Any]:
    orders = load_orders()
    rid = str(safe_int(row.get("restaurant_id")))
    if orders.empty or "restaurant_id" not in orders.columns or "order_datetime" not in orders.columns:
        return {"available": False, "reason": "orders table unavailable"}
    subset = orders[orders["restaurant_id"].astype(str) == rid].copy()
    subset = subset.dropna(subset=["order_datetime"])
    if subset.empty:
        return {"available": False, "reason": "no orders for this restaurant"}
    subset["hour"] = subset["order_datetime"].dt.hour
    peak = subset["hour"].value_counts().sort_values(ascending=False).head(3)
    quiet = subset["hour"].value_counts().sort_values(ascending=True).head(2)
    return {
        "available": True,
        "orders_count": int(len(subset)),
        "peak_hours": [{"hour": int(h), "orders": int(c)} for h, c in peak.items()],
        "quiet_hours": [{"hour": int(h), "orders": int(c)} for h, c in quiet.items()],
    }


def summarize_menu(row: pd.Series) -> Dict[str, Any]:
    menu = load_menu()
    rid = str(safe_int(row.get("restaurant_id")))
    if menu.empty or "restaurant_id" not in menu.columns:
        return {"available": False, "reason": "menu table unavailable"}
    subset = menu[menu["restaurant_id"].astype(str) == rid].copy()
    if subset.empty:
        return {"available": False, "reason": "no menu items for this restaurant"}
    if "popularity_score" in subset.columns:
        subset = subset.sort_values("popularity_score", ascending=False)
    items = []
    for _, item in subset.head(8).iterrows():
        items.append({
            "item_name": str(item.get("item_name") or ""),
            "category": str(item.get("category") or ""),
            "price": safe_float(item.get("price")),
            "is_vegan": str(item.get("is_vegan") or ""),
            "is_gluten_free": str(item.get("is_gluten_free") or ""),
            "popularity_score": safe_float(item.get("popularity_score")),
        })
    return {"available": True, "items": items}


# ---------------------------------------------------------------------
# Cosine Similarity Tool
# ---------------------------------------------------------------------


def is_similarity_request(message: str, intent: Optional[str] = None) -> bool:
    msg = normalize_text(message)
    if intent == "similar_restaurants":
        return True
    return any(w in msg for w in ["דומה", "דומות", "דומים", "בסגנון", "כמו", "similar", "like"])


def cosine_similarity_score(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").fillna(0)
    b = pd.to_numeric(b, errors="coerce").fillna(0)
    numerator = float((a * b).sum())
    denominator = math.sqrt(float((a * a).sum())) * math.sqrt(float((b * b).sum()))
    return 0.0 if denominator == 0 else numerator / denominator


def recommend_similar_restaurants(row: Optional[pd.Series], limit: int = 5) -> pd.DataFrame:
    if row is None:
        return pd.DataFrame()
    df = load_restaurants()
    if df.empty or "restaurant_id" not in df.columns:
        return pd.DataFrame()

    feature_cols = ["cuisine", "restaurant_type", "region", "kosher", "price_level", "suitable_for"]
    available = [c for c in feature_cols if c in df.columns]
    cats = pd.get_dummies(df[available].fillna("").astype(str), columns=available)
    nums = pd.DataFrame(index=df.index)
    nums["rating_norm"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0) / 5.0
    price = pd.to_numeric(df.get("avg_price_per_person", 0), errors="coerce").fillna(0)
    max_price = price.max()
    nums["price_norm"] = price / max_price if max_price and max_price > 0 else 0
    matrix = pd.concat([cats, nums], axis=1)

    target_id = safe_int(row.get("restaurant_id"))
    target_rows = df[df["restaurant_id"].astype(int) == target_id]
    if target_rows.empty:
        return pd.DataFrame()
    target_idx = target_rows.index[0]
    target_vec = matrix.loc[target_idx]

    scores: List[Tuple[int, float]] = []
    for idx, features in matrix.iterrows():
        if safe_int(df.loc[idx, "restaurant_id"]) == target_id:
            continue
        scores.append((idx, cosine_similarity_score(target_vec, features)))
    scores.sort(key=lambda x: x[1], reverse=True)
    chosen = scores[:limit]
    if not chosen:
        return pd.DataFrame()
    result = df.loc[[idx for idx, _ in chosen]].copy()
    result["similarity_score"] = [score for _, score in chosen]
    return result


# ---------------------------------------------------------------------
# Google Places external tool
# ---------------------------------------------------------------------


def google_places_api_key() -> str:
    return os.environ.get("GOOGLE_PLACES_API_KEY", "").strip()


def search_google_places(query: str, language: str = "he") -> Dict[str, Any]:
    api_key = google_places_api_key()
    if not api_key or not query.strip():
        return {"available": False, "reason": "missing GOOGLE_PLACES_API_KEY or query"}

    # Places API (New) Text Search
    try:
        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.currentOpeningHours,places.regularOpeningHours,places.googleMapsUri,places.priceLevel,places.internationalPhoneNumber,places.websiteUri",
        }
        payload = {"textQuery": query, "languageCode": "he" if language == "he" else "en", "regionCode": "IL"}
        resp = requests.post(url, headers=headers, json=payload, timeout=5)
        data = resp.json() if resp.content else {}
        if resp.ok and data.get("places"):
            places = []
            for place in data.get("places", [])[:3]:
                places.append({
                    "name": (place.get("displayName") or {}).get("text"),
                    "address": place.get("formattedAddress"),
                    "rating": place.get("rating"),
                    "user_rating_count": place.get("userRatingCount"),
                    "google_maps_url": place.get("googleMapsUri"),
                    "phone": place.get("internationalPhoneNumber"),
                    "website": place.get("websiteUri"),
                    "opening_hours": place.get("currentOpeningHours") or place.get("regularOpeningHours"),
                    "price_level": place.get("priceLevel"),
                })
            return {"available": True, "source": "Google Places API", "query": query, "places": places}
    except Exception as exc:
        new_api_error = str(exc)
    else:
        new_api_error = str(data)[:500] if 'data' in locals() else "no response"

    # Legacy textsearch fallback
    try:
        legacy_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {"query": query, "key": api_key, "language": "he" if language == "he" else "en", "region": "il"}
        resp = requests.get(legacy_url, params=params, timeout=5)
        data = resp.json() if resp.content else {}
        if resp.ok and data.get("results"):
            places = []
            for place in data.get("results", [])[:3]:
                places.append({
                    "name": place.get("name"),
                    "address": place.get("formatted_address"),
                    "rating": place.get("rating"),
                    "user_rating_count": place.get("user_ratings_total"),
                    "place_id": place.get("place_id"),
                    "opening_hours": place.get("opening_hours"),
                    "price_level": place.get("price_level"),
                })
            return {"available": True, "source": "Google Places API Legacy", "query": query, "places": places}
    except Exception as exc:
        return {"available": False, "reason": f"Google Places error: {exc}; new API error: {new_api_error}"}

    return {"available": False, "reason": f"No external result. New API: {new_api_error}"}


# ---------------------------------------------------------------------
# LLM Agent: planner + final natural answer
# ---------------------------------------------------------------------


def json_from_text(text_value: str) -> Dict[str, Any]:
    try:
        return json.loads(text_value)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text_value or "", flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


def local_intent_hint(message: str) -> Dict[str, Any]:
    msg = normalize_text(message)
    if any(w in msg for w in ["מתכון", "איך מכינים", "איך להכין", "recipe", "how to make", "ingredients"]):
        return {"intent": "off_topic", "is_restaurant_related": False, "off_topic_reason": "recipe"}
    if any(w in msg for w in ["רכב", "מכונית", "אוטו", "טלפון", "מחשב", "פוליטיקה", "מזג אוויר", "טיסה", "מלון", "קוד", "שיעורי בית"]):
        return {"intent": "off_topic", "is_restaurant_related": False}
    if is_similarity_request(message):
        return {"intent": "similar_restaurants", "is_restaurant_related": True}
    return {}


def plan_with_gpt(message: str) -> Dict[str, Any]:
    language = detect_language(message)
    hint = local_intent_hint(message)

    history = _context.get("conversation_history", [])[-8:]
    pending = _context.get("pending_question")
    system_prompt = """
You are the planning brain of SmartBite, an AI restaurant agent for Israel.
Return ONLY valid JSON. Do not answer the user.

SmartBite can discuss restaurants, food places, restaurant recommendations, menus, dishes in restaurants, reviews, ratings, peak hours, opening hours, kosher/non-kosher, prices, cities/regions, cuisines, similar restaurants, and external restaurant information.

Important domain rule:
- Recipe requests such as "give me a pasta recipe" or "how do I make pasta" are OFF_TOPIC, because SmartBite is not a cooking recipe bot.
- Restaurant questions about pasta such as "where can I eat good pasta" ARE restaurant-related.
- Cars, phones, coding, schoolwork, medicine, politics, weather, flights, hotels, movies, sports are off_topic.

Use the conversation history and pending_question. If the user replies with only a city/region/short answer, infer the missing intent from pending_question.

Supported intents:
- smalltalk
- profile_update
- recommendation
- reviews
- rating
- peak_hours
- opening_hours
- recommended_dish
- menu
- similar_restaurants
- general_restaurant_question
- off_topic
- unclear

Return JSON fields:
{
  "intent": "...",
  "language": "he" or "en",
  "is_restaurant_related": true/false,
  "restaurant_name": string|null,
  "city": string|null,
  "region": "North"|"Center"|"South"|"Jerusalem"|null,
  "cuisine": string|null,
  "restaurant_type": string|null,
  "kosher": "Yes"|"No"|null,
  "price_level": "Cheap"|"Medium"|"Expensive"|null,
  "budget": number|null,
  "rating_min": number|null,
  "dish": string|null,
  "suitable_for": "Families"|"Couples"|"Friends"|"Business"|null,
  "needs_internal_data": true/false,
  "needs_external_search": true/false,
  "external_search_query": string|null,
  "missing_info": string|null
}
"""
    try:
        raw = gemini_generate(
            system_prompt=system_prompt,
            user_payload={
                "message": message,
                "language_hint": language,
                "conversation_history": history,
                "pending_question": pending,
                "local_hint": hint,
                "last_restaurant_name": _context.get("last_restaurant_name"),
                "last_params": _context.get("last_params"),
                "user_profile": _context.get("user_profile"),
            },
            temperature=0.1,
            timeout=10,
        )
        parsed = json_from_text(raw)
    except Exception as exc:
        print("Gemini planner technical error:", exc)
        parsed = {
            "intent": "technical_error",
            "language": language,
            "is_restaurant_related": True,
            "technical_error": str(exc),
        }

    if not parsed:
        parsed = {"intent": hint.get("intent") or "unclear", "language": language, "is_restaurant_related": hint.get("is_restaurant_related", True)}

    # Enforce local hard safety rules when needed
    if hint.get("intent") == "off_topic":
        parsed["intent"] = "off_topic"
        parsed["is_restaurant_related"] = False
    if hint.get("intent") == "similar_restaurants":
        parsed["intent"] = "similar_restaurants"
        parsed["is_restaurant_related"] = True

    parsed["language"] = parsed.get("language") or language
    return parsed


def answer_with_gpt(message: str, plan: Dict[str, Any], tool_results: Dict[str, Any]) -> str:
    language = plan.get("language") or detect_language(message)

    if plan.get("intent") == "technical_error":
        return technical_error_message(language)

    history = _context.get("conversation_history", [])[-8:]
    system_prompt = """
You are SmartBite, a real AI restaurant agent powered by Gemini.
Talk naturally, like a helpful human assistant in a chat conversation.

Core behavior:
- Chat naturally with the user, not like a fixed template.
- Answer in Hebrew if language is he; answer in English if language is en.
- Stay strictly within the restaurant/food-place domain.
- If the user asks about an unrelated topic, politely explain that SmartBite focuses only on restaurants and food places.
- If the user asks for a recipe, say you do not provide recipes, but you can help find a restaurant serving that dish.
- Prefer internal database results when available.
- If internal data is not available and external/Gemini general knowledge is used, clearly say that it is not from the internal SmartBite database.
- Do not invent internal database facts.
- Preserve exact names, numbers, prices, ratings, cities, hours and source labels from tool_results.
- If information is missing, ask one natural follow-up question.
- If there are recommendations, present them as clear cards.
- If Cosine Similarity was used, briefly mention that similar restaurants were found by comparing restaurant features.
"""
    try:
        answer = gemini_generate(
            system_prompt=system_prompt,
            user_payload={
                "user_message": message,
                "plan": plan,
                "conversation_history": history,
                "tool_results": tool_results,
                "language": language,
            },
            temperature=0.7,
            timeout=12,
        ).strip()
        if answer:
            return answer
    except Exception as exc:
        print("Gemini final answer technical error:", exc)

    return technical_error_message(language)


# ---------------------------------------------------------------------
# Agent orchestration
# ---------------------------------------------------------------------


def build_external_query(message: str, plan: Dict[str, Any], restaurant_name: Optional[str], params: Dict[str, Any]) -> str:
    parts = []
    if restaurant_name:
        parts.append(restaurant_name)
    elif plan.get("restaurant_name"):
        parts.append(str(plan.get("restaurant_name")))
    if params.get("city"):
        parts.append(str(params["city"]))
    elif params.get("region"):
        parts.append(str(params["region"]))
    parts.append("restaurant Israel")
    query = " ".join(p for p in parts if p).strip()
    return query or plan.get("external_search_query") or f"{message} restaurant Israel"


def run_internal_tools(message: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    language = plan.get("language") or detect_language(message)
    intent = plan.get("intent") or "unclear"
    local_params = extract_local_params(message)
    params = merge_params(plan, local_params)

    # If user gave only a city after a pending question, reuse previous intent and restaurant.
    pending = _context.get("pending_question") or {}
    if pending and (params.get("city") or params.get("region")) and intent in ["unclear", "recommendation", "general_restaurant_question"]:
        intent = pending.get("intent") or intent
        if pending.get("params"):
            merged = pending["params"].copy()
            merged.update({k: v for k, v in params.items() if v not in [None, "", False]})
            params = merged

    restaurant = find_restaurant(message, plan.get("restaurant_name"))
    if restaurant is None and pending.get("restaurant_id"):
        df = load_restaurants()
        found = df[df["restaurant_id"].astype(str) == str(pending["restaurant_id"])]
        if not found.empty:
            restaurant = found.iloc[0]
            remember_restaurant(restaurant)

    result: Dict[str, Any] = {
        "source_priority": "internal_database_first",
        "intent": intent,
        "params": params,
        "internal_found": False,
        "external_used": False,
        "chatgpt_general_used": False,
        "fallback_answer": None,
    }

    if intent == "technical_error":
        result.update({"type": "technical_error", "fallback_answer": technical_error_message(language)})
        return result

    if plan.get("is_restaurant_related") is False or intent == "off_topic":
        result.update({"type": "off_topic", "fallback_answer": "אני מתמחה במסעדות ואוכל בלבד."})
        return result

    if intent in ["smalltalk", "profile_update"]:
        result.update({
            "type": intent,
            "capabilities": ["restaurant recommendations", "reviews", "ratings", "peak hours", "menus", "similar restaurants", "Google Places fallback"],
            "internal_found": True,
        })
        return result

    # Similarity model
    if intent == "similar_restaurants" or is_similarity_request(message, intent):
        if restaurant is None:
            restaurant = get_last_restaurant()
        similar = recommend_similar_restaurants(restaurant, limit=5)
        result.update({
            "type": "similar_restaurants",
            "model": "Cosine Similarity",
            "base_restaurant": row_to_dict(restaurant) if restaurant is not None else None,
            "restaurants": [row_to_dict(r) | {"similarity_score": round(float(r.get("similarity_score", 0)) * 100, 1)} for _, r in similar.iterrows()] if not similar.empty else [],
            "internal_found": restaurant is not None and not similar.empty,
            "fallback_answer": "לא זיהיתי לאיזו מסעדה לחפש מסעדות דומות." if restaurant is None else "לא נמצאו מסעדות דומות בדאטה.",
        })
        if result["internal_found"]:
            _context["pending_question"] = None
        return result

    # Restaurant-specific tools
    if intent in ["reviews", "rating", "peak_hours", "opening_hours", "recommended_dish", "menu", "general_restaurant_question"] and restaurant is not None:
        target = restaurant
        # For branch/location-specific questions, try exact branch first.
        if params.get("city") or params.get("region"):
            located = get_restaurant_in_location(restaurant, params)
            if located is not None:
                target = located

        base_data = row_to_dict(target)
        result["restaurant"] = base_data

        if intent == "reviews":
            result["reviews"] = summarize_reviews(target)
            result["internal_found"] = bool(result["reviews"].get("available"))
        elif intent == "rating":
            result["rating"] = base_data.get("rating")
            result["reviews"] = summarize_reviews(target)
            result["internal_found"] = True
        elif intent == "peak_hours":
            result["peak_hours"] = summarize_peak_hours(target)
            result["internal_found"] = bool(result["peak_hours"].get("available"))
        elif intent == "recommended_dish":
            result["recommended_dish"] = base_data.get("recommended_dish")
            result["menu"] = summarize_menu(target)
            result["internal_found"] = bool(base_data.get("recommended_dish")) or bool(result["menu"].get("available"))
        elif intent == "menu":
            result["menu"] = summarize_menu(target)
            result["internal_found"] = bool(result["menu"].get("available"))
        else:
            # general restaurant questions need external/LLM if DB does not have the specific fact.
            result["internal_found"] = True
            result["note"] = "Internal DB contains basic restaurant facts, but specific general questions may require external knowledge."

        if result["internal_found"]:
            _context["pending_question"] = None
            return result

        # Internal restaurant exists, but the needed detail is missing. Continue to external.
        restaurant_name = clean_name(str(target.get("name_he") or target.get("name")))
        query = build_external_query(message, plan, restaurant_name, params)
        external = search_google_places(query, language=language)
        result["external"] = external
        result["external_used"] = bool(external.get("available"))
        if result["external_used"]:
            return result

    # Recommendations from internal DB
    if intent in ["recommendation", "menu", "general_restaurant_question", "unclear"]:
        if has_any_param(params):
            recs = filter_restaurants(params)
            used_fallback = False
            if recs.empty:
                recs = fallback_relaxed_search(params)
                used_fallback = True
            if not recs.empty:
                result.update({
                    "type": "recommendations",
                    "restaurants": [row_to_dict(r) for _, r in recs.head(6).iterrows()],
                    "used_relaxed_search": used_fallback,
                    "internal_found": True,
                })
                _context["last_params"] = params
                remember_restaurant(recs.iloc[0])
                _context["pending_question"] = None
                return result

        # No useful params / not found, let agent ask naturally or use external if there is enough query text.
        if not has_any_param(params) and intent == "recommendation":
            _context["pending_question"] = {"intent": "recommendation", "params": params}
            result.update({
                "type": "missing_info",
                "missing": "city_or_preferences",
                "fallback_answer": "באיזו עיר, אזור או סגנון אוכל תרצי שאחפש?",
            })
            return result

    # If restaurant not in internal DB or details missing — external Google Places
    restaurant_name = plan.get("restaurant_name") or (_context.get("last_restaurant_name") if intent in ["opening_hours", "reviews", "rating", "peak_hours", "recommended_dish", "menu"] else None)
    external_query = build_external_query(message, plan, restaurant_name, params)
    external = search_google_places(external_query, language=language)
    result["external"] = external
    result["external_used"] = bool(external.get("available"))
    if result["external_used"]:
        _context["pending_question"] = None
        return result

    # Last stage: allow ChatGPT general restaurant knowledge, but mark that it is not internal DB data.
    result["chatgpt_general_used"] = True
    result["type"] = "chatgpt_restaurant_knowledge_fallback"
    result["fallback_answer"] = "לא מצאתי תשובה בדאטה הפנימי או בחיפוש החיצוני. אפשר לענות בזהירות מתוך ידע כללי או לבקש הבהרה."
    return result


def answer_free_text(message: str) -> Dict[str, str]:
    message = (message or "").strip()
    if not message:
        return {"message": "אני כאן 😊 כתבי לי מה תרצי לדעת על מסעדות."}

    plan = plan_with_gpt(message)
    tool_results = run_internal_tools(message, plan)
    answer = answer_with_gpt(message, plan, tool_results)
    remember_conversation(message, answer)
    return {"message": answer}


# ---------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    message = data.get("message", "")
    return jsonify(answer_free_text(message))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=True)
