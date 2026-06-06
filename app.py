from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
except Exception:  # Render will install scikit-learn from requirements.txt
    IsolationForest = None
    TfidfVectorizer = None
    cosine_similarity = None
    StandardScaler = None

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
    "last_chain_key": None,
    "last_params": {},
    "last_results": [],
    "pending_question": None,
}

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    value = str(value or "").lower().strip()
    value = value.replace("׳", "'").replace("’", "'").replace("`", "'")
    value = value.replace("\u200f", " ").replace("\u200e", " ")
    value = re.sub(r"[\"'״׳`.,:;!?()\[\]{}\-_/\\|@#$%^&*+=~]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def detect_language(message: str) -> str:
    return "he" if re.search(r"[\u0590-\u05FF]", message or "") else "en"


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
        return int(float(value))
    except Exception:
        return default


def clean_name(name: Any) -> str:
    name = str(name or "")
    name = re.sub(r"\s*-\s*.*?Branch\s*\d+", "", name, flags=re.I)
    name = re.sub(r"\s*-\s*.*?סניף\s*\d+", "", name)
    return name.strip()


def chain_key_from_row(row: pd.Series) -> str:
    return normalize_text(clean_name(row.get("name_he") or row.get("name"))).replace(" ", "")


def gemini_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_AI_API_KEY", "")).strip()


def gemini_model_candidates() -> List[str]:
    configured = os.environ.get("GEMINI_MODEL", "").strip()
    models = [configured] if configured else []
    models += ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]
    seen: List[str] = []
    for model in models:
        if model and model not in seen:
            seen.append(model)
    return seen


def gemini_generate(system_prompt: str, user_payload: Any, temperature: float = 0.5, timeout: int = 8) -> str:
    api_key = gemini_api_key()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    user_text = user_payload if isinstance(user_payload, str) else json.dumps(user_payload, ensure_ascii=False)
    last_error = None
    for model in gemini_model_candidates():
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {"temperature": temperature, "topP": 0.9, "maxOutputTokens": 800},
        }
        try:
            response = requests.post(url, params={"key": api_key}, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout)
            if response.status_code >= 400:
                last_error = f"{response.status_code}: {response.text[:300]}"
                continue
            data = response.json()
            parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
            text = "".join(str(p.get("text", "")) for p in parts).strip()
            if text:
                return text
            last_error = "empty response"
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(f"Gemini request failed: {last_error}")


def remember_conversation(user_message: str, assistant_message: str) -> None:
    history = _context.setdefault("conversation_history", [])
    history.append({"user": user_message, "assistant": assistant_message})
    _context["conversation_history"] = history[-10:]

# ---------------------------------------------------------------------
# Data loading
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
        for col in ["rating", "avg_price_per_person"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["chain_key"] = df.apply(chain_key_from_row, axis=1)
        df["search_doc"] = df.apply(make_restaurant_doc, axis=1)
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
    if not df.empty:
        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
        if "popularity_score" in df.columns:
            df["popularity_score"] = pd.to_numeric(df["popularity_score"], errors="coerce").fillna(0)
    return df

# ---------------------------------------------------------------------
# Mappings and extraction
# ---------------------------------------------------------------------

CITY_MAP = {
    "תל אביב": "Tel Aviv", "בתל אביב": "Tel Aviv", "תא": "Tel Aviv", "ת א": "Tel Aviv", "בתא": "Tel Aviv", "תל אביב יפו": "Tel Aviv",
    "ירושלים": "Jerusalem", "בירושלים": "Jerusalem",
    "חיפה": "Haifa", "בחיפה": "Haifa",
    "רמת גן": "Ramat Gan", "ברמת גן": "Ramat Gan",
    "גבעתיים": "Givatayim", "בגבעתיים": "Givatayim",
    "חולון": "Holon", "בחולון": "Holon",
    "ראשון לציון": "Rishon LeZion", "בראשון לציון": "Rishon LeZion",
    "פתח תקווה": "Petah Tikva", "בפתח תקווה": "Petah Tikva", "פתח תקוה": "Petah Tikva",
    "הרצליה": "Herzliya", "בהרצליה": "Herzliya",
    "נתניה": "Netanya", "בנתניה": "Netanya",
    "כפר סבא": "Kfar Saba", "בכפר סבא": "Kfar Saba",
    "רעננה": "Ra'anana", "ברעננה": "Ra'anana",
    "רחובות": "Rehovot", "ברחובות": "Rehovot",
    "מודיעין": "Modiin", "במודיעין": "Modiin",
    "אשדוד": "Ashdod", "באשדוד": "Ashdod",
    "אשקלון": "Ashkelon", "באשקלון": "Ashkelon",
    "באר שבע": "Beer Sheva", "בבאר שבע": "Beer Sheva",
    "אילת": "Eilat", "באילת": "Eilat",
    "נצרת": "Nazareth", "בנצרת": "Nazareth",
    "עכו": "Acre", "בעכו": "Acre",
    "קיסריה": "Caesarea", "בקיסריה": "Caesarea",
    "זכרון יעקב": "Zichron Yaakov", "בזכרון יעקב": "Zichron Yaakov",
    "טבריה": "Tiberias", "בטבריה": "Tiberias",
    "ראש פינה": "Rosh Pina", "בראש פינה": "Rosh Pina",
}

CITY_TO_REGION = {
    "Tel Aviv": "Center", "Ramat Gan": "Center", "Givatayim": "Center", "Holon": "Center", "Rishon LeZion": "Center",
    "Petah Tikva": "Center", "Herzliya": "Center", "Netanya": "Center", "Kfar Saba": "Center", "Ra'anana": "Center",
    "Rehovot": "Center", "Modiin": "Center",
    "Haifa": "North", "Nazareth": "North", "Acre": "North", "Caesarea": "North", "Zichron Yaakov": "North", "Tiberias": "North", "Rosh Pina": "North",
    "Ashdod": "South", "Ashkelon": "South", "Beer Sheva": "South", "Eilat": "South", "Jerusalem": "Jerusalem",
}
REGION_MAP = {"צפון": "North", "בצפון": "North", "מרכז": "Center", "במרכז": "Center", "דרום": "South", "בדרום": "South", "אזור ירושלים": "Jerusalem"}

CUISINE_MAP = {
    "איטלקי": "Italian", "איטלקית": "Italian", "אילטקית": "Italian", "אילטקי": "Italian", "פסטה": "Italian",
    "פיצה": "Pizza", "פיצריה": "Pizza",
    "יפני": "Japanese", "יפנית": "Japanese", "סושי": "Japanese", "ניגירי": "Japanese", "מאקי": "Japanese", "סשימי": "Japanese",
    "אסייתי": "Asian", "אסייתית": "Asian", "נודלס": "Asian", "אוכל אסייתי": "Asian",
    "המבורגר": "Burgers", "בורגר": "Burgers",
    "טבעוני": "Vegan", "טבעונית": "Vegan", "צמחוני": "Vegetarian", "צמחונית": "Vegetarian",
    "ישראלי": "Israeli", "ישראלית": "Israeli", "ים תיכוני": "Mediterranean",
    "ערבי": "Arab", "ערבית": "Arab", "יווני": "Greek", "יוונית": "Greek",
    "תאילנדי": "Thai", "דגים": "Seafood", "פירות ים": "Seafood", "סטייק": "Steakhouse",
    "קפה": "Cafe", "בית קפה": "Cafe", "ארוחת בוקר": "Breakfast", "בוקר": "Breakfast",
    "הודי": "Indian", "הודית": "Indian", "מקסיקני": "Mexican", "מקסיקנית": "Mexican",
}

SEMANTIC_EXPANSIONS = {
    "דייט": ["רומנטי", "זוגות", "אווירה", "Couples"],
    "רומנטי": ["דייט", "זוגות", "Couples"],
    "משפחה": ["ילדים", "Families"],
    "חברים": ["Friends", "סטודנטים"],
    "סושי": ["יפני", "אסייתי", "ניגירי", "מאקי", "Japanese", "Asian"],
    "פסטה": ["איטלקי", "Italian"],
    "זול": ["Cheap", "תקציב"],
    "יוקרתי": ["Expensive", "דייט", "רומנטי"],
}

RESTAURANT_ALIASES = {
    "מקדונלדס": ["מקדונלדס", "מקדונלד'ס", "מקדונלד׳ס", "מקדנולדס", "מקדולנדס", "mcdonalds", "mcdonald's"],
    "ארומה": ["ארומה", "aroma"],
    "קפה קפה": ["קפה קפה", "cafe cafe"],
    "גרג": ["גרג", "greg"],
    "לנדוור": ["לנדוור", "landwer"],
    "גפניקה": ["גפניקה", "ג'פניקה", "ג׳פניקה", "japanika"],
    "מחניודה": ["מחניודה", "machneyuda", "machne yuda", "machneyuda restaurant"],
    "אבו חסן": ["אבו חסן", "abu hassan"],
    "טאיזו": ["טאיזו", "taizu"],
    "שילה": ["שילה", "shila"],
    "קלארו": ["קלארו", "claro"],
    "אורי בורי": ["אורי בורי", "uri buri"],
}

PRICE_TEXT = {"Cheap": "זול", "Medium": "בינוני", "Expensive": "יקר"}


def display_city(value: Any) -> str:
    """Returns Hebrew city name when possible, otherwise the original value."""
    if not value:
        return ""
    val = str(value)
    df = load_csv_cached("restaurants", RESTAURANTS_CSV)
    if not df.empty and "city" in df.columns and "city_he" in df.columns:
        found = df[df["city"].astype(str).str.lower() == val.lower()]
        if not found.empty:
            he = str(found.iloc[0].get("city_he") or "").strip()
            if he:
                return he
    return val


def display_region(value: Any) -> str:
    if not value:
        return ""
    mapping = {"Center": "מרכז", "North": "צפון", "South": "דרום", "Jerusalem": "ירושלים"}
    return mapping.get(str(value), str(value))


def display_requested_place(params: Dict[str, Any]) -> str:
    if params.get("city"):
        return display_city(params.get("city"))
    if params.get("region"):
        return display_region(params.get("region"))
    return "האזור שביקשת"


def cuisine_display(value: Any) -> str:
    if not value:
        return ""
    mapping = {
        "Italian": "איטלקית", "Pizza": "פיצה", "Japanese": "יפנית/סושי", "Asian": "אסייתית",
        "Burgers": "המבורגר", "Vegan": "טבעונית", "Vegetarian": "צמחונית", "Israeli": "ישראלית",
        "Mediterranean": "ים תיכונית", "Arab": "ערבית", "Greek": "יוונית", "Thai": "תאילנדית",
        "Seafood": "דגים/פירות ים", "Steakhouse": "סטייקים", "Cafe": "בית קפה", "Breakfast": "ארוחת בוקר",
        "Indian": "הודית", "Mexican": "מקסיקנית",
    }
    return mapping.get(str(value), str(value))

INTENT_WORDS = {
    "reviews": ["ביקורות", "חוות דעת", "reviews", "review"],
    "peak_hours": ["שעות עומס", "עומס", "עמוס", "busy", "peak"],
    "recommended_dish": ["מנה מומלצת", "המנה המומלצת", "מה לאכול", "dish", "recommended dish"],
    "similar_restaurants": ["דומות", "דומה", "דומים", "בסגנון", "כמו", "similar", "like"],
    "rating_trust": ["לסמוך על הדירוג", "אפשר לסמוך", "אמינות הדירוג", "הדירוג אמין", "דירוג אמין", "trust rating", "rating trust", "reliable rating"],
    "anomalies": ["אנומל", "חריג", "חריגות", "חשוד", "חשודות", "isolation", "anomaly"],
    "menu": ["תפריט", "מנות", "menu"],
    "opening_hours": ["שעות פתיחה", "פתוח", "פתיחה", "opening"],
    "rating": ["דירוג", "ציון", "rating"],
}

OFF_TOPIC_WORDS = ["רכב", "מכונית", "אוטו", "טלפון", "מחשב", "פוליטיקה", "מזג אוויר", "טיסה", "מלון", "קוד", "שיעורי בית", "רופא", "תרופה"]
RECIPE_WORDS = ["מתכון", "איך מכינים", "איך להכין", "recipe", "how to make", "ingredients"]

# ---------------------------------------------------------------------
# Data representation
# ---------------------------------------------------------------------


def make_restaurant_doc(row: pd.Series) -> str:
    fields = [
        row.get("name", ""), row.get("name_he", ""), row.get("city", ""), row.get("city_he", ""),
        row.get("region", ""), row.get("region_he", ""), row.get("cuisine", ""), row.get("cuisine_he", ""),
        row.get("restaurant_type", ""), row.get("restaurant_type_he", ""), row.get("kosher", ""),
        row.get("price_level", ""), row.get("suitable_for", ""), row.get("suitable_for_he", ""),
        row.get("recommended_dish", ""), row.get("recommended_dish_he", ""),
    ]
    return " ".join(str(x) for x in fields if str(x).strip())


def expand_query_semantically(text: str) -> str:
    """Lightweight semantic embedding layer: expands natural Hebrew/English phrases
    into meaning-related tokens before TF-IDF and Cosine Similarity ranking.
    This gives the project an embeddings-style representation without another API.
    """
    norm = normalize_text(text)
    extra: List[str] = []
    for key, words in SEMANTIC_EXPANSIONS.items():
        if normalize_text(key) in norm:
            extra.extend(words)
    return f"{text} {' '.join(extra)}".strip()


def row_to_dict(row: pd.Series) -> Dict[str, Any]:
    return {
        "restaurant_id": safe_int(row.get("restaurant_id")),
        "name": str(row.get("name_he") or row.get("name")),
        "clean_name": clean_name(row.get("name_he") or row.get("name")),
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
    }


def remember_restaurant(row: pd.Series) -> None:
    rid = safe_int(row.get("restaurant_id"))
    if rid:
        _context["last_restaurant_id"] = rid
        _context["last_restaurant_name"] = str(row.get("name_he") or row.get("name"))
        _context["last_chain_key"] = chain_key_from_row(row)

# ---------------------------------------------------------------------
# Intent / params / branch logic
# ---------------------------------------------------------------------


def detect_intent(message: str) -> str:
    msg = normalize_text(message)
    if any(w in msg for w in RECIPE_WORDS):
        return "recipe_off_topic"
    if any(w in msg for w in OFF_TOPIC_WORDS):
        return "off_topic"
    for intent, words in INTENT_WORDS.items():
        if any(normalize_text(w) in msg for w in words):
            return intent
    if any(w in msg for w in ["היי", "שלום", "מה קורה", "מה נשמע", "תודה", "hi", "hello", "thanks"]):
        return "smalltalk"
    return "recommendation"


def extract_params(message: str) -> Dict[str, Any]:
    msg = normalize_text(message)
    params: Dict[str, Any] = {"city": None, "region": None, "cuisine": None, "kosher": None, "price_level": None, "budget": None, "rating_min": None, "suitable_for": None}
    for heb, eng in sorted(CITY_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if normalize_text(heb) in msg:
            params["city"] = eng
            params["region"] = CITY_TO_REGION.get(eng)
            break
    for heb, eng in REGION_MAP.items():
        if normalize_text(heb) in msg:
            params["region"] = eng
    for heb, eng in sorted(CUISINE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if normalize_text(heb) in msg:
            params["cuisine"] = eng
            break
    if "כשר" in msg or "כשרה" in msg or "כשרות" in msg or "kosher" in msg:
        params["kosher"] = "No" if any(w in msg for w in ["לא כשר", "בלי כשרות", "non kosher"]) else "Yes"
    if any(w in msg for w in ["זול", "זולה", "cheap"]):
        params["price_level"] = "Cheap"
    elif any(w in msg for w in ["בינוני", "סביר", "medium"]):
        params["price_level"] = "Medium"
    elif any(w in msg for w in ["יקר", "יוקרתי", "expensive"]):
        params["price_level"] = "Expensive"
    nums = [int(x) for x in re.findall(r"\d+", msg)]
    if nums and any(w in msg for w in ["עד", "תקציב", "שקל", "₪", "מחיר", "price", "budget"]):
        params["budget"] = max(nums)
    if nums and any(w in msg for w in ["דירוג", "כוכבים", "rating"]):
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


def has_any_param(params: Dict[str, Any]) -> bool:
    return any(v not in [None, "", False] for v in params.values())


def has_location(params: Dict[str, Any]) -> bool:
    return bool(params.get("city") or params.get("region"))


def message_chain_alias(message: str) -> Optional[str]:
    msg = normalize_text(message)
    msg_compact = msg.replace(" ", "")
    for canonical, aliases in RESTAURANT_ALIASES.items():
        for alias in aliases + [canonical]:
            a = normalize_text(alias)
            if a and (a in msg or a.replace(" ", "") in msg_compact):
                return normalize_text(canonical).replace(" ", "")
    return None


def find_chain_candidates(message: str, explicit_chain: Optional[str] = None) -> pd.DataFrame:
    df = load_restaurants()
    if df.empty:
        return df
    chain = explicit_chain or message_chain_alias(message)
    if chain:
        c = normalize_text(chain).replace(" ", "")
        return df[df["chain_key"].astype(str).str.contains(re.escape(c), na=False)].copy()
    msg = normalize_text(message)
    msg_compact = msg.replace(" ", "")
    matches = []
    for idx, row in df.iterrows():
        for n in [row.get("name", ""), row.get("name_he", ""), clean_name(row.get("name", "")), clean_name(row.get("name_he", ""))]:
            rn = normalize_text(n)
            rc = rn.replace(" ", "")
            if len(rc) >= 3 and (rc in msg_compact or rn in msg):
                matches.append(idx)
                break
    return df.loc[matches].copy() if matches else pd.DataFrame()


def select_branch(candidates: pd.DataFrame, params: Dict[str, Any]) -> Optional[pd.Series]:
    if candidates.empty:
        return None
    subset = candidates
    if params.get("city"):
        by_city = subset[subset["city"].astype(str).str.lower() == str(params["city"]).lower()]
        if not by_city.empty:
            return by_city.sort_values("rating", ascending=False).iloc[0]
        return None
    if params.get("region"):
        by_region = subset[subset["region"].astype(str).str.lower() == str(params["region"]).lower()]
        if not by_region.empty:
            return by_region.sort_values("rating", ascending=False).iloc[0]
    return subset.sort_values("rating", ascending=False).iloc[0]


def available_cities_for_chain(candidates: pd.DataFrame, limit: int = 8) -> List[str]:
    if candidates.empty:
        return []
    cities = candidates["city_he"].replace("", np.nan).dropna().astype(str).value_counts().index.tolist()
    return cities[:limit]


def format_no_branch_found(chain_name: str, params: Dict[str, Any], candidates: pd.DataFrame, intent: str) -> str:
    requested = display_requested_place(params)
    clean = clean_name(chain_name)
    cities = available_cities_for_chain(candidates)
    examples = " / ".join(cities[:6]) if cities else "עיר אחרת"

    if intent == "similar_restaurants":
        return f"לא מצאתי סניף של {clean} ב{requested}, אבל עדיין אפשר לחפש מסעדות דומות לפי הסגנון שלה. באיזו עיר או אזור אחר תרצי שאחפש? למשל: {examples}."
    return f"לא מצאתי סניף של {clean} ב{requested}. באיזו עיר אחרת תרצי לבדוק? למשל: {examples}."


def choose_base_restaurant_for_similarity(candidates: pd.DataFrame) -> Optional[pd.Series]:
    if candidates.empty:
        return None
    # למסעדות דומות לא חייב להיות סניף בעיר המבוקשת; משתמשים ברשת/מסעדה כבסיס סגנוני.
    return candidates.sort_values("rating", ascending=False).iloc[0]


def should_ask_city_for_chain(intent: str, candidates: pd.DataFrame, params: Dict[str, Any]) -> bool:
    if candidates.empty or has_location(params):
        return False
    if intent in ["reviews", "rating", "rating_trust", "peak_hours", "opening_hours", "recommended_dish", "menu", "similar_restaurants"]:
        return candidates["city"].nunique() > 1 or len(candidates) > 1
    return False


def ask_city(chain_name: str, candidates: pd.DataFrame, intent: str, params: Dict[str, Any]) -> str:
    cities = available_cities_for_chain(candidates)
    chain_key = message_chain_alias(chain_name) or normalize_text(chain_name).replace(" ", "")
    _context["pending_question"] = {
        "intent": intent,
        "chain_key": chain_key,
        "params": params,
        "missing": "city_or_region",
    }
    examples = " / ".join(cities[:5]) if cities else "למשל תל אביב, רמת גן או ירושלים"
    clean = clean_name(chain_name)

    if intent == "similar_restaurants":
        return f"באיזו עיר או אזור תרצי שאחפש מסעדות דומות ל{clean}? למשל: {examples}."
    if intent == "reviews":
        return f"באיזו עיר או אזור תרצי שאבדוק את הביקורות של {clean}? למשל: {examples}."
    if intent == "peak_hours":
        return f"באיזו עיר או אזור תרצי שאבדוק את שעות העומס של {clean}? למשל: {examples}."
    if intent == "recommended_dish":
        return f"באיזו עיר או אזור תרצי שאבדוק את המנה המומלצת של {clean}? למשל: {examples}."
    if intent == "rating_trust":
        return f"באיזו עיר או אזור תרצי שאבדוק את אמינות הדירוג של {clean}? למשל: {examples}."

    return f"באיזו עיר או אזור תרצי שאבדוק את {clean}? למשל: {examples}."


def is_affirmative(message: str) -> bool:
    msg = normalize_text(message)
    return any(w in msg for w in ["כן", "בטח", "סבבה", "מעולה", "נשמע טוב", "נשמע מעולה", "יאללה", "אפשר", "כן נשמע", "yes", "sure", "ok", "okay"])


def is_negative(message: str) -> bool:
    msg = normalize_text(message)
    return any(w in msg for w in ["לא", "לא תודה", "no", "no thanks"])

def apply_pending(message: str, intent: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[str]]:
    pending = _context.get("pending_question") or {}
    if not pending:
        return intent, params, None

    pending_intent = pending.get("intent") or intent
    old = pending.get("params") or {}

    # אם המשתמש ענה רק עיר/אזור, ממשיכים את הבקשה הקודמת.
    if has_location(params):
        merged = old.copy()
        merged.update({k: v for k, v in params.items() if v not in [None, "", False]})
        return pending_intent, merged, pending.get("chain_key")

    # אם המשתמש ענה כן/מעולה/נשמע טוב, לא מתחילים שיחה חדשה.
    # ממשיכים את הפעולה הקודמת ושואלים את השדה שחסר, בדרך כלל עיר/אזור.
    if is_affirmative(message):
        merged = old.copy()
        merged.update({k: v for k, v in params.items() if v not in [None, "", False]})
        return pending_intent, merged, pending.get("chain_key")

    if is_negative(message):
        _context["pending_question"] = None

    return intent, params, None

# ---------------------------------------------------------------------
# Restaurant tools
# ---------------------------------------------------------------------


def summarize_reviews(row: pd.Series) -> Dict[str, Any]:
    reviews = load_reviews()
    rid = str(safe_int(row.get("restaurant_id")))
    if reviews.empty or "restaurant_id" not in reviews.columns:
        return {"available": False, "count": 0}
    subset = reviews[reviews["restaurant_id"].astype(str) == rid].copy()
    if subset.empty:
        return {"available": False, "count": 0}
    result: Dict[str, Any] = {"available": True, "count": int(len(subset))}
    if "rating" in subset.columns:
        avg = subset["rating"].dropna().mean()
        result["average_review_rating"] = None if pd.isna(avg) else round(float(avg), 2)
    if "sentiment" in subset.columns:
        result["sentiment_counts"] = {str(k): int(v) for k, v in subset["sentiment"].astype(str).str.lower().value_counts().to_dict().items()}
    samples = []
    if "review_text" in subset.columns:
        for txt in subset["review_text"].dropna().astype(str).head(4):
            samples.append(txt[:220])
    result["sample_reviews"] = samples
    return result


def summarize_peak_hours(row: pd.Series) -> Dict[str, Any]:
    orders = load_orders()
    rid = str(safe_int(row.get("restaurant_id")))
    if orders.empty or "restaurant_id" not in orders.columns or "order_datetime" not in orders.columns:
        return {"available": False}
    subset = orders[orders["restaurant_id"].astype(str) == rid].dropna(subset=["order_datetime"]).copy()
    if subset.empty:
        return {"available": False}
    subset["hour"] = subset["order_datetime"].dt.hour
    peak = subset["hour"].value_counts().sort_values(ascending=False).head(3)
    return {"available": True, "orders_count": int(len(subset)), "peak_hours": [{"hour": int(h), "orders": int(c)} for h, c in peak.items()]}


def summarize_menu(row: pd.Series) -> Dict[str, Any]:
    menu = load_menu()
    rid = str(safe_int(row.get("restaurant_id")))
    if menu.empty or "restaurant_id" not in menu.columns:
        return {"available": False, "items": []}
    subset = menu[menu["restaurant_id"].astype(str) == rid].copy()
    if subset.empty:
        return {"available": False, "items": []}
    if "popularity_score" in subset.columns:
        subset = subset.sort_values("popularity_score", ascending=False)
    items = []
    for _, item in subset.head(8).iterrows():
        items.append({"item_name": str(item.get("item_name") or ""), "category": str(item.get("category") or ""), "price": safe_float(item.get("price")), "popularity_score": safe_float(item.get("popularity_score"))})
    return {"available": True, "items": items}


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
        result = result[result["suitable_for"].astype(str).str.contains(sf, case=False, na=False) | result["suitable_for_he"].astype(str).str.contains(sf, case=False, na=False)]
    if result.empty:
        return result
    result = result.copy()
    known = result.get("is_known_recommended", pd.Series([""] * len(result), index=result.index)).astype(str).str.lower().eq("yes").astype(float)
    result["rank_score"] = result["rating"] * 2 + known * 1.5 - result["avg_price_per_person"] / 1000
    return result.sort_values(["rank_score", "rating"], ascending=False)

# ---------------------------------------------------------------------
# TF-IDF, Embeddings / semantic representation, Cosine Similarity, Anomaly Detection
# ---------------------------------------------------------------------


def tfidf_rank_restaurants(query: str, df: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if df.empty or TfidfVectorizer is None or cosine_similarity is None:
        return df.head(limit)
    docs = df["search_doc"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    try:
        matrix = vectorizer.fit_transform(docs)
        q_vec = vectorizer.transform([expand_query_semantically(query)])
        sims = cosine_similarity(q_vec, matrix).flatten()
        ranked = df.copy()
        ranked["tfidf_score"] = sims
        # Use TF-IDF as a model signal, but do not let tiny scores erase filtered matches.
        ranked["combined_score"] = ranked.get("rank_score", ranked["rating"] * 2) + ranked["tfidf_score"] * 4
        return ranked.sort_values(["combined_score", "rating"], ascending=False).head(limit)
    except Exception:
        return df.head(limit)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["cuisine", "restaurant_type", "region", "kosher", "price_level", "suitable_for"]
    available = [c for c in feature_cols if c in df.columns]
    cats = pd.get_dummies(df[available].fillna("").astype(str), columns=available)
    nums = pd.DataFrame(index=df.index)
    nums["rating_norm"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0) / 5.0
    price = pd.to_numeric(df.get("avg_price_per_person", 0), errors="coerce").fillna(0)
    max_price = float(price.max()) if len(price) else 0
    nums["price_norm"] = price / max_price if max_price > 0 else 0
    return pd.concat([cats, nums], axis=1)


def recommend_similar_restaurants(row: Optional[pd.Series], params: Dict[str, Any], limit: int = 5) -> pd.DataFrame:
    if row is None:
        return pd.DataFrame()
    df = load_restaurants()
    if df.empty or cosine_similarity is None:
        return pd.DataFrame()
    candidates = df.copy()
    if params.get("city"):
        candidates = candidates[candidates["city"].astype(str).str.lower() == str(params["city"]).lower()]
    elif params.get("region"):
        candidates = candidates[candidates["region"].astype(str).str.lower() == str(params["region"]).lower()]
    if candidates.empty:
        candidates = df.copy()
    matrix_all = build_feature_matrix(pd.concat([pd.DataFrame([row]), candidates], ignore_index=True)).fillna(0)
    target_vec = matrix_all.iloc[[0]]
    cand_matrix = matrix_all.iloc[1:]
    sims = cosine_similarity(target_vec, cand_matrix).flatten()
    result = candidates.copy().reset_index(drop=True)
    result["similarity_score"] = sims
    target_id = safe_int(row.get("restaurant_id"))
    result = result[result["restaurant_id"].astype(str) != str(target_id)]
    result = result.sort_values(["similarity_score", "rating"], ascending=False).head(limit)
    return result


def anomaly_detection(limit: int = 8) -> pd.DataFrame:
    df = load_restaurants()
    reviews = load_reviews()
    if df.empty:
        return df
    work = df.copy()
    if not reviews.empty and "restaurant_id" in reviews.columns:
        agg = reviews.groupby("restaurant_id").agg(
            review_count=("review_id", "count"),
            review_avg=("rating", "mean"),
            negative_share=("sentiment", lambda s: float((s.astype(str).str.lower() == "negative").mean())),
        ).reset_index()
        work = work.merge(agg, on="restaurant_id", how="left")
    else:
        work["review_count"] = 0
        work["review_avg"] = work["rating"]
        work["negative_share"] = 0
    for col in ["review_count", "review_avg", "negative_share"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
    work["rating_review_gap"] = work["rating"] - work["review_avg"]
    work["perfect_low_reviews"] = ((work["rating"] >= 4.8) & (work["review_count"] <= 2)).astype(int)
    # Explicit business anomaly score helps identify interpretable restaurant anomalies.
    work["business_anomaly_score"] = (
        work["negative_share"] * 3
        + work["rating_review_gap"].clip(lower=0) * 1.5
        + work["perfect_low_reviews"] * 1.2
        + (work["avg_price_per_person"] > work["avg_price_per_person"].quantile(0.97)).astype(int) * 0.8
    )
    if IsolationForest is not None and StandardScaler is not None and len(work) > 20:
        features = work[["rating", "avg_price_per_person", "review_count", "review_avg", "negative_share", "rating_review_gap", "perfect_low_reviews"]].fillna(0)
        scaled = StandardScaler().fit_transform(features)
        iso = IsolationForest(n_estimators=120, contamination=0.04, random_state=42)
        work["isolation_score"] = -iso.fit_predict(scaled) + (-iso.decision_function(scaled))
    else:
        work["isolation_score"] = 0
    work["anomaly_score"] = work["business_anomaly_score"] + pd.to_numeric(work["isolation_score"], errors="coerce").fillna(0)
    return work.sort_values("anomaly_score", ascending=False).head(limit)

# ---------------------------------------------------------------------
# Google Places fallback
# ---------------------------------------------------------------------


def google_places_api_key() -> str:
    return os.environ.get("GOOGLE_PLACES_API_KEY", "").strip()


def search_google_places(query: str, language: str = "he") -> Dict[str, Any]:
    api_key = google_places_api_key()
    if not api_key or not query.strip():
        return {"available": False}
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
                    "opening_hours": place.get("currentOpeningHours") or place.get("regularOpeningHours"),
                    "phone": place.get("internationalPhoneNumber"),
                    "website": place.get("websiteUri"),
                })
            return {"available": True, "places": places}
    except Exception as exc:
        print("Google Places error:", exc)
    return {"available": False}

# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------


def restaurant_card(row_or_dict: Any, idx: int = 1, include_similarity: bool = False) -> str:
    r = row_to_dict(row_or_dict) if isinstance(row_or_dict, pd.Series) else row_or_dict
    parts = [f"**{idx}. {r.get('name', 'מסעדה')}**"]
    if r.get("city"):
        parts.append(f"עיר: {r.get('city')}")
    if r.get("cuisine"):
        parts.append(f"מטבח: {r.get('cuisine')}")
    if r.get("restaurant_type"):
        parts.append(f"סוג: {r.get('restaurant_type')}")
    if r.get("kosher"):
        parts.append(f"כשרות: {r.get('kosher')}")
    if r.get("rating"):
        parts.append(f"דירוג: {r.get('rating')}")
    if r.get("avg_price_per_person"):
        parts.append(f"מחיר ממוצע לאדם: ₪{int(float(r.get('avg_price_per_person')))}")
    if r.get("recommended_dish"):
        parts.append(f"מנה מומלצת: {r.get('recommended_dish')}")
    if include_similarity and r.get("similarity_score") is not None:
        parts.append(f"דמיון: {round(float(r.get('similarity_score')) * 100, 1)}%")
    return "\n".join(parts)


def format_reviews(row: pd.Series) -> str:
    r = row_to_dict(row)
    rev = summarize_reviews(row)
    if not rev.get("available"):
        return f"לא מצאתי ביקורות זמינות עבור {r['name']}."
    samples = rev.get("sample_reviews") or []
    lines = [f"מצאתי ביקורות עבור {r['name']} ב{r['city']}.", f"דירוג ביקורות ממוצע: {rev.get('average_review_rating', r.get('rating'))}", f"מספר ביקורות: {rev.get('count')}"]
    if samples:
        lines.append("\nדוגמאות מביקורות:")
        lines.extend([f"• {s}" for s in samples[:3]])
    return "\n".join(lines)


def format_peak(row: pd.Series, params: Dict[str, Any]) -> str:
    r = row_to_dict(row)
    peak = summarize_peak_hours(row)
    if peak.get("available"):
        hours = ", ".join([f"{x['hour']}:00" for x in peak.get("peak_hours", [])])
        return f"שעות העומס הבולטות של {r['name']} ב{r['city']}: {hours}."
    query = f"{clean_name(r['name'])} {params.get('city') or r.get('city_en') or ''} restaurant busy hours Israel"
    ext = search_google_places(query)
    if ext.get("available"):
        place = ext["places"][0]
        return f"לא מצאתי נתוני עומס מדויקים, אבל מצאתי את {place.get('name')}. אפשר לבדוק שעות פתיחה ופרטים נוספים במפות Google."
    return f"לא מצאתי נתוני עומס זמינים עבור {r['name']} ב{r['city']}."


def format_recommended_dish(row: pd.Series) -> str:
    r = row_to_dict(row)
    dish = r.get("recommended_dish")
    if dish:
        return f"המנה המומלצת של {r['name']} ב{r['city']} היא: {dish}."
    menu = summarize_menu(row)
    if menu.get("available") and menu.get("items"):
        item = menu["items"][0]
        return f"המנה שהכי כדאי לבדוק ב{r['name']} היא {item.get('item_name')} במחיר של כ־₪{int(item.get('price', 0))}."
    return f"לא מצאתי מנה מומלצת עבור {r['name']}."


def format_recommendations(message: str, params: Dict[str, Any]) -> str:
    recs = filter_restaurants(params)
    if recs.empty:
        return "לא מצאתי התאמה מדויקת. אפשר לנסות עיר אחרת, תקציב אחר או סוג מטבח אחר."
    ranked = tfidf_rank_restaurants(message, recs, limit=5)
    _context["last_results"] = [safe_int(x) for x in ranked["restaurant_id"].head(5).tolist()]
    cards = [restaurant_card(row, i + 1) for i, (_, row) in enumerate(ranked.head(5).iterrows())]
    return "מצאתי לך כמה אפשרויות מתאימות 😊\n\n" + "\n\n".join(cards)


def format_similar(base: pd.Series, params: Dict[str, Any]) -> str:
    similar = recommend_similar_restaurants(base, params, limit=5)
    base_name = clean_name(base.get("name_he") or base.get("name"))
    if similar.empty:
        loc = params.get("city") or params.get("region")
        if loc:
            return f"לא הצלחתי למצוא כרגע מסעדות דומות ל{base_name} באזור שביקשת. אפשר לנסות עיר או אזור אחר."
        return "לא הצלחתי למצוא מסעדות דומות כרגע."
    location_text = ""
    if params.get("city"):
        location_text = f" ב{display_city(params.get('city'))}"
    elif params.get("region"):
        location_text = f" באזור {display_region(params.get('region'))}"
    cards = [restaurant_card(row, i + 1, include_similarity=True) for i, (_, row) in enumerate(similar.iterrows())]
    return f"מצאתי מסעדות דומות ל{base_name}{location_text}:\n\n" + "\n\n".join(cards)


def format_rating_trust(row: pd.Series) -> str:
    r = row_to_dict(row)
    rev = summarize_reviews(row)
    rating = safe_float(r.get("rating"))
    count = safe_int(rev.get("count"))
    review_avg = safe_float(rev.get("average_review_rating"), rating)
    sentiments = rev.get("sentiment_counts") or {}
    total_sentiments = sum(safe_int(v) for v in sentiments.values())
    negative = safe_int(sentiments.get("negative", 0))
    negative_share = (negative / total_sentiments) if total_sentiments else 0
    gap = rating - review_avg

    warnings = []
    positives = []

    if count == 0:
        warnings.append("לא מצאתי מספיק ביקורות כדי להעריך את אמינות הדירוג בצורה חזקה")
    elif count < 5:
        warnings.append("מספר הביקורות נמוך יחסית, ולכן קשה לסמוך רק על הדירוג")
    else:
        positives.append("יש כמות ביקורות שמאפשרת לקבל תמונה טובה יותר")

    if gap > 0.7:
        warnings.append("יש פער בין הדירוג הכללי לבין ממוצע הדירוגים בביקורות")
    if negative_share > 0.30:
        warnings.append("חלק משמעותי מהביקורות מסומנות כשליליות")
    if not warnings and count > 0:
        positives.append("לא זיהיתי פער חריג בין הדירוג לבין הביקורות")

    lines = [
        f"בדקתי את אמינות הדירוג של {r['name']} ב{r['city']}.",
        f"דירוג המסעדה: {rating:.1f}",
    ]
    if count:
        lines.append(f"מספר ביקורות שנבדקו: {count}")
        lines.append(f"ממוצע דירוג בביקורות: {review_avg:.1f}")

    if warnings:
        lines.append("\nשימי לב:")
        lines.extend([f"• {w}" for w in warnings])
        lines.append("\nלכן הייתי מתייחסת לדירוג בזהירות וקוראת עוד כמה ביקורות לפני בחירה.")
    else:
        lines.append("\nנראה שהדירוג יחסית עקבי עם הביקורות, ולכן אפשר להתייחס אליו כאמין יותר.")

    if positives:
        lines.append("\nמה מחזק את האמינות:")
        lines.extend([f"• {p}" for p in positives])
    return "\n".join(lines)


def format_anomalies() -> str:
    anomalies = anomaly_detection(limit=6)
    if anomalies.empty:
        return "לא מצאתי חריגות משמעותיות כרגע."
    lines = ["מצאתי כמה מסעדות שנראות חריגות ושווה לבדוק:"]
    for i, (_, row) in enumerate(anomalies.iterrows(), 1):
        r = row_to_dict(row)
        reasons = []
        if safe_float(row.get("negative_share")) > 0.35:
            reasons.append("הרבה ביקורות שליליות יחסית")
        if safe_float(row.get("rating_review_gap")) > 0.8:
            reasons.append("פער בין דירוג המסעדה לדירוג הביקורות")
        if safe_int(row.get("perfect_low_reviews")):
            reasons.append("דירוג גבוה מאוד עם מעט ביקורות")
        if safe_float(row.get("avg_price_per_person")) > load_restaurants()["avg_price_per_person"].quantile(0.97):
            reasons.append("מחיר חריג יחסית")
        reason_txt = ", ".join(reasons) if reasons else "דפוס נתונים לא רגיל"
        lines.append(f"\n**{i}. {r['name']}**\nעיר: {r['city']}\nדירוג: {r['rating']}\nסיבה: {reason_txt}")
    return "\n".join(lines)

# ---------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------


def natural_gemini_fallback(message: str) -> str:
    try:
        return gemini_generate(
            """You are SmartBite, a friendly restaurant AI agent. Answer naturally in the user's language. Stay only in the restaurant/food-place domain. Do not mention databases, internal sources, external sources, or APIs.""",
            {"message": message, "history": _context.get("conversation_history", [])[-6:]},
            temperature=0.7,
            timeout=6,
        )
    except Exception:
        return "אני כאן 😊 אפשר לשאול אותי על מסעדות, המלצות, ביקורות, שעות עומס, מנות מומלצות או מסעדות דומות."


def answer_free_text(message: str) -> Dict[str, str]:
    message = (message or "").strip()
    if not message:
        return {"message": "אני כאן 😊 כתבי לי מה תרצי לדעת על מסעדות."}

    language = detect_language(message)
    intent = detect_intent(message)
    params = extract_params(message)
    intent, params, pending_chain = apply_pending(message, intent, params)

    if intent == "recipe_off_topic":
        answer = "אני לא נותנת מתכונים, אבל יכולה לעזור למצוא מסעדה שמגישה את המנה הזו 😊"
        remember_conversation(message, answer)
        return {"message": answer}
    if intent == "off_topic":
        answer = "אני מתמחה במסעדות ואוכל 🍽️ אשמח לעזור בהמלצות, ביקורות, שעות עומס, מנות מומלצות או מסעדות דומות."
        remember_conversation(message, answer)
        return {"message": answer}
    if intent == "smalltalk":
        answer = natural_gemini_fallback(message)
        remember_conversation(message, answer)
        return {"message": answer}

    # Anomaly model
    if intent == "anomalies":
        answer = format_anomalies()
        remember_conversation(message, answer)
        return {"message": answer}

    # New recommendation requests must include location first.
    if intent == "recommendation" and has_any_param(params) and not has_location(params):
        _context["pending_question"] = {
            "intent": "recommendation",
            "params": params,
            "missing": "city_or_region",
        }
        cuisine_text = ""
        if params.get("cuisine"):
            cuisine_text = f" מסעדה בסגנון {cuisine_display(params.get('cuisine'))}"
        budget_text = ""
        if params.get("budget"):
            budget_text = f" עד {params.get('budget')} ₪"
        answer = f"מעולה 😊 באיזו עיר או אזור תרצי שאחפש{cuisine_text}{budget_text}?"
        remember_conversation(message, answer)
        return {"message": answer}

    chain_candidates = find_chain_candidates(message, pending_chain)
    chain_name = clean_name(chain_candidates.iloc[0].get("name_he") if not chain_candidates.empty else (pending_chain or "המסעדה"))

    # Branch-specific questions: ask for location if multiple branches and no location.
    if intent in ["reviews", "rating", "rating_trust", "peak_hours", "opening_hours", "recommended_dish", "menu", "similar_restaurants"] and not chain_candidates.empty:
        if should_ask_city_for_chain(intent, chain_candidates, params):
            answer = ask_city(chain_name, chain_candidates, intent, params)
            remember_conversation(message, answer)
            return {"message": answer}

        # במסעדות דומות העיר היא אזור החיפוש של ההמלצות, לא בהכרח סניף של מסעדת הבסיס.
        if intent == "similar_restaurants":
            base = choose_base_restaurant_for_similarity(chain_candidates)
            if base is None:
                answer = "לאיזו מסעדה תרצי שאחפש מסעדות דומות?"
            else:
                remember_restaurant(base)
                _context["pending_question"] = None
                answer = format_similar(base, params)
            remember_conversation(message, answer)
            return {"message": answer}

        branch = select_branch(chain_candidates, params)
        if branch is None:
            answer = format_no_branch_found(chain_name, params, chain_candidates, intent)
            # לא מחזירים דירוג/סניף אחר כאשר העיר לא קיימת בדאטה.
            _context["pending_question"] = {
                "intent": intent,
                "chain_key": message_chain_alias(chain_name) or normalize_text(chain_name).replace(" ", ""),
                "params": {k: v for k, v in params.items() if k not in ["city", "region"]},
                "missing": "valid_city_or_region",
            }
            remember_conversation(message, answer)
            return {"message": answer}
        remember_restaurant(branch)
        _context["pending_question"] = None
        if intent in ["reviews", "rating"]:
            answer = format_reviews(branch)
        elif intent == "rating_trust":
            answer = format_rating_trust(branch)
        elif intent == "peak_hours":
            answer = format_peak(branch, params)
        elif intent == "recommended_dish":
            answer = format_recommended_dish(branch)
        elif intent == "menu":
            menu = summarize_menu(branch)
            if menu.get("available"):
                items = "\n".join([f"• {x['item_name']} — ₪{int(x['price'])}" for x in menu.get("items", [])[:6]])
                answer = f"מצאתי כמה מנות ב{row_to_dict(branch)['name']}:\n{items}"
            else:
                answer = "לא מצאתי תפריט זמין למסעדה הזו."
        elif intent == "similar_restaurants":
            answer = format_similar(branch, params)
        else:
            answer = restaurant_card(branch)
        remember_conversation(message, answer)
        return {"message": answer}

    # Similar without identifiable restaurant
    if intent == "similar_restaurants":
        if _context.get("last_restaurant_id"):
            df = load_restaurants()
            found = df[df["restaurant_id"].astype(str) == str(_context["last_restaurant_id"])]
            if not found.empty:
                answer = format_similar(found.iloc[0], params)
            else:
                answer = "לאיזו מסעדה תרצי שאחפש מסעדות דומות?"
        else:
            answer = "לאיזו מסעדה תרצי שאחפש מסעדות דומות?"
        remember_conversation(message, answer)
        return {"message": answer}

    # General recommendations after location was supplied.
    if intent == "recommendation" and has_location(params):
        answer = format_recommendations(message, params)
        _context["pending_question"] = None
        remember_conversation(message, answer)
        return {"message": answer}

    # If no clear params but restaurant-related, let Gemini answer naturally.
    answer = natural_gemini_fallback(message)
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
    try:
        data = request.get_json(force=True)
        message = data.get("message", "")
        return jsonify(answer_free_text(message))
    except Exception as exc:
        print("API chat technical error:", exc)
        return jsonify({"message": "מצטערת, יש כרגע תקלה טכנית. נסי שוב בעוד רגע."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=True)
