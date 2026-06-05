from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

try:
    from openai import OpenAI
except ImportError:  # keeps the app alive locally even before pip install
    OpenAI = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESTAURANTS_CSV = os.path.join(BASE_DIR, "data", "smartbite_12000_restaurants_hebrew.csv")
ORDERS_CSV = os.path.join(BASE_DIR, "data", "orders.csv")
REVIEWS_CSV = os.path.join(BASE_DIR, "data", "reviews.csv")
CUSTOMERS_CSV = os.path.join(BASE_DIR, "data", "customers.csv")
MENU_ITEMS_CSV = os.path.join(BASE_DIR, "data", "menu_items.csv")

app = Flask(__name__)

_cache: Dict[str, Any] = {}
_context: Dict[str, Any] = {
    "conversation_history": [],
    "last_restaurant_id": None,
    "last_restaurant_name": None,
    "last_intent": None,
    "last_params": {},
    "pending_action": None,
    "user_age": None,
    "user_city": None,
    "user_region": None,
    "dietary_preference": None,
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


def clean_name(name: str) -> str:
    name = str(name or "")
    name = re.sub(r"\s*-\s*.*?Branch\s*\d+", "", name, flags=re.I)
    name = re.sub(r"\s*-\s*.*?סניף\s*\d+", "", name)
    return name.strip()


def detect_language(message: str) -> str:
    return "he" if re.search(r"[\u0590-\u05FF]", message or "") else "en"


def openai_model_name() -> str:
    return os.environ.get("OPENAI_MODEL", os.environ.get("OPENAI_ANSWER_MODEL", "gpt-4o-mini"))


def get_openai_client() -> Optional[OpenAI]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def remember_conversation(user_message: str, assistant_message: str) -> None:
    history = _context.setdefault("conversation_history", [])
    history.append({"user": user_message, "assistant": assistant_message})
    _context["conversation_history"] = history[-10:]


def json_from_model_text(text_value: str) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("").astype(str)
    return df


def load_restaurants() -> pd.DataFrame:
    if "restaurants" not in _cache:
        df = _read_csv_safe(RESTAURANTS_CSV)
        if not df.empty:
            if "rating" in df.columns:
                df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
            if "avg_price_per_person" in df.columns:
                df["avg_price_per_person"] = pd.to_numeric(
                    df["avg_price_per_person"], errors="coerce"
                ).fillna(0)
        _cache["restaurants"] = df
    return _cache["restaurants"].copy()


def load_reviews() -> pd.DataFrame:
    if "reviews" not in _cache:
        df = _read_csv_safe(REVIEWS_CSV)
        if not df.empty:
            if "review_date" in df.columns:
                df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
            if "rating" in df.columns:
                df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        _cache["reviews"] = df
    return _cache["reviews"].copy()


def load_orders() -> pd.DataFrame:
    if "orders" not in _cache:
        df = _read_csv_safe(ORDERS_CSV)
        if not df.empty and "order_datetime" in df.columns:
            df["order_datetime"] = pd.to_datetime(df["order_datetime"], errors="coerce")
        _cache["orders"] = df
    return _cache["orders"].copy()


def load_menu_items() -> pd.DataFrame:
    if "menu_items" not in _cache:
        df = _read_csv_safe(MENU_ITEMS_CSV)
        if not df.empty:
            for col in ["price", "popularity_score"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        _cache["menu_items"] = df
    return _cache["menu_items"].copy()


# ---------------------------------------------------------------------
# Mappings and simple extraction
# ---------------------------------------------------------------------

CITY_MAP = {
    "בתל אביב": "Tel Aviv", "תל אביב": "Tel Aviv", "תל-אביב": "Tel Aviv",
    "בתל-אביב": "Tel Aviv", "בת״א": "Tel Aviv", "ת״א": "Tel Aviv",
    "בת\"א": "Tel Aviv", "ת\"א": "Tel Aviv", "בתא": "Tel Aviv", "תא": "Tel Aviv",
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

CITY_HE_BY_EN = {
    "Tel Aviv": "תל אביב", "Jerusalem": "ירושלים", "Haifa": "חיפה",
    "Ramat Gan": "רמת גן", "Givatayim": "גבעתיים", "Holon": "חולון",
    "Rishon LeZion": "ראשון לציון", "Petah Tikva": "פתח תקווה",
    "Herzliya": "הרצליה", "Netanya": "נתניה", "Kfar Saba": "כפר סבא",
    "Ra'anana": "רעננה", "Rehovot": "רחובות", "Modiin": "מודיעין",
    "Ashdod": "אשדוד", "Ashkelon": "אשקלון", "Beer Sheva": "באר שבע",
    "Eilat": "אילת", "Nazareth": "נצרת", "Acre": "עכו",
    "Caesarea": "קיסריה", "Zichron Yaakov": "זכרון יעקב",
    "Tiberias": "טבריה", "Rosh Pina": "ראש פינה",
}

REGION_MAP = {
    "צפון": "North", "בצפון": "North", "north": "North",
    "מרכז": "Center", "במרכז": "Center", "center": "Center",
    "דרום": "South", "בדרום": "South", "south": "South",
    "ירושלים": "Jerusalem", "jerusalem": "Jerusalem",
}

CUISINE_MAP = {
    "איטלקי": "Italian", "איטלקית": "Italian", "pasta": "Italian", "פסטה": "Italian",
    "פיצה": "Pizza", "pizza": "Pizza",
    "יפני": "Japanese", "יפנית": "Japanese", "סושי": "Japanese", "sushi": "Japanese",
    "אסייתי": "Asian", "אסייתית": "Asian", "נודלס": "Asian", "asian": "Asian",
    "המבורגר": "Burgers", "בורגר": "Burgers", "burger": "Burgers",
    "טבעוני": "Vegan", "טבעונית": "Vegan", "vegan": "Vegan",
    "ישראלי": "Israeli", "ישראלית": "Israeli", "israeli": "Israeli",
    "ים תיכוני": "Mediterranean", "mediterranean": "Mediterranean",
    "ערבי": "Arab", "ערבית": "Arab",
    "יווני": "Greek", "יוונית": "Greek",
    "תאילנדי": "Thai", "thai": "Thai",
    "דגים": "Seafood", "פירות ים": "Seafood", "seafood": "Seafood",
    "סטייק": "Steakhouse", "steak": "Steakhouse",
    "קפה": "Cafe", "בית קפה": "Cafe", "cafe": "Cafe",
    "ארוחת בוקר": "Breakfast", "breakfast": "Breakfast",
}

RESTAURANT_ALIASES = {
    "מקדונלדס": ["מקדונלדס", "מקדונלד ס", "מקדונלד'ס", "מקדונלד׳ס", "מק דונלדס", "mcdonalds", "mcdonald's", "mc donalds"],
    "ארומה": ["ארומה", "aroma"],
    "קפה קפה": ["קפה קפה", "cafe cafe"],
    "גרג": ["גרג", "greg"],
    "לנדוור": ["לנדוור", "landwer"],
    "גפניקה": ["גפניקה", "ג'פניקה", "ג׳פניקה", "japanika"],
    "מחניודה": ["מחניודה", "machneyuda", "mahane yehuda", "machne yuda"],
    "אבו חסן": ["אבו חסן", "abu hassan"],
    "טאיזו": ["טאיזו", "taizu"],
    "שילה": ["שילה", "shila"],
    "קלארו": ["קלארו", "claro"],
    "אורי בורי": ["אורי בורי", "uri buri"],
}

RECIPE_WORDS = [
    "מתכון", "מתכונים", "איך מכינים", "איך להכין", "מצרכים", "רכיבים",
    "אופן הכנה", "recipe", "how to make", "ingredients", "cook", "cooking instructions"
]

OFF_TOPIC_EXAMPLES = [
    "רכב", "מכונית", "אוטו", "טלפון", "מחשב", "פוליטיקה", "מזג אוויר",
    "תרופה", "רופא", "קוד", "מטלה", "סרט", "טיסה", "מלון"
]


def city_to_region(city: Optional[str]) -> Optional[str]:
    if not city:
        return None
    center = {"Tel Aviv", "Ramat Gan", "Givatayim", "Holon", "Rishon LeZion", "Petah Tikva", "Herzliya", "Netanya", "Kfar Saba", "Ra'anana", "Rehovot", "Modiin"}
    north = {"Haifa", "Nazareth", "Acre", "Caesarea", "Zichron Yaakov", "Tiberias", "Rosh Pina"}
    south = {"Ashdod", "Ashkelon", "Beer Sheva", "Eilat"}
    if city in center:
        return "Center"
    if city in north:
        return "North"
    if city in south:
        return "South"
    if city == "Jerusalem":
        return "Jerusalem"
    return None


def extract_params_local(message: str) -> Dict[str, Any]:
    msg = message or ""
    msg_norm = normalize_text(msg)
    without_vav = msg.strip()[1:].strip() if msg.strip().startswith("ו") else msg.strip()

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

    for heb, eng in sorted(CITY_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if heb in msg or heb in without_vav or normalize_text(eng) in msg_norm:
            params["city"] = eng
            break

    for heb, eng in REGION_MAP.items():
        if heb in msg_norm or heb in msg:
            params["region"] = eng

    for key, eng in CUISINE_MAP.items():
        if normalize_text(key) in msg_norm or key in msg:
            params["cuisine"] = eng

    if any(w in msg_norm for w in ["כשר", "kosher"]):
        params["kosher"] = "Yes"
    if any(w in msg_norm for w in ["לא כשר", "בלי כשרות", "non kosher", "not kosher"]):
        params["kosher"] = "No"

    if any(w in msg_norm for w in ["זול", "זולה", "cheap"]):
        params["price_level"] = "Cheap"
    elif any(w in msg_norm for w in ["בינוני", "מחיר סביר", "medium"]):
        params["price_level"] = "Medium"
    elif any(w in msg_norm for w in ["יקר", "יוקרתי", "expensive"]):
        params["price_level"] = "Expensive"

    if any(w in msg_norm for w in ["שף", "chef"]):
        params["restaurant_type"] = "Chef Restaurant"
    elif any(w in msg_norm for w in ["בשרי", "בשרים", "meat"]):
        params["restaurant_type"] = "Meat"
    elif any(w in msg_norm for w in ["חלבי", "dairy"]):
        params["restaurant_type"] = "Dairy"
    elif any(w in msg_norm for w in ["טבעוני", "vegan"]):
        params["restaurant_type"] = "Vegan"
        params["cuisine"] = "Vegan"

    nums = [int(n) for n in re.findall(r"\d+", msg)]
    if nums and any(w in msg_norm for w in ["עד", "תקציב", "שקל", "₪", "budget", "under"]):
        params["budget"] = max(nums)

    rating_match = re.search(r"([0-5](?:\.\d+)?)", msg)
    if rating_match and any(w in msg_norm for w in ["דירוג", "rating", "מעל", "לפחות"]):
        params["rating_min"] = float(rating_match.group(1))

    if any(w in msg_norm for w in ["דייט", "זוג", "רומנטי", "date", "couple"]):
        params["suitable_for"] = "Couples"
    elif any(w in msg_norm for w in ["משפחה", "ילדים", "family"]):
        params["suitable_for"] = "Families"
    elif any(w in msg_norm for w in ["חברים", "סטודנטים", "friends"]):
        params["suitable_for"] = "Friends"
    elif any(w in msg_norm for w in ["עסקי", "פגישה", "business"]):
        params["suitable_for"] = "Business"

    return params


def merge_params(ai_params: Dict[str, Any], local_params: Dict[str, Any]) -> Dict[str, Any]:
    merged = {}
    for key in [
        "city", "region", "cuisine", "restaurant_type", "kosher",
        "price_level", "budget", "rating_min", "dish", "suitable_for"
    ]:
        merged[key] = ai_params.get(key) or local_params.get(key)
    return merged


def has_any_params(params: Dict[str, Any]) -> bool:
    return any(value not in [None, "", False] for value in params.values())


# ---------------------------------------------------------------------
# User profile memory
# ---------------------------------------------------------------------

def age_to_group(age: int) -> str:
    if age <= 18:
        return "Teen"
    if age <= 30:
        return "Young Adult"
    if age <= 50:
        return "Adult"
    return "Senior"


def update_user_profile_from_message(message: str, parsed: Optional[Dict[str, Any]] = None) -> None:
    msg = message.lower()

    age_match = re.search(r"(?:בן|בת|גיל)\s*(\d{1,2})", msg)
    if age_match:
        age = int(age_match.group(1))
        _context["user_age"] = age

    for heb_city, eng_city in sorted(CITY_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        if heb_city in message:
            _context["user_city"] = eng_city
            _context["user_region"] = city_to_region(eng_city)
            break

    if parsed:
        if parsed.get("age"):
            try:
                _context["user_age"] = int(parsed["age"])
            except Exception:
                pass
        if parsed.get("city"):
            _context["user_city"] = parsed["city"]
            _context["user_region"] = city_to_region(parsed["city"])
        if parsed.get("region"):
            _context["user_region"] = parsed["region"]

    if any(w in msg for w in ["טבעוני", "טבעונית", "vegan"]):
        _context["dietary_preference"] = "vegan"
    elif any(w in msg for w in ["צמחוני", "צמחונית", "vegetarian"]):
        _context["dietary_preference"] = "vegetarian"
    elif any(w in msg for w in ["ללא גלוטן", "בלי גלוטן", "gluten free"]):
        _context["dietary_preference"] = "gluten_free"
    elif any(w in msg for w in ["כשר", "כשרה", "כשרות", "kosher"]):
        _context["dietary_preference"] = "kosher"


def apply_user_profile(params: Dict[str, Any]) -> Dict[str, Any]:
    result = params.copy()

    if not result.get("city") and not result.get("region"):
        if _context.get("user_city"):
            result["city"] = _context["user_city"]
        elif _context.get("user_region"):
            result["region"] = _context["user_region"]

    dietary = _context.get("dietary_preference")
    if dietary == "vegan":
        result["cuisine"] = result.get("cuisine") or "Vegan"
        result["restaurant_type"] = result.get("restaurant_type") or "Vegan"
    elif dietary == "kosher":
        result["kosher"] = result.get("kosher") or "Yes"

    if _context.get("user_age") and int(_context["user_age"]) <= 18:
        result["budget"] = result.get("budget") or 120

    return result


# ---------------------------------------------------------------------
# Restaurant lookup and internal data search
# ---------------------------------------------------------------------

def remember_restaurant(row: pd.Series) -> None:
    _context["last_restaurant_id"] = int(row.get("restaurant_id", 0))
    _context["last_restaurant_name"] = clean_name(str(row.get("name_he") or row.get("name")))


def get_last_restaurant() -> Optional[pd.Series]:
    rid = _context.get("last_restaurant_id")
    if not rid:
        return None
    df = load_restaurants()
    found = df[df["restaurant_id"].astype(int) == int(rid)]
    return found.iloc[0] if not found.empty else None


def restaurant_row_to_dict(row: Optional[pd.Series]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {
        "restaurant_id": int(row.get("restaurant_id", 0)),
        "name": str(row.get("name_he") or row.get("name")),
        "clean_name": clean_name(str(row.get("name_he") or row.get("name"))),
        "city": str(row.get("city_he") or row.get("city")),
        "city_en": str(row.get("city", "")),
        "region": str(row.get("region_he") or row.get("region")),
        "region_en": str(row.get("region", "")),
        "cuisine": str(row.get("cuisine_he") or row.get("cuisine")),
        "restaurant_type": str(row.get("restaurant_type_he") or row.get("restaurant_type")),
        "kosher": "כשר" if str(row.get("kosher", "")).lower() == "yes" else "לא כשר",
        "avg_price_per_person": float(row.get("avg_price_per_person", 0) or 0),
        "price_level": str(row.get("price_level", "")),
        "rating": float(row.get("rating", 0) or 0),
        "suitable_for": str(row.get("suitable_for_he") or row.get("suitable_for")),
        "recommended_dish": str(row.get("recommended_dish_he") or row.get("recommended_dish")),
    }


def records_from_df(df: pd.DataFrame, limit: int = 6) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    return [restaurant_row_to_dict(row) for _, row in df.head(limit).iterrows()]


def find_restaurant_by_name(name_or_message: Optional[str]) -> Optional[pd.Series]:
    if not name_or_message:
        return None

    df = load_restaurants()
    if df.empty:
        return None

    msg_norm = normalize_text(name_or_message)
    msg_compact = msg_norm.replace(" ", "")

    candidates_aliases: List[Tuple[str, List[str]]] = []
    for canonical, aliases in RESTAURANT_ALIASES.items():
        candidates_aliases.append((canonical, aliases + [canonical]))

    for canonical, aliases in candidates_aliases:
        alias_norms = [normalize_text(a) for a in aliases]
        alias_compacts = [a.replace(" ", "") for a in alias_norms]
        if any(a in msg_norm or ac in msg_compact for a, ac in zip(alias_norms, alias_compacts)):
            for _, row in df.iterrows():
                names = [
                    str(row.get("name", "")),
                    str(row.get("name_he", "")),
                    clean_name(str(row.get("name", ""))),
                    clean_name(str(row.get("name_he", ""))),
                ]
                row_norms = [normalize_text(n) for n in names if n]
                row_compacts = [rn.replace(" ", "") for rn in row_norms]
                if any(
                    a in rn or rn in a or ac in rc or rc in ac
                    for a, ac in zip(alias_norms, alias_compacts)
                    for rn, rc in zip(row_norms, row_compacts)
                ):
                    remember_restaurant(row)
                    return row

    # generic search
    for _, row in df.iterrows():
        names = [
            str(row.get("name", "")),
            str(row.get("name_he", "")),
            clean_name(str(row.get("name", ""))),
            clean_name(str(row.get("name_he", ""))),
        ]
        for name in names:
            n_norm = normalize_text(name)
            n_compact = n_norm.replace(" ", "")
            if len(n_norm) >= 2 and (n_norm in msg_norm or n_compact in msg_compact):
                remember_restaurant(row)
                return row

    return None


def find_restaurant_in_location(base_restaurant: Optional[pd.Series], params: Dict[str, Any]) -> Optional[pd.Series]:
    if base_restaurant is None:
        return None

    df = load_restaurants()
    base_name = clean_name(str(base_restaurant.get("name_he") or base_restaurant.get("name")))
    chain_key = re.split(r"\s*-\s*", base_name)[0].strip()
    chain_norm = normalize_text(chain_key)

    candidates = df[
        df["name_he"].astype(str).apply(lambda x: chain_norm in normalize_text(clean_name(x)))
        | df["name"].astype(str).apply(lambda x: chain_norm in normalize_text(clean_name(x)))
    ].copy()

    if candidates.empty:
        candidates = df.copy()

    if params.get("city"):
        by_city = candidates[candidates["city"].astype(str).str.lower() == str(params["city"]).lower()]
        if not by_city.empty:
            remember_restaurant(by_city.iloc[0])
            return by_city.iloc[0]

    if params.get("region"):
        by_region = candidates[candidates["region"].astype(str).str.lower() == str(params["region"]).lower()]
        if not by_region.empty:
            remember_restaurant(by_region.iloc[0])
            return by_region.iloc[0]

    return None


def apply_filters(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    result = df.copy()

    if result.empty:
        return result

    if params.get("city") and "city" in result.columns:
        result = result[result["city"].astype(str).str.lower() == str(params["city"]).lower()]
    if params.get("region") and "region" in result.columns:
        result = result[result["region"].astype(str).str.lower() == str(params["region"]).lower()]
    if params.get("cuisine") and "cuisine" in result.columns:
        result = result[result["cuisine"].astype(str).str.lower() == str(params["cuisine"]).lower()]

    if params.get("restaurant_type") and "restaurant_type" in result.columns:
        rt = str(params["restaurant_type"]).lower()
        if rt == "chef restaurant":
            result = result[
                result["restaurant_type"].astype(str).str.lower().str.contains("chef", na=False)
                | result.get("restaurant_type_he", pd.Series("", index=result.index)).astype(str).str.contains("שף", na=False)
            ]
        else:
            result = result[result["restaurant_type"].astype(str).str.lower() == rt]

    if params.get("kosher") and "kosher" in result.columns:
        result = result[result["kosher"].astype(str).str.lower() == str(params["kosher"]).lower()]

    if params.get("price_level") and "price_level" in result.columns:
        result = result[result["price_level"].astype(str).str.lower() == str(params["price_level"]).lower()]

    if params.get("budget") and "avg_price_per_person" in result.columns:
        result = result[result["avg_price_per_person"] <= float(params["budget"])]

    if params.get("rating_min") and "rating" in result.columns:
        result = result[result["rating"] >= float(params["rating_min"])]

    if params.get("suitable_for") and "suitable_for" in result.columns:
        sf = str(params["suitable_for"])
        result = result[
            result["suitable_for"].astype(str).str.contains(sf, case=False, na=False)
            | result.get("suitable_for_he", pd.Series("", index=result.index)).astype(str).str.contains(sf, case=False, na=False)
        ]

    return result


def recommend_restaurants(params: Dict[str, Any], limit: int = 6) -> pd.DataFrame:
    df = load_restaurants()
    filtered = apply_filters(df, params)
    if filtered.empty:
        return filtered

    filtered = filtered.copy()
    known = (
        filtered.get("is_known_recommended", pd.Series("", index=filtered.index))
        .astype(str).str.lower().eq("yes").astype(int)
    )
    filtered["score"] = filtered["rating"] * 2.0 + known * 1.5 - filtered["avg_price_per_person"] / 1000
    return filtered.sort_values(["score", "rating"], ascending=False).head(limit)


def fallback_recommendations(params: Dict[str, Any], limit: int = 6) -> pd.DataFrame:
    relax_steps = [
        ["budget", "price_level", "suitable_for"],
        ["budget", "price_level", "suitable_for", "kosher"],
    ]

    for remove_keys in relax_steps:
        relaxed = params.copy()
        for k in remove_keys:
            relaxed[k] = None
        recs = recommend_restaurants(relaxed, limit)
        if not recs.empty:
            return recs

    if params.get("city"):
        region = city_to_region(params["city"])
        if region:
            relaxed = params.copy()
            relaxed["city"] = None
            relaxed["region"] = region
            for k in ["budget", "price_level", "suitable_for", "kosher"]:
                relaxed[k] = None
            recs = recommend_restaurants(relaxed, limit)
            if not recs.empty:
                return recs

    relaxed = params.copy()
    for k in ["city", "region", "budget", "price_level", "suitable_for", "kosher"]:
        relaxed[k] = None
    return recommend_restaurants(relaxed, limit)


def get_reviews_summary(restaurant: Optional[pd.Series]) -> Dict[str, Any]:
    if restaurant is None:
        return {"found": False}

    reviews = load_reviews()
    if reviews.empty or "restaurant_id" not in reviews.columns:
        return {"found": False, "reason": "reviews_table_missing"}

    rid = str(int(restaurant["restaurant_id"]))
    r = reviews[reviews["restaurant_id"].astype(str) == rid].copy()
    if r.empty:
        return {"found": False, "restaurant": restaurant_row_to_dict(restaurant)}

    summary: Dict[str, Any] = {
        "found": True,
        "restaurant": restaurant_row_to_dict(restaurant),
        "review_count": int(len(r)),
    }

    if "rating" in r.columns:
        avg = r["rating"].dropna().mean()
        if not pd.isna(avg):
            summary["average_review_rating"] = round(float(avg), 2)

    if "sentiment" in r.columns:
        counts = r["sentiment"].astype(str).str.lower().value_counts().to_dict()
        summary["sentiment_counts"] = {k: int(v) for k, v in counts.items()}
        if counts:
            summary["dominant_sentiment"] = max(counts, key=counts.get)

    if "review_text" in r.columns:
        summary["sample_reviews"] = r["review_text"].dropna().astype(str).head(3).tolist()

    return summary


def get_rating_summary(restaurant: Optional[pd.Series]) -> Dict[str, Any]:
    if restaurant is None:
        return {"found": False}
    summary = {
        "found": True,
        "restaurant": restaurant_row_to_dict(restaurant),
        "restaurant_table_rating": float(restaurant.get("rating", 0) or 0),
    }
    review_summary = get_reviews_summary(restaurant)
    if review_summary.get("found"):
        summary["review_average_rating"] = review_summary.get("average_review_rating")
        summary["review_count"] = review_summary.get("review_count")
    return summary


def get_peak_hours_summary(restaurant: Optional[pd.Series]) -> Dict[str, Any]:
    if restaurant is None:
        return {"found": False}

    orders = load_orders()
    if orders.empty or "restaurant_id" not in orders.columns or "order_datetime" not in orders.columns:
        return {"found": False, "restaurant": restaurant_row_to_dict(restaurant), "reason": "orders_missing"}

    rid = str(int(restaurant["restaurant_id"]))
    order_rows = orders[orders["restaurant_id"].astype(str) == rid].copy()
    order_rows = order_rows.dropna(subset=["order_datetime"])
    if order_rows.empty:
        return {"found": False, "restaurant": restaurant_row_to_dict(restaurant), "reason": "not_enough_orders"}

    order_rows["hour"] = order_rows["order_datetime"].dt.hour
    busiest = order_rows["hour"].value_counts().sort_values(ascending=False).head(3)
    quiet = order_rows["hour"].value_counts().sort_values(ascending=True).head(2)

    return {
        "found": True,
        "restaurant": restaurant_row_to_dict(restaurant),
        "busiest_hours": [
            {"hour_range": f"{int(hour):02d}:00-{(int(hour)+1)%24:02d}:00", "orders": int(count)}
            for hour, count in busiest.items()
        ],
        "quiet_hours": [f"{int(hour):02d}:00" for hour in quiet.index],
        "orders_count": int(len(order_rows)),
    }


def get_menu_summary(restaurant: Optional[pd.Series]) -> Dict[str, Any]:
    if restaurant is None:
        return {"found": False}

    menu = load_menu_items()
    if menu.empty or "restaurant_id" not in menu.columns:
        return {"found": False, "restaurant": restaurant_row_to_dict(restaurant), "reason": "menu_missing"}

    rid = str(int(restaurant["restaurant_id"]))
    items = menu[menu["restaurant_id"].astype(str) == rid].copy()
    if items.empty:
        return {"found": False, "restaurant": restaurant_row_to_dict(restaurant), "reason": "no_menu_items"}

    items = items.sort_values("popularity_score", ascending=False)
    top_items = []
    for _, row in items.head(5).iterrows():
        top_items.append({
            "item_name": str(row.get("item_name", "")),
            "category": str(row.get("category", "")),
            "price": float(row.get("price", 0) or 0),
            "is_vegan": str(row.get("is_vegan", "")),
            "is_gluten_free": str(row.get("is_gluten_free", "")),
            "popularity_score": float(row.get("popularity_score", 0) or 0),
        })

    return {
        "found": True,
        "restaurant": restaurant_row_to_dict(restaurant),
        "top_menu_items": top_items,
    }


# ---------------------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------------------

def is_similarity_question(message: str) -> bool:
    msg = normalize_text(message)
    return any(w in msg for w in ["דומה", "דומות", "דומים", "בסגנון", "כמו", "similar", "like"])


def cosine_similarity_score(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").fillna(0)
    b = pd.to_numeric(b, errors="coerce").fillna(0)
    numerator = float((a * b).sum())
    denominator = float((a.pow(2).sum() ** 0.5) * (b.pow(2).sum() ** 0.5))
    return 0.0 if denominator == 0 else numerator / denominator


def recommend_similar_restaurants(restaurant: Optional[pd.Series], limit: int = 5) -> pd.DataFrame:
    if restaurant is None:
        return pd.DataFrame()

    df = load_restaurants()
    if df.empty:
        return pd.DataFrame()

    feature_cols = ["cuisine", "restaurant_type", "region", "kosher", "price_level", "suitable_for"]
    available = [c for c in feature_cols if c in df.columns]

    categorical = pd.get_dummies(df[available].fillna("").astype(str), columns=available)
    numeric = pd.DataFrame(index=df.index)

    if "rating" in df.columns:
        numeric["rating_norm"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0) / 5.0
    if "avg_price_per_person" in df.columns:
        price = pd.to_numeric(df["avg_price_per_person"], errors="coerce").fillna(0)
        max_price = price.max()
        numeric["price_norm"] = price / max_price if max_price and max_price > 0 else 0

    matrix = pd.concat([categorical, numeric], axis=1)
    target_id = int(restaurant["restaurant_id"])
    target_rows = df[df["restaurant_id"].astype(int) == target_id]
    if target_rows.empty:
        return pd.DataFrame()

    target_idx = target_rows.index[0]
    target_vec = matrix.loc[target_idx]

    scores = []
    for idx, row in matrix.iterrows():
        current_id = int(df.loc[idx, "restaurant_id"])
        if current_id == target_id:
            continue
        scores.append((idx, cosine_similarity_score(target_vec, row)))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:limit]
    if not scores:
        return pd.DataFrame()

    result = df.loc[[idx for idx, _ in scores]].copy()
    result["similarity_score"] = [score for _, score in scores]
    return result


# ---------------------------------------------------------------------
# Google Places fallback
# ---------------------------------------------------------------------

def google_places_key() -> str:
    return os.environ.get("GOOGLE_PLACES_API_KEY", "").strip()


def search_google_places(query: str, language: str = "he", limit: int = 5) -> Dict[str, Any]:
    api_key = google_places_key()
    if not api_key or not query:
        return {"found": False, "reason": "missing_google_places_api_key_or_query"}

    # New Places API
    try:
        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": (
                "places.displayName,places.formattedAddress,places.rating,"
                "places.userRatingCount,places.priceLevel,places.regularOpeningHours,"
                "places.currentOpeningHours,places.internationalPhoneNumber,"
                "places.websiteUri,places.googleMapsUri,places.businessStatus"
            ),
        }
        body = {
            "textQuery": query,
            "languageCode": "he" if language == "he" else "en",
            "regionCode": "IL",
            "maxResultCount": min(max(limit, 1), 10),
        }
        response = requests.post(url, headers=headers, json=body, timeout=8)
        if response.ok:
            data = response.json()
            places = data.get("places", [])
            if places:
                return {"found": True, "source": "Google Places API", "query": query, "places": places[:limit]}
    except Exception as exc:
        print("Google Places new API error:", exc)

    # Legacy Text Search fallback
    try:
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            "query": query,
            "key": api_key,
            "language": "iw" if language == "he" else "en",
            "region": "il",
        }
        response = requests.get(url, params=params, timeout=8)
        if response.ok:
            data = response.json()
            results = data.get("results", [])
            if results:
                places = []
                for item in results[:limit]:
                    places.append({
                        "displayName": {"text": item.get("name")},
                        "formattedAddress": item.get("formatted_address"),
                        "rating": item.get("rating"),
                        "userRatingCount": item.get("user_ratings_total"),
                        "businessStatus": item.get("business_status"),
                        "googleMapsUri": None,
                    })
                return {"found": True, "source": "Google Places Legacy Text Search", "query": query, "places": places}
            return {"found": False, "source": "Google Places", "query": query, "status": data.get("status"), "error_message": data.get("error_message")}
    except Exception as exc:
        print("Google Places legacy API error:", exc)

    return {"found": False, "query": query}


def build_external_query(message: str, parsed: Dict[str, Any], params: Dict[str, Any], restaurant_name: Optional[str]) -> str:
    parts = []

    if restaurant_name:
        parts.append(clean_name(restaurant_name))
    elif parsed.get("restaurant_name"):
        parts.append(str(parsed["restaurant_name"]))

    if params.get("city"):
        parts.append(CITY_HE_BY_EN.get(params["city"], params["city"]))
    elif params.get("region"):
        parts.append(str(params["region"]))

    if parsed.get("cuisine") or params.get("cuisine"):
        parts.append(str(parsed.get("cuisine") or params.get("cuisine")))

    if not parts:
        parts.append(message)

    return " ".join(parts) + " מסעדה ישראל"


# ---------------------------------------------------------------------
# ChatGPT intent parser and final answer
# ---------------------------------------------------------------------

def local_intent_guess(message: str) -> Dict[str, Any]:
    msg = normalize_text(message)
    language = detect_language(message)

    local_params = extract_params_local(message)
    intent = "general_restaurant_question"
    related = True

    if any(w in msg for w in RECIPE_WORDS):
        return {"intent": "off_topic", "language": language, "is_restaurant_related": False, "reason": "recipe"}

    if any(w in msg for w in OFF_TOPIC_EXAMPLES):
        return {"intent": "off_topic", "language": language, "is_restaurant_related": False}

    if msg in ["היי", "הי", "שלום", "אהלן", "hi", "hello", "hey"] or any(w in msg for w in ["מה קורה", "מה נשמע"]):
        intent = "greeting"
    elif any(w in msg for w in ["תודה", "thanks"]):
        intent = "thanks"
    elif is_similarity_question(message):
        intent = "similar_restaurants"
    elif any(w in msg for w in ["ביקורת", "ביקורות", "חוות דעת", "reviews"]):
        intent = "reviews"
    elif any(w in msg for w in ["דירוג", "rating"]):
        intent = "rating"
    elif any(w in msg for w in ["עומס", "עמוס", "שעות עומס", "מתי כדאי להגיע"]):
        intent = "peak_hours"
    elif any(w in msg for w in ["שעות פתיחה", "פתוח", "סגור", "opening hours", "open now"]):
        intent = "opening_hours"
    elif any(w in msg for w in ["כתובת", "איפה נמצא", "address", "location"]):
        intent = "location"
    elif any(w in msg for w in ["מנה", "תפריט", "מה להזמין", "menu", "dish"]):
        intent = "recommended_dish"
    elif any(w in msg for w in ["תמליץ", "המלצה", "מסעדה", "איפה לאכול", "לאכול", "רעב", "רעבה", "restaurant", "recommend"]):
        intent = "recommendation"
    elif not any(v for v in local_params.values()):
        # Let GPT handle friendly/general restaurant talk when available
        intent = "unclear"

    return {
        "intent": intent,
        "language": language,
        "is_restaurant_related": related,
        "restaurant_name": None,
        **local_params,
    }


def parse_user_message_with_ai(message: str) -> Dict[str, Any]:
    client = get_openai_client()
    if client is None:
        return local_intent_guess(message)

    history = _context.get("conversation_history", [])[-8:]
    profile = {
        "user_age": _context.get("user_age"),
        "user_city": _context.get("user_city"),
        "user_region": _context.get("user_region"),
        "dietary_preference": _context.get("dietary_preference"),
        "last_restaurant_name": _context.get("last_restaurant_name"),
        "last_intent": _context.get("last_intent"),
        "pending_action": _context.get("pending_action"),
    }

    system_prompt = """
You are the conversation brain and intent parser for SmartBite, an AI restaurant agent in Israel.
Return ONLY valid JSON.

SmartBite should behave like a natural ChatGPT-style assistant, but only inside the restaurant/food-outside-home domain.

Important domain rules:
- Restaurant recommendations, restaurant reviews, ratings, peak hours, opening hours, address/location, dishes, menu items, kosher, prices, cities/regions, cuisines, and similar restaurants are in-domain.
- Recipes are NOT in-domain. If the user asks how to cook or asks for a recipe, classify as off_topic, even if the food is pasta/sushi/etc.
- Cars, phones, coding, schoolwork, medicine, politics, weather, movies, flights, hotels, and general non-restaurant questions are off_topic.
- If the user replies with only a city/region and pending_action exists, infer the previous intent from pending_action.
- If the user uses "it", "there", "that restaurant", or asks a follow-up, use conversation history and last_restaurant_name.

Supported intents:
greeting, thanks, profile_update, recommendation, reviews, rating, peak_hours,
opening_hours, location, recommended_dish, menu, similar_restaurants,
general_restaurant_question, off_topic, unclear.

Return JSON:
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
  "age": number|null,
  "dietary_preference": "vegan"|"vegetarian"|"gluten_free"|"kosher"|null,
  "needs_external": boolean,
  "missing_info": string|null
}
"""

    try:
        response = client.chat.completions.create(
            model=openai_model_name(),
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "message": message,
                            "conversation_history": history,
                            "context": profile,
                            "local_guess": local_intent_guess(message),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        parsed = json_from_model_text(response.choices[0].message.content or "")
        if parsed:
            local = local_intent_guess(message)
            # Force recipe/off-topic local guard even if model is too broad
            if local.get("is_restaurant_related") is False:
                parsed["intent"] = "off_topic"
                parsed["is_restaurant_related"] = False
            # Force similarity if clear
            if is_similarity_question(message):
                parsed["intent"] = "similar_restaurants"
                parsed["is_restaurant_related"] = True
            return parsed
    except Exception as exc:
        print("OpenAI parser error:", exc)

    return local_intent_guess(message)


def ai_final_answer(user_message: str, parsed: Dict[str, Any], data_payload: Dict[str, Any]) -> str:
    client = get_openai_client()
    language = parsed.get("language") or detect_language(user_message)

    fallback = data_payload.get("fallback_answer")
    if client is None:
        return fallback or (
            "אני יכולה לעזור רק בנושאי מסעדות ואוכל בישראל 🍽️"
            if language == "he" else
            "I can help only with restaurants and food in Israel 🍽️"
        )

    system_prompt = """
You are SmartBite, a ChatGPT-style AI Agent for restaurants in Israel.

Your role:
- Manage the conversation naturally.
- The internal database is the first and preferred source.
- If internal data is provided, use it as the factual source.
- If external Google Places data is provided, use it and mention naturally that the information comes from an external source only when relevant.
- If both internal and external data are missing, you may answer from your general restaurant-domain knowledge, but do not invent specific live facts such as current opening hours, exact ratings, or addresses. Ask a useful follow-up if needed.
- Never answer non-restaurant topics. Politely say you specialize in restaurants/food.
- Recipes are out of scope; offer to find a restaurant that serves that dish instead.
- Preserve names, cities, ratings, prices, hours, and dishes exactly as supplied in data.
- Answer in Hebrew if language is "he", English if language is "en".
- Sound natural, warm, and helpful. Avoid robotic template phrases.
- Keep the answer concise but useful.
- If the payload contains Cosine Similarity results, briefly explain they were found by comparing restaurant features.
"""

    try:
        response = client.chat.completions.create(
            model=openai_model_name(),
            temperature=0.75,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_message": user_message,
                            "parsed_intent": parsed,
                            "conversation_history": _context.get("conversation_history", [])[-10:],
                            "known_context": {
                                "last_restaurant": _context.get("last_restaurant_name"),
                                "user_city": _context.get("user_city"),
                                "user_region": _context.get("user_region"),
                                "dietary_preference": _context.get("dietary_preference"),
                            },
                            "data_payload": data_payload,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        answer = response.choices[0].message.content
        if answer and answer.strip():
            return answer.strip()
    except Exception as exc:
        print("OpenAI final answer error:", exc)

    return fallback or "אני רוצה לעזור 😊 אפשר לנסח שוב מה תרצי לדעת על מסעדות?"


# ---------------------------------------------------------------------
# Agent orchestration: ChatGPT conversation -> internal DB -> external -> final answer
# ---------------------------------------------------------------------

def external_payload_for_query(message: str, parsed: Dict[str, Any], params: Dict[str, Any], restaurant: Optional[pd.Series]) -> Dict[str, Any]:
    restaurant_name = None
    if restaurant is not None:
        restaurant_name = clean_name(str(restaurant.get("name_he") or restaurant.get("name")))
    elif parsed.get("restaurant_name"):
        restaurant_name = str(parsed.get("restaurant_name"))
    elif _context.get("last_restaurant_name"):
        restaurant_name = str(_context.get("last_restaurant_name"))

    query = build_external_query(message, parsed, params, restaurant_name)
    google = search_google_places(query, parsed.get("language") or detect_language(message), limit=5)
    return {
        "type": "external_search",
        "query": query,
        "google_places": google,
    }


def build_internal_response(message: str, parsed: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Returns (payload, found_sufficient_internal_data).
    If False, the caller will try Google Places / GPT restaurant-domain knowledge.
    """
    language = parsed.get("language") or detect_language(message)
    intent = parsed.get("intent", "unclear")
    local_params = extract_params_local(message)
    ai_params = {k: parsed.get(k) for k in local_params.keys()}
    params = merge_params(ai_params, local_params)

    # Fill follow-up params from pending context
    pending = _context.get("pending_action")
    if pending and (params.get("city") or params.get("region")) and not parsed.get("restaurant_name"):
        intent = pending.get("intent") or intent
        if pending.get("restaurant_id"):
            restaurant = get_restaurant_by_id(int(pending["restaurant_id"]))
        else:
            restaurant = get_last_restaurant()
    else:
        restaurant = find_restaurant_by_name(parsed.get("restaurant_name") or message) or get_last_restaurant()

    update_user_profile_from_message(message, parsed)

    if parsed.get("is_restaurant_related") is False or intent == "off_topic":
        return {
            "type": "off_topic",
            "fallback_answer": (
                "אני מתמחה במסעדות ואוכל בישראל 🍽️ אם תרצי, אוכל לעזור עם המלצה למסעדה, ביקורות, שעות פתיחה או מנה מומלצת."
                if language == "he" else
                "I specialize in restaurants and food in Israel 🍽️ I can help with recommendations, reviews, opening hours, or dishes."
            )
        }, True

    if intent in ["greeting", "thanks", "profile_update"]:
        return {
            "type": intent,
            "capabilities": [
                "restaurant recommendations", "reviews", "ratings", "peak hours",
                "opening hours via external search", "recommended dishes", "similar restaurants with Cosine Similarity"
            ],
            "fallback_answer": (
                "היי 😊 אני SmartBite. אפשר לדבר איתי טבעי על מסעדות — המלצות, ביקורות, מנות, שעות עומס או מסעדות דומות."
                if language == "he" else
                "Hi 😊 I’m SmartBite. You can ask me naturally about restaurants, recommendations, reviews, dishes, peak hours, or similar places."
            )
        }, True

    # Similarity model
    if intent == "similar_restaurants" or is_similarity_question(message):
        if restaurant is None:
            return {
                "type": "missing_restaurant_for_similarity",
                "model": "Cosine Similarity",
                "fallback_answer": "לאיזו מסעדה תרצי שאמצא מסעדות דומות?"
            }, True

        similar_df = recommend_similar_restaurants(restaurant, limit=5)
        if similar_df.empty:
            return {
                "type": "similar_restaurants",
                "model": "Cosine Similarity",
                "base_restaurant": restaurant_row_to_dict(restaurant),
                "restaurants": [],
                "fallback_answer": "מצאתי את המסעדה, אבל אין לי מספיק נתונים כדי לחשב מסעדות דומות."
            }, False

        return {
            "type": "similar_restaurants",
            "model": "Cosine Similarity",
            "base_restaurant": restaurant_row_to_dict(restaurant),
            "restaurants": records_from_df(similar_df, limit=5),
            "similarity_scores": [
                {"name": str(row.get("name_he") or row.get("name")), "score": round(float(row.get("similarity_score", 0)) * 100, 1)}
                for _, row in similar_df.head(5).iterrows()
            ],
        }, True

    # Reviews
    if intent == "reviews":
        if restaurant is None:
            _context["pending_action"] = {"intent": "reviews"}
            return {"type": "missing_restaurant", "fallback_answer": "על איזו מסעדה תרצי שאבדוק ביקורות?"}, True

        if not params.get("city") and not params.get("region"):
            _context["pending_action"] = {"intent": "reviews", "restaurant_id": int(restaurant["restaurant_id"])}
            return {
                "type": "missing_location",
                "restaurant": restaurant_row_to_dict(restaurant),
                "fallback_answer": f"באיזו עיר או אזור תרצי לבדוק ביקורות על {clean_name(str(restaurant.get('name_he') or restaurant.get('name')))}?"
            }, True

        restaurant_for_city = find_restaurant_in_location(restaurant, params)
        if restaurant_for_city is None:
            return {
                "type": "internal_branch_not_found",
                "intent": "reviews",
                "base_restaurant": restaurant_row_to_dict(restaurant),
                "requested_params": params,
                "fallback_answer": "לא מצאתי את הסניף המבוקש בדאטה הפנימי."
            }, False

        _context["pending_action"] = None
        summary = get_reviews_summary(restaurant_for_city)
        return {"type": "reviews", "summary": summary}, bool(summary.get("found"))

    # Rating
    if intent == "rating":
        if restaurant is None:
            return {"type": "missing_restaurant", "fallback_answer": "על איזו מסעדה תרצי שאבדוק דירוג?"}, True
        return {"type": "rating", "summary": get_rating_summary(restaurant)}, True

    # Peak hours from internal orders
    if intent == "peak_hours":
        if restaurant is None:
            _context["pending_action"] = {"intent": "peak_hours"}
            return {"type": "missing_restaurant", "fallback_answer": "על איזו מסעדה תרצי לבדוק שעות עומס?"}, True

        if not params.get("city") and not params.get("region"):
            _context["pending_action"] = {"intent": "peak_hours", "restaurant_id": int(restaurant["restaurant_id"])}
            return {
                "type": "missing_location",
                "restaurant": restaurant_row_to_dict(restaurant),
                "fallback_answer": f"באיזו עיר או אזור תרצי לבדוק את שעות העומס של {clean_name(str(restaurant.get('name_he') or restaurant.get('name')))}?"
            }, True

        restaurant_for_city = find_restaurant_in_location(restaurant, params)
        if restaurant_for_city is None:
            return {
                "type": "internal_branch_not_found",
                "intent": "peak_hours",
                "base_restaurant": restaurant_row_to_dict(restaurant),
                "requested_params": params,
                "fallback_answer": "לא מצאתי את הסניף המבוקש בדאטה הפנימי."
            }, False

        _context["pending_action"] = None
        summary = get_peak_hours_summary(restaurant_for_city)
        return {"type": "peak_hours", "summary": summary}, bool(summary.get("found"))

    # Opening hours/address are usually external if not in internal data
    if intent in ["opening_hours", "location"]:
        if restaurant is None and not parsed.get("restaurant_name") and not _context.get("last_restaurant_name"):
            return {"type": "missing_restaurant", "fallback_answer": "על איזו מסעדה תרצי שאבדוק את המידע הזה?"}, True
        return {
            "type": "internal_data_not_available",
            "intent": intent,
            "restaurant": restaurant_row_to_dict(restaurant) if restaurant is not None else None,
            "fallback_answer": "המידע הזה לא נמצא בדאטה הפנימי."
        }, False

    # Menu / recommended dish
    if intent in ["recommended_dish", "menu"]:
        if restaurant is None:
            restaurant = get_last_restaurant()
        if restaurant is None:
            return {"type": "missing_restaurant", "fallback_answer": "על איזו מסעדה תרצי שאבדוק מנה או תפריט?"}, True

        menu_summary = get_menu_summary(restaurant)
        if menu_summary.get("found"):
            return {"type": "menu", "summary": menu_summary}, True

        # fallback to restaurant table recommended dish
        dish = str(restaurant.get("recommended_dish_he") or restaurant.get("recommended_dish") or "")
        if dish:
            return {"type": "recommended_dish", "restaurant": restaurant_row_to_dict(restaurant), "recommended_dish": dish}, True
        return {"type": "menu_not_found", "restaurant": restaurant_row_to_dict(restaurant)}, False

    # Recommendations / general search
    if intent in ["recommendation", "general_restaurant_question", "unclear"]:
        params = apply_user_profile(params)

        # If user just chats naturally without enough information, let GPT ask a follow-up
        if intent == "recommendation" and not has_any_params(params):
            _context["pending_action"] = {"intent": "recommendation"}
            return {
                "type": "missing_recommendation_params",
                "fallback_answer": "בשמחה 😊 באיזו עיר, אזור או סגנון אוכל תרצי שאחפש?"
            }, True

        recs = recommend_restaurants(params, limit=6)
        used_fallback = False
        if recs.empty:
            recs = fallback_recommendations(params, limit=6)
            used_fallback = True

        if recs.empty:
            return {"type": "internal_recommendations_not_found", "requested_params": params}, False

        _context["pending_action"] = None
        remember_restaurant(recs.iloc[0])
        return {
            "type": "recommendations",
            "requested_params": params,
            "used_fallback_relaxation": used_fallback,
            "restaurants": records_from_df(recs, limit=6),
        }, True

    return {
        "type": "unclear",
        "fallback_answer": "אני כאן לעזור עם מסעדות 😊 אפשר לשאול על המלצה, ביקורות, שעות פתיחה, דירוג או מסעדות דומות."
    }, True


def get_restaurant_by_id(rid: int) -> Optional[pd.Series]:
    df = load_restaurants()
    found = df[df["restaurant_id"].astype(int) == int(rid)]
    return found.iloc[0] if not found.empty else None


def answer_free_text(message: str) -> Dict[str, str]:
    message = (message or "").strip()
    if not message:
        return {"message": "אני כאן 😊 כתבי לי מה תרצי לדעת על מסעדות."}

    parsed = parse_user_message_with_ai(message)
    _context["last_intent"] = parsed.get("intent")

    internal_payload, internal_found = build_internal_response(message, parsed)

    if internal_found:
        answer = ai_final_answer(message, parsed, {
            "source_priority": "internal_database_first",
            "internal_data": internal_payload,
            "external_data": None,
            "fallback_answer": internal_payload.get("fallback_answer"),
        })
        remember_conversation(message, answer)
        return {"message": answer}

    # Not enough in internal DB -> external Google Places -> GPT general restaurant-domain response
    params = merge_params(
        {k: parsed.get(k) for k in extract_params_local(message).keys()},
        extract_params_local(message)
    )

    restaurant = None
    if internal_payload.get("base_restaurant"):
        restaurant = find_restaurant_by_name(internal_payload["base_restaurant"].get("clean_name") or internal_payload["base_restaurant"].get("name"))
    if restaurant is None:
        restaurant = find_restaurant_by_name(parsed.get("restaurant_name") or message) or get_last_restaurant()

    external_payload = external_payload_for_query(message, parsed, params, restaurant)

    answer = ai_final_answer(message, parsed, {
        "source_priority": "internal_database_then_google_places_then_gpt_restaurant_knowledge",
        "internal_data": internal_payload,
        "external_data": external_payload,
        "fallback_answer": (
            "לא מצאתי מידע מספיק בדאטה הפנימי. בדקתי גם מקור חיצוני, ואם אין שם מידע מספיק — אנסה לעזור לפי ידע כללי בתחום המסעדות או אבקש הבהרה."
        ),
    })
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
    app.run(debug=True, port=5001)
