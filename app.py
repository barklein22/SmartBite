from __future__ import annotations

import os
import json
import random
import re
from typing import Any, Dict, Optional

import pandas as pd
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESTAURANTS_CSV = os.path.join(
    BASE_DIR, "data", "smartbite_12000_restaurants_hebrew.csv"
)

ORDERS_CSV = os.path.join(
    BASE_DIR, "data", "orders.csv"
)

REVIEWS_CSV = os.path.join(
    BASE_DIR, "data", "reviews.csv"
)

CUSTOMERS_CSV = os.path.join(
    BASE_DIR, "data", "customers.csv"
)

app = Flask(__name__)

_cache: Dict[str, Any] = {}
_context: Dict[str, Any] = {
    "last_restaurant_id": None,
    "last_params": {},
    "user_age": None,
    "user_age_group": None,
    "user_city": None,
    "user_region": None,
    "dietary_preference": None,
    "pending_peak_question": False,
    "awaiting_city": False,
    "awaiting_peak_city": False,
    "pending_peak_restaurant": None,
    "awaiting_reviews_city": False,
    "pending_reviews_restaurant": None,
    "awaiting_available_branch_city": False,
    "pending_available_branch_restaurant": None,
    "pending_available_branch_purpose": None,
    "last_info_restaurant": None,
    "last_info_purpose": None,
    "conversation_history": [],
    "pending_ai_action": None,
}


def normalize_text(value: str) -> str:
    value = str(value).lower().strip()
    value = value.replace("׳", "'").replace("’", "'").replace("`", "'")
    value = re.sub(r"[\"'״׳`.,:;!?()\[\]{}\-_/\\]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


RESTAURANT_ALIASES = {
    "מקדונלדס": [
        "מקדונלדס", "מקדונלד ס", "מקדונלד'ס", "מקדונלד׳ס",
        "מק דונלדס", "מק דונלד'ס", "mcdonalds", "mcdonald s",
        "mc donalds", "mc donald's", "mcdonald's israel"
    ],
    "ארומה": ["ארומה", "aroma"],
    "קפה קפה": ["קפה קפה", "cafe cafe"],
    "גרג": ["גרג", "greg"],
    "לנדוור": ["לנדוור", "landwer"],
    "גפניקה": ["גפניקה", "ג'פניקה", "ג׳פניקה", "japanika"],
    "מחניודה": ["מחניודה", "machneyuda"],
    "אבו חסן": ["אבו חסן", "abu hassan"],
    "טאיזו": ["טאיזו", "taizu"],
    "שילה": ["שילה", "shila"],
    "קלארו": ["קלארו", "claro"],
    "אורי בורי": ["אורי בורי", "uri buri"],
}

CITY_MAP = {
    "בתל אביב": "Tel Aviv",
    "תל אביב": "Tel Aviv",
    "תל-אביב": "Tel Aviv",
    "בתל-אביב": "Tel Aviv",
    "בת״א": "Tel Aviv",
    "ת״א": "Tel Aviv",
    "בת\"א": "Tel Aviv",
    "ת\"א": "Tel Aviv",
    "בתא": "Tel Aviv",
    "תא": "Tel Aviv",

    "בירושלים": "Jerusalem",
    "ירושלים": "Jerusalem",

    "בחיפה": "Haifa",
    "חיפה": "Haifa",

    "ברמת גן": "Ramat Gan",
    "רמת גן": "Ramat Gan",

    "בגבעתיים": "Givatayim",
    "גבעתיים": "Givatayim",

    "בחולון": "Holon",
    "חולון": "Holon",

    "בראשון לציון": "Rishon LeZion",
    "ראשון לציון": "Rishon LeZion",

    "בפתח תקווה": "Petah Tikva",
    "פתח תקווה": "Petah Tikva",

    "בהרצליה": "Herzliya",
    "הרצליה": "Herzliya",

    "בנתניה": "Netanya",
    "נתניה": "Netanya",

    "בכפר סבא": "Kfar Saba",
    "כפר סבא": "Kfar Saba",

    "ברעננה": "Ra'anana",
    "רעננה": "Ra'anana",

    "ברחובות": "Rehovot",
    "רחובות": "Rehovot",

    "במודיעין": "Modiin",
    "מודיעין": "Modiin",

    "באשדוד": "Ashdod",
    "אשדוד": "Ashdod",

    "באשקלון": "Ashkelon",
    "אשקלון": "Ashkelon",

    "בבאר שבע": "Beer Sheva",
    "באר שבע": "Beer Sheva",

    "באילת": "Eilat",
    "אילת": "Eilat",

    "בנצרת": "Nazareth",
    "נצרת": "Nazareth",

    "בעכו": "Acre",
    "עכו": "Acre",

    "בקיסריה": "Caesarea",
    "קיסריה": "Caesarea",

    "בזכרון יעקב": "Zichron Yaakov",
    "זכרון יעקב": "Zichron Yaakov",

    "בטבריה": "Tiberias",
    "טבריה": "Tiberias",

    "בראש פינה": "Rosh Pina",
    "ראש פינה": "Rosh Pina",
}

REGION_MAP = {
    "צפון": "North",
    "בצפון": "North",
    "מרכז": "Center",
    "במרכז": "Center",
    "דרום": "South",
    "בדרום": "South",
    "ירושלים": "Jerusalem",
}

CUISINE_MAP = {
    "איטלקי": "Italian",
    "איטלקית": "Italian",
    "פסטה": "Italian",
    "פיצה": "Pizza",
    "יפני": "Japanese",
    "יפנית": "Japanese",
    "סושי": "Japanese",
    "אסייתי": "Asian",
    "אסייתית": "Asian",
    "נודלס": "Asian",
    "המבורגר": "Burgers",
    "בורגר": "Burgers",
    "טבעוני": "Vegan",
    "טבעונית": "Vegan",
    "ישראלי": "Israeli",
    "ישראלית": "Israeli",
    "ים תיכוני": "Mediterranean",
    "ערבי": "Arab",
    "ערבית": "Arab",
    "יווני": "Greek",
    "יוונית": "Greek",
    "תאילנדי": "Thai",
    "דגים": "Seafood",
    "פירות ים": "Seafood",
    "סטייק": "Steakhouse",
    "קפה": "Cafe",
    "בית קפה": "Cafe",
    "ארוחת בוקר": "Breakfast",
}

DISH_KEYWORDS = {
    "פסטה": ["Pasta", "Fresh Pasta", "Pasta Alfredo", "Seafood Pasta", "פסטה"],
    "סושי": ["Sushi", "Sushi Roll", "Sushi Combo", "סושי"],
    "המבורגר": ["Burger", "Classic Burger", "Agadir Burger", "Moses Burger", "Black Burger"],
    "בורגר": ["Burger", "Classic Burger", "Agadir Burger", "Moses Burger", "Black Burger"],
    "פיצה": ["Pizza", "Margherita Pizza", "Pan Pizza", "Pepperoni Pizza"],
    "סטייק": ["Steak", "Entrecote"],
    "דגים": ["Fish", "Sea Bass", "Grilled Fish", "Fish Dish"],
    "חומוס": ["Hummus"],
    "שקשוקה": ["Shakshuka"],
    "קבב": ["Kebab"],
}

RESTAURANT_TYPE_KEYWORDS = {
    "chef": ["מסעדת שף", "מסעדות שף", "שף", "chef"],
    "meat": ["בשרי", "בשרית", "בשרים", "גריל"],
    "dairy": ["חלבי", "חלבית"],
    "fish": ["דגים", "דג", "פירות ים"],
    "vegan": ["טבעוני", "טבעונית"],
}

PRICE_TEXT = {
    "Cheap": "זולה",
    "Medium": "במחיר בינוני",
    "Expensive": "יקרה / יוקרתית",
}


def load_restaurants() -> pd.DataFrame:
    if "restaurants" not in _cache:
        df = pd.read_csv(RESTAURANTS_CSV)

        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
        df["avg_price_per_person"] = pd.to_numeric(
            df["avg_price_per_person"], errors="coerce"
        ).fillna(0)

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("").astype(str)

        _cache["restaurants"] = df

    return _cache["restaurants"].copy()


def load_orders() -> pd.DataFrame:
    """
    טוען את טבלת ההזמנות לצורך ניתוח שעות עומס ופופולריות.
    אם הקובץ לא קיים, מחזיר טבלה ריקה כדי לא להפיל את הצ'אט.
    """
    if "orders" not in _cache:
        if not os.path.exists(ORDERS_CSV):
            _cache["orders"] = pd.DataFrame()
            return _cache["orders"].copy()

        df = pd.read_csv(ORDERS_CSV)

        if "order_datetime" in df.columns:
            df["order_datetime"] = pd.to_datetime(df["order_datetime"], errors="coerce")

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("").astype(str)

        _cache["orders"] = df

    return _cache["orders"].copy()


def load_customers() -> pd.DataFrame:
    """
    טוען את טבלת הלקוחות לצורך הסבר/הרחבה של התאמה אישית.
    בפועל, ההתאמה בצ'אט מתבצעת לפי מה שהמשתמש כותב בשיחה:
    גיל, עיר, אזור והעדפה תזונתית.
    """
    if "customers" not in _cache:
        if not os.path.exists(CUSTOMERS_CSV):
            _cache["customers"] = pd.DataFrame()
            return _cache["customers"].copy()

        df = pd.read_csv(CUSTOMERS_CSV)

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("").astype(str)

        _cache["customers"] = df

    return _cache["customers"].copy()


def city_to_region(city: str) -> Optional[str]:
    if not city:
        return None

    city_eng = CITY_MAP.get(city, city)

    center_cities = {
        "Tel Aviv", "Ramat Gan", "Givatayim", "Holon", "Rishon LeZion",
        "Petah Tikva", "Herzliya", "Netanya", "Kfar Saba", "Ra'anana",
        "Rehovot", "Modiin"
    }
    north_cities = {
        "Haifa", "Nazareth", "Acre", "Caesarea", "Zichron Yaakov",
        "Tiberias", "Rosh Pina"
    }
    south_cities = {
        "Ashdod", "Ashkelon", "Beer Sheva", "Eilat"
    }

    if city_eng in center_cities:
        return "Center"
    if city_eng in north_cities:
        return "North"
    if city_eng in south_cities:
        return "South"
    if city_eng == "Jerusalem":
        return "Jerusalem"

    return None


def age_to_group(age: int) -> str:
    if age <= 18:
        return "Teen"
    if age <= 30:
        return "Young Adult"
    if age <= 50:
        return "Adult"
    return "Senior"


def detect_user_profile(message: str) -> Optional[str]:
    """
    מזהה פרטים אישיים פשוטים מתוך השיחה:
    גיל, עיר/אזור והעדפה תזונתית.
    לדוגמה:
    - אני בן 16 טבעוני
    - אני בת 22 מתל אביב
    - אני גרה בחיפה
    """
    msg = message.lower()
    updates = []

    # גיל: בן 16 / בת 22 / גיל 35
    age_match = re.search(r"(?:בן|בת|גיל)\s*(\d{1,2})", msg)
    if age_match:
        age = int(age_match.group(1))
        _context["user_age"] = age
        _context["user_age_group"] = age_to_group(age)
        updates.append(f"גיל {age}")

    # עיר - ממיינים לפי אורך כדי לזהות קודם "בתל אביב" ולא רק "תא"
    for heb_city, eng_city in sorted(CITY_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        if heb_city in message:
            _context["user_city"] = eng_city
            region = city_to_region(eng_city)
            if region:
                _context["user_region"] = region
            updates.append(f"אזור {heb_city}")
            break

    # אזור
    for heb_region, eng_region in REGION_MAP.items():
        if heb_region in message:
            _context["user_region"] = eng_region
            updates.append(f"אזור {heb_region}")
            break

    # העדפות תזונה
    if any(w in msg for w in ["טבעוני", "טבעונית"]):
        _context["dietary_preference"] = "vegan"
        updates.append("העדפה טבעונית")
    elif any(w in msg for w in ["צמחוני", "צמחונית"]):
        _context["dietary_preference"] = "vegetarian"
        updates.append("העדפה צמחונית")
    elif any(w in msg for w in ["ללא גלוטן", "בלי גלוטן", "gluten free"]):
        _context["dietary_preference"] = "gluten_free"
        updates.append("ללא גלוטן")
    elif any(w in msg for w in ["כשר", "כשרה", "כשרות"]):
        _context["dietary_preference"] = "kosher"
        updates.append("כשרות")

    if not updates:
        return None

    return "זכרתי את הפרטים שלך: " + ", ".join(updates) + " 😊"


def apply_user_profile(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    מוסיף התאמה אישית להמלצות אם המשתמש סיפק גיל/אזור/העדפות.
    לא דורס בקשות מפורשות של המשתמש, אלא משלים מה שחסר.
    """
    updated = params.copy()

    # אזור/עיר מהזיכרון
    if not updated.get("city") and _context.get("user_city"):
        updated["city"] = _context["user_city"]
    elif not updated.get("city") and not updated.get("region") and _context.get("user_region"):
        updated["region"] = _context["user_region"]

    # העדפות תזונה
    dietary = _context.get("dietary_preference")
    if dietary == "vegan":
        updated["restaurant_type"] = "Vegan"
        updated["cuisine"] = "Vegan"
    elif dietary == "kosher" and not updated.get("kosher"):
        updated["kosher"] = "Yes"

    # התאמה לפי גיל
    age = _context.get("user_age")
    if age:
        if age <= 18:
            # בני נוער: נעדיף מקומות במחיר נגיש.
            # לא מחייבים suitable_for=Friends כדי לא לצמצם יותר מדי וליפול לעיר אחרת.
            if not updated.get("budget") and not updated.get("price_level"):
                updated["budget"] = 120
        elif 19 <= age <= 30:
            # צעירים: אם אין העדפה אחרת, נשאיר פתוח כדי לא לצמצם מדי
            pass
        elif age >= 50:
            # גיל מבוגר יותר: נעדיף דירוג גבוה אם לא ביקשו משהו אחר
            if not updated.get("rating_min"):
                updated["rating_min"] = 4.3

    return updated


def user_profile_note() -> str:
    # שומרים התאמה אישית מאחורי הקלעים, בלי להציג למשתמש שורת הסבר טכנית.
    return ""


def is_peak_time_question(message: str) -> bool:
    msg = message.lower()
    return any(
        phrase in msg
        for phrase in [
            "שעת עומס",
            "שעת העומס",
            "שעות עומס",
            "שעות העומס",
            "זמן עומס",
            "זמני עומס",
            "עומס",
            "העומס",
            "עמוס",
            "הכי עמוס",
            "מתי עמוס",
            "מתי הכי עמוס",
            "באיזה שעה עמוס",
            "באילו שעות עמוס",
            "באילו שעות הכי עמוס",
            "מתי כדאי להגיע",
            "מתי עדיף להגיע",
            "מתי פחות עמוס",
            "מתי לא עמוס",
        ]
    )


def answer_peak_hours(restaurant: Optional[pd.Series]) -> str:
    if restaurant is None:
        return "לא זיהיתי על איזו מסעדה שאלת. תכתבי למשל: מה שעת העומס של ארומה?"

    orders = load_orders()

    if orders.empty:
        return "אין לי כרגע נתוני הזמנות זמינים כדי לחשב שעות עומס."

    if "restaurant_id" not in orders.columns or "order_datetime" not in orders.columns:
        return "טבלת ההזמנות לא כוללת את העמודות הדרושות לחישוב שעות עומס."

    restaurant_id = int(restaurant["restaurant_id"])
    restaurant_orders = orders[orders["restaurant_id"].astype(str) == str(restaurant_id)].copy()

    if restaurant_orders.empty:
        name = str(restaurant.get("name_he") or restaurant.get("name"))
        return f"מצאתי את {name}, אבל אין לי מספיק הזמנות עבורה כדי לחשב שעות עומס."

    restaurant_orders["order_datetime"] = pd.to_datetime(
        restaurant_orders["order_datetime"],
        errors="coerce"
    )
    restaurant_orders = restaurant_orders.dropna(subset=["order_datetime"])

    if restaurant_orders.empty:
        return "נתוני ההזמנות קיימים, אבל לא הצלחתי לקרוא את זמני ההזמנה."

    restaurant_orders["hour"] = restaurant_orders["order_datetime"].dt.hour
    counts = restaurant_orders["hour"].value_counts().sort_values(ascending=False).head(3)

    name = str(restaurant.get("name_he") or restaurant.get("name"))
    lines = [f"לפי נתוני ההזמנות, השעות העמוסות ביותר ב־**{name}** הן:"]

    for hour, count in counts.items():
        lines.append(f"• {int(hour):02d}:00–{(int(hour) + 1) % 24:02d}:00 — {int(count)} הזמנות")

    quiet_hours = restaurant_orders["hour"].value_counts().sort_values(ascending=True).head(2)
    if not quiet_hours.empty:
        quiet_text = ", ".join(
            f"{int(hour):02d}:00" for hour in quiet_hours.index
        )
        lines.append(f"\nאם את רוצה זמן רגוע יותר, כדאי לבדוק סביב: {quiet_text}.")

    return "\n".join(lines)



def load_reviews() -> pd.DataFrame:
    """
    טוען את טבלת הביקורות כדי לענות על שאלות כמו:
    - מה הביקורות על מקדונלדס?
    - מה הדירוג של מקדונלדס?
    """
    if "reviews" not in _cache:
        if not os.path.exists(REVIEWS_CSV):
            _cache["reviews"] = pd.DataFrame()
            return _cache["reviews"].copy()

        df = pd.read_csv(REVIEWS_CSV)

        if "review_date" in df.columns:
            df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("").astype(str)

        _cache["reviews"] = df

    return _cache["reviews"].copy()


def is_reviews_question(message: str) -> bool:
    msg = message.lower()
    return any(w in msg for w in [
        "ביקורת", "ביקורות", "חוות דעת", "מה אומרים", "מה אנשים אומרים",
        "תגובות", "reviews"
    ])


def is_rating_question(message: str) -> bool:
    msg = message.lower()
    return any(w in msg for w in [
        "מה הדירוג", "דירוג של", "כמה דירוג", "איזה דירוג", "rating"
    ])


def answer_rating_for_restaurant(restaurant: Optional[pd.Series]) -> str:
    if restaurant is None:
        return "לא זיהיתי על איזו מסעדה שאלת. תכתבי למשל: מה הדירוג של מקדונלדס?"

    name = str(restaurant.get("name_he") or restaurant.get("name"))
    restaurant_rating = float(restaurant.get("rating", 0))

    reviews = load_reviews()
    if reviews.empty or "restaurant_id" not in reviews.columns or "rating" not in reviews.columns:
        return f"הדירוג של **{name}** לפי טבלת המסעדות הוא **{restaurant_rating:.1f}** ⭐"

    rid = str(int(restaurant["restaurant_id"]))
    restaurant_reviews = reviews[reviews["restaurant_id"].astype(str) == rid].copy()

    if restaurant_reviews.empty:
        return f"הדירוג של **{name}** לפי טבלת המסעדות הוא **{restaurant_rating:.1f}** ⭐"

    avg_rating = restaurant_reviews["rating"].dropna().mean()
    count_reviews = len(restaurant_reviews)

    if pd.isna(avg_rating):
        return f"הדירוג של **{name}** לפי טבלת המסעדות הוא **{restaurant_rating:.1f}** ⭐"

    return (
        f"הדירוג של **{name}** הוא **{restaurant_rating:.1f}** ⭐\n"
        f"לפי טבלת הביקורות, ממוצע הביקורות הוא **{avg_rating:.1f}** מתוך {count_reviews} ביקורות."
    )


def answer_reviews_for_restaurant(restaurant: Optional[pd.Series]) -> str:
    if restaurant is None:
        return "לא זיהיתי על איזו מסעדה שאלת. תכתבי למשל: מה הביקורות על ארומה?"

    reviews = load_reviews()
    name = str(restaurant.get("name_he") or restaurant.get("name"))

    if reviews.empty:
        return f"מצאתי את **{name}**, אבל אין לי כרגע טבלת ביקורות זמינה."

    if "restaurant_id" not in reviews.columns:
        return "טבלת הביקורות לא כוללת restaurant_id ולכן אי אפשר לקשר ביקורות למסעדה."

    rid = str(int(restaurant["restaurant_id"]))
    restaurant_reviews = reviews[reviews["restaurant_id"].astype(str) == rid].copy()

    if restaurant_reviews.empty:
        return f"מצאתי את **{name}**, אבל אין לי ביקורות עבורה בדאטה."

    answer = [f"מצאתי ביקורות על **{name}** 😊"]

    if "rating" in restaurant_reviews.columns:
        avg_rating = restaurant_reviews["rating"].dropna().mean()
        if not pd.isna(avg_rating):
            answer.append(f"דירוג ממוצע בביקורות: **{avg_rating:.1f}** מתוך {len(restaurant_reviews)} ביקורות.")

    if "sentiment" in restaurant_reviews.columns:
        sentiment_counts = restaurant_reviews["sentiment"].astype(str).str.lower().value_counts()
        if not sentiment_counts.empty:
            top_sentiment = sentiment_counts.index[0]
            sentiment_he = {
                "positive": "חיובי",
                "neutral": "ניטרלי",
                "negative": "שלילי"
            }.get(top_sentiment, top_sentiment)
            answer.append(f"הסנטימנט הבולט בביקורות: **{sentiment_he}**.")

    if "sentiment" in restaurant_reviews.columns:
        counts = restaurant_reviews["sentiment"].astype(str).str.lower().value_counts()
        total = int(counts.sum())

        if total > 0:
            pos = int(counts.get("positive", 0))
            neu = int(counts.get("neutral", 0))
            neg = int(counts.get("negative", 0))

            answer.append("\nסיכום הביקורות:")
            answer.append(f"• חיוביות: {pos} מתוך {total}")
            answer.append(f"• ניטרליות: {neu} מתוך {total}")
            answer.append(f"• שליליות: {neg} מתוך {total}")

    answer.append("\nבמקום להציג טקסטים גולמיים, המערכת מסכמת את הביקורות לפי דירוג וסנטימנט.")

    return "\n".join(answer)



def clean_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"\s*-\s*.*?Branch\s*\d+", "", name, flags=re.I)
    name = re.sub(r"\s*-\s*.*?סניף\s*\d+", "", name)
    return name.strip()


def remember_restaurant(row: pd.Series) -> None:
    _context["last_restaurant_id"] = int(row["restaurant_id"])


def get_last_restaurant() -> Optional[pd.Series]:
    rid = _context.get("last_restaurant_id")
    if not rid:
        return None

    df = load_restaurants()
    found = df[df["restaurant_id"] == int(rid)]
    if found.empty:
        return None

    return found.iloc[0]



def find_restaurant_in_location(base_restaurant: Optional[pd.Series], params: Dict[str, Any]) -> Optional[pd.Series]:
    """
    מקבל מסעדה/רשת שכבר זוהתה, ומנסה למצוא סניף שלה בעיר או באזור שהמשתמש ביקש.
    לדוגמה: ארומה + תל אביב -> סניף ארומה בעיר תל אביב אם קיים בדאטה.
    """
    if base_restaurant is None:
        return None

    df = load_restaurants()

    base_name = clean_name(str(base_restaurant.get("name_he") or base_restaurant.get("name")))
    base_norm = normalize_text(base_name)

    chain_key = re.split(r"\s*-\s*", base_name)[0].strip()
    chain_norm = normalize_text(chain_key)

    candidates = df.copy()

    if chain_norm:
        candidates = candidates[
            candidates["name_he"].astype(str).apply(lambda x: chain_norm in normalize_text(clean_name(x)))
            | candidates["name"].astype(str).apply(lambda x: chain_norm in normalize_text(clean_name(x)))
        ]

    if candidates.empty:
        candidates = df[
            df["name_he"].astype(str).apply(lambda x: base_norm in normalize_text(clean_name(x)))
            | df["name"].astype(str).apply(lambda x: base_norm in normalize_text(clean_name(x)))
        ]

    if params.get("city") and not candidates.empty:
        by_city = candidates[candidates["city"].astype(str).str.lower() == params["city"].lower()]
        if not by_city.empty:
            remember_restaurant(by_city.iloc[0])
            return by_city.iloc[0]

    if params.get("region") and not candidates.empty:
        by_region = candidates[candidates["region"].astype(str).str.lower() == params["region"].lower()]
        if not by_region.empty:
            remember_restaurant(by_region.iloc[0])
            return by_region.iloc[0]

    # אם המשתמש ביקש עיר/אזור ולא נמצא סניף מתאים — לא מחזירים סניף מעיר אחרת.
    # כך הסוכן לא "ממציא" התאמה לא נכונה.
    return None



def available_branches_message(base_restaurant: Optional[pd.Series], params: Dict[str, Any], purpose: str) -> str:
    """
    מחזיר הודעה ידידותית כאשר לא נמצא סניף של הרשת בעיר/אזור שהמשתמש ביקש.
    purpose יכול להיות "שעות עומס" או "ביקורות".
    בנוסף שומר Context כדי שאם המשתמש יבחר עיר מהרשימה, נמשיך את אותה שאלה.
    """
    if base_restaurant is None:
        return "לא מצאתי את המסעדה שביקשת."

    _context["awaiting_available_branch_city"] = True
    _context["pending_available_branch_restaurant"] = base_restaurant
    _context["pending_available_branch_purpose"] = purpose

    df = load_restaurants()

    base_name = clean_name(str(base_restaurant.get("name_he") or base_restaurant.get("name")))
    chain_key = re.split(r"\s*-\s*", base_name)[0].strip()
    chain_norm = normalize_text(chain_key)

    candidates = df[
        df["name_he"].astype(str).apply(lambda x: chain_norm in normalize_text(clean_name(x)))
        | df["name"].astype(str).apply(lambda x: chain_norm in normalize_text(clean_name(x)))
    ].copy()

    requested_place = "האזור שביקשת"
    if params.get("city"):
        city_rows = df[df["city"].astype(str).str.lower() == params["city"].lower()]
        if not city_rows.empty and "city_he" in city_rows.columns:
            requested_place = str(city_rows.iloc[0].get("city_he") or params["city"])
        else:
            requested_place = params["city"]
    elif params.get("region"):
        region_rows = df[df["region"].astype(str).str.lower() == params["region"].lower()]
        if not region_rows.empty and "region_he" in region_rows.columns:
            requested_place = str(region_rows.iloc[0].get("region_he") or params["region"])
        else:
            requested_place = params["region"]

    if candidates.empty:
        return f"לא מצאתי סניפים של **{chain_key}** בדאטה."

    branch_places = []
    for _, row in candidates.head(8).iterrows():
        city = str(row.get("city_he") or row.get("city"))
        region = str(row.get("region_he") or row.get("region"))
        item = f"{city} ({region})" if region else city
        if item not in branch_places:
            branch_places.append(item)

    branches_text = "\n".join(f"• {place}" for place in branch_places)

    return (
        f"לא מצאתי סניף של **{chain_key}** ב{requested_place}, "
        f"לכן אני לא רוצה להציג {purpose} של עיר אחרת.\n\n"
        f"מצאתי סניפים זמינים בדאטה ב:\n"
        f"{branches_text}\n\n"
        "אפשר לבחור אחת מהערים האלו."
    )


def extract_rating(text: str) -> Optional[float]:
    if not any(w in text for w in ["דירוג", "מעל", "לפחות", "ומעלה", "גבוה"]):
        return None

    match = re.search(r"([0-5](?:\.\d+)?)", text)
    if match:
        return float(match.group(1))

    return None


def extract_budget(text: str) -> Optional[int]:
    if not any(w in text for w in ["עד", "תקציב", "שקל", "₪", "מחיר"]):
        return None

    nums = [int(x) for x in re.findall(r"\d+", text)]
    return max(nums) if nums else None


def find_restaurant(text: str) -> Optional[pd.Series]:
    df = load_restaurants()
    msg_norm = normalize_text(text)
    msg_compact = msg_norm.replace(" ", "")

    for canonical, aliases in RESTAURANT_ALIASES.items():
        aliases_norm = [normalize_text(a) for a in aliases] + [normalize_text(canonical)]
        aliases_compact = [a.replace(" ", "") for a in aliases_norm]

        if any(alias in msg_norm or alias_compact in msg_compact for alias, alias_compact in zip(aliases_norm, aliases_compact)):
            for _, row in df.iterrows():
                row_names = [
                    str(row.get("name", "")),
                    str(row.get("name_he", "")),
                    clean_name(str(row.get("name", ""))),
                    clean_name(str(row.get("name_he", ""))),
                ]

                row_norms = [normalize_text(n) for n in row_names]
                row_compacts = [rn.replace(" ", "") for rn in row_norms]

                if any(
                    alias in rn or rn in alias or alias_compact in rc or rc in alias_compact
                    for alias, alias_compact in zip(aliases_norm, aliases_compact)
                    for rn, rc in zip(row_norms, row_compacts)
                    if rn
                ):
                    remember_restaurant(row)
                    return row

    for _, row in df.iterrows():
        names = [
            str(row.get("name", "")),
            str(row.get("name_he", "")),
            clean_name(str(row.get("name", ""))),
            clean_name(str(row.get("name_he", ""))),
        ]

        for n in names:
            n_norm = normalize_text(n)
            n_compact = n_norm.replace(" ", "")
            if len(n_norm) >= 2 and (n_norm in msg_norm or n_compact in msg_compact):
                remember_restaurant(row)
                return row

    return None


def extract_params(text: str) -> Dict[str, Any]:
    text_norm = normalize_text(text)
    clean_input = text.strip()
    # מאפשר לזהות הודעות המשך כמו "ובתל אביב?" או "וברמת גן?"
    if clean_input.startswith("ו") and len(clean_input) > 2:
        clean_input_without_vav = clean_input[1:].strip()
    else:
        clean_input_without_vav = clean_input

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
        "known_only": False,
    }

    for heb, eng in sorted(CITY_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        if heb in text or heb in clean_input_without_vav:
            params["city"] = eng
            break

    for heb, eng in REGION_MAP.items():
        if heb in text or heb in clean_input_without_vav:
            params["region"] = eng

    for heb, eng in CUISINE_MAP.items():
        if heb in text:
            params["cuisine"] = eng

    for dish_key in DISH_KEYWORDS:
        if dish_key in text:
            params["dish"] = dish_key

    # זיהוי חזק של מסעדות שף
    if "שף" in text_norm or "chef" in text_norm:
        params["restaurant_type"] = "Chef Restaurant"
    elif any(w in text for w in RESTAURANT_TYPE_KEYWORDS["meat"]):
        params["restaurant_type"] = "Meat"
    elif any(w in text for w in RESTAURANT_TYPE_KEYWORDS["dairy"]):
        params["restaurant_type"] = "Dairy"
    elif any(w in text for w in RESTAURANT_TYPE_KEYWORDS["fish"]):
        params["restaurant_type"] = "Fish"
    elif any(w in text for w in RESTAURANT_TYPE_KEYWORDS["vegan"]):
        params["restaurant_type"] = "Vegan"
        params["cuisine"] = "Vegan"

    if any(w in text for w in ["לא כשר", "לא כשרה", "בלי כשרות"]):
        params["kosher"] = "No"
    elif any(w in text for w in ["כשר", "כשרה", "כשרות"]):
        params["kosher"] = "Yes"

    if any(w in text for w in ["זול", "זולה", "זולות"]):
        params["price_level"] = "Cheap"
    elif any(w in text for w in ["בינוני", "בינונית", "מחיר סביר"]):
        params["price_level"] = "Medium"
    elif any(w in text for w in ["יקר", "יקרה", "יוקרתי", "יוקרתית"]):
        params["price_level"] = "Expensive"

    params["budget"] = extract_budget(text)
    params["rating_min"] = extract_rating(text)

    if any(w in text for w in ["משפחה", "משפחות", "ילדים"]):
        params["suitable_for"] = "Families"
    elif any(w in text for w in ["דייט", "זוג", "זוגות", "רומנטי"]):
        params["suitable_for"] = "Couples"
    elif any(w in text for w in ["חברים", "סטודנטים"]):
        params["suitable_for"] = "Friends"
    elif any(w in text for w in ["עסקי", "עסקים", "פגישה"]):
        params["suitable_for"] = "Business"

    if any(w in text for w in ["מוכרות", "מוכרת", "מומלצות", "מומלצת", "אמיתיות"]):
        params["known_only"] = True

    return params

def has_search_params(params: Dict[str, Any]) -> bool:
    return any(v not in [None, False, ""] for v in params.values())


def apply_filters(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    result = df.copy()

    if params.get("city"):
        result = result[result["city"].str.lower() == params["city"].lower()]

    if params.get("region"):
        result = result[result["region"].str.lower() == params["region"].lower()]

    if params.get("cuisine"):
        result = result[result["cuisine"].str.lower() == params["cuisine"].lower()]

    if params.get("restaurant_type"):
        rt = params["restaurant_type"].lower()

        if rt == "chef restaurant":
            result = result[
                result["restaurant_type"].str.lower().str.contains("chef", na=False)
                | result["restaurant_type_he"].str.contains("שף", na=False)
            ]
        else:
            result = result[result["restaurant_type"].str.lower() == rt]

    if params.get("kosher"):
        result = result[result["kosher"].str.lower() == params["kosher"].lower()]

    if params.get("price_level"):
        result = result[result["price_level"].str.lower() == params["price_level"].lower()]

    if params.get("budget"):
        result = result[result["avg_price_per_person"] <= float(params["budget"])]

    if params.get("rating_min"):
        result = result[result["rating"] >= float(params["rating_min"])]

    if params.get("dish"):
        words = DISH_KEYWORDS.get(params["dish"], [])
        pattern = "|".join(re.escape(w) for w in words)

        result = result[
            result["recommended_dish"].str.contains(pattern, case=False, na=False)
            | result["recommended_dish_he"].str.contains(pattern, case=False, na=False)
        ]

    if params.get("suitable_for"):
        sf = params["suitable_for"]
        result = result[
            result["suitable_for"].str.contains(sf, case=False, na=False)
            | result["suitable_for_he"].str.contains(sf, case=False, na=False)
        ]

    if params.get("known_only") and "is_known_recommended" in result.columns:
        result = result[
            result["is_known_recommended"].astype(str).str.lower() == "yes"
        ]

    return result

def recommend_restaurants(params: Dict[str, Any], limit: int = 6) -> pd.DataFrame:
    df = load_restaurants()
    filtered = apply_filters(df, params)

    if filtered.empty:
        return filtered

    filtered = filtered.copy()

    if "is_known_recommended" in filtered.columns:
        filtered["known_score"] = (
            filtered["is_known_recommended"]
            .astype(str)
            .str.lower()
            .eq("yes")
            .astype(int)
        )
    else:
        filtered["known_score"] = 0

    filtered["score"] = (
        filtered["rating"] * 2.0
        + filtered["known_score"] * 1.5
        - filtered["avg_price_per_person"] / 1000
    )

    return filtered.sort_values(["score", "rating"], ascending=False).head(limit)


def fallback_recommendations(params: Dict[str, Any], limit: int = 6) -> pd.DataFrame:
    """
    החזרת תוצאות קרובות בצורה חכמה:
    אם המשתמש ביקש עיר מפורשת, לא מורידים אותה מהר מדי.
    קודם מורידים תנאים משניים כמו התאמת גיל, תקציב, suitable_for ורק בסוף מתפשרים על עיר.
    """
    # שלב 1: שומרים עיר/אזור, מורידים תנאים שיכולים להגיע מהתאמת גיל
    relaxed = params.copy()
    for key in ["budget", "price_level", "suitable_for"]:
        relaxed[key] = None

    df = recommend_restaurants(relaxed, limit=limit)
    if not df.empty:
        return df

    # שלב 2: עדיין שומרים עיר/אזור, מורידים גם כשרות אם לא הייתה בקשה מפורשת
    relaxed = params.copy()
    for key in ["budget", "price_level", "suitable_for", "kosher"]:
        relaxed[key] = None

    df = recommend_restaurants(relaxed, limit=limit)
    if not df.empty:
        return df

    # שלב 3: אם יש עיר, ננסה אזור של אותה עיר לפני שעוברים לעיר אחרת
    if params.get("city"):
        region = city_to_region(params["city"])
        if region:
            relaxed = params.copy()
            relaxed["city"] = None
            relaxed["region"] = region
            for key in ["budget", "price_level", "suitable_for", "kosher"]:
                relaxed[key] = None

            df = recommend_restaurants(relaxed, limit=limit)
            if not df.empty:
                return df

    # שלב 4: רק בסוף מורידים עיר/אזור
    relaxed = params.copy()
    for key in ["city", "region", "budget", "price_level", "suitable_for", "kosher"]:
        relaxed[key] = None

    df = recommend_restaurants(relaxed, limit=limit)
    if not df.empty:
        return df

    return pd.DataFrame()


def format_recommendations(
    df: pd.DataFrame,
    params: Dict[str, Any],
    fallback: bool = False,
) -> str:
    if df.empty:
        return "לא מצאתי התאמה מדויקת. אפשר לנסות עיר אחרת, אזור אחר, תקציב אחר או סוג מטבח אחר?"

    _context["last_params"] = params
    remember_restaurant(df.iloc[0])

    title = "מצאתי לך אפשרויות מתאימות 😊"
    if fallback:
        title = "לא מצאתי התאמה מדויקת, אבל מצאתי אפשרויות קרובות 😊"

    answer = f"{title}\n\n"

    for _, r in df.iterrows():
        name = str(r.get("name_he") or r.get("name"))
        city = str(r.get("city_he") or r.get("city"))
        region = str(r.get("region_he") or r.get("region"))
        cuisine = str(r.get("cuisine_he") or r.get("cuisine"))
        rtype = str(r.get("restaurant_type_he") or r.get("restaurant_type"))
        dish = str(r.get("recommended_dish_he") or r.get("recommended_dish"))

        kosher = "כשרה" if str(r.get("kosher", "")).lower() == "yes" else "לא כשרה"
        price_level = PRICE_TEXT.get(
            str(r.get("price_level", "")),
            str(r.get("price_level", "")),
        )

        answer += (
            f"🍽️ **{name}**\n"
            f"עיר: {city}\n"
            f"אזור: {region}\n"
            f"סוג מטבח: {cuisine}\n"
            f"סוג מסעדה: {rtype}\n"
            f"כשרות: {kosher}\n"
            f"דירוג: {float(r['rating']):.1f}\n"
            f"מחיר ממוצע לאדם: {float(r['avg_price_per_person']):.0f} ₪\n"
            f"רמת מחיר: {price_level}\n"
            f"מנה מומלצת: {dish}\n\n"
        )

    return answer.strip()



# ---------------------------------------------------------------------
# Cosine Similarity recommendation model
# ---------------------------------------------------------------------

def is_similarity_question(message: str) -> bool:
    msg = normalize_text(message)
    return any(
        phrase in msg
        for phrase in [
            "דומה",
            "דומות",
            "דומים",
            "בסגנון",
            "כמו",
            "מסעדה כמו",
            "מסעדות כמו",
            "similar",
            "like",
        ]
    )


def cosine_similarity_score(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").fillna(0)
    b = pd.to_numeric(b, errors="coerce").fillna(0)

    numerator = float((a * b).sum())
    denominator = float((a.pow(2).sum() ** 0.5) * (b.pow(2).sum() ** 0.5))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def recommend_similar_restaurants(
    restaurant: Optional[pd.Series],
    limit: int = 5,
) -> pd.DataFrame:
    """
    Recommendation model based on Cosine Similarity.
    The model compares restaurants using categorical and numeric features:
    cuisine, restaurant_type, region, kosher, price_level, suitable_for,
    rating and avg_price_per_person.
    """
    if restaurant is None:
        return pd.DataFrame()

    df = load_restaurants()
    if df.empty or "restaurant_id" not in df.columns:
        return pd.DataFrame()

    feature_cols = [
        "cuisine",
        "restaurant_type",
        "region",
        "kosher",
        "price_level",
        "suitable_for",
    ]
    available_feature_cols = [col for col in feature_cols if col in df.columns]

    categorical_features = df[available_feature_cols].fillna("").astype(str).copy()
    categorical_features = pd.get_dummies(categorical_features, columns=available_feature_cols)

    numeric_features = pd.DataFrame(index=df.index)

    if "rating" in df.columns:
        rating = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
        numeric_features["rating_norm"] = rating / 5.0

    if "avg_price_per_person" in df.columns:
        price = pd.to_numeric(df["avg_price_per_person"], errors="coerce").fillna(0)
        max_price = price.max()
        numeric_features["price_norm"] = price / max_price if max_price and max_price > 0 else 0

    feature_matrix = pd.concat([categorical_features, numeric_features], axis=1)

    target_id = int(restaurant["restaurant_id"])
    target_rows = df[df["restaurant_id"].astype(int) == target_id]
    if target_rows.empty:
        return pd.DataFrame()

    target_index = target_rows.index[0]
    target_vector = feature_matrix.loc[target_index]

    similarities = []
    for idx, row in feature_matrix.iterrows():
        current_id = int(df.loc[idx, "restaurant_id"])
        if current_id == target_id:
            continue
        score = cosine_similarity_score(target_vector, row)
        similarities.append((idx, score))

    similarities = sorted(similarities, key=lambda item: item[1], reverse=True)[:limit]
    if not similarities:
        return pd.DataFrame()

    result_indexes = [idx for idx, _ in similarities]
    result_scores = [score for _, score in similarities]

    result = df.loc[result_indexes].copy()
    result["similarity_score"] = result_scores

    return result


def format_similar_restaurants(
    base_restaurant: Optional[pd.Series],
    df: pd.DataFrame,
) -> str:
    if base_restaurant is None:
        return "לא זיהיתי לאיזו מסעדה לחפש מסעדות דומות. אפשר לכתוב למשל: תמליץ לי על מסעדות דומות לארומה."

    base_name = clean_name(str(base_restaurant.get("name_he") or base_restaurant.get("name")))

    if df.empty:
        return f"מצאתי את {base_name}, אבל לא הצלחתי למצוא מסעדות דומות לפי הנתונים שיש לי."

    lines = [
        f"מצאתי מסעדות דומות ל־{base_name} לפי מודל Cosine Similarity.",
        "ההשוואה מתבססת על סוג מטבח, סוג מסעדה, אזור, כשרות, רמת מחיר, התאמה לקהל יעד, דירוג ומחיר ממוצע.",
        "",
    ]

    for _, r in df.iterrows():
        name = str(r.get("name_he") or r.get("name"))
        city = str(r.get("city_he") or r.get("city"))
        region = str(r.get("region_he") or r.get("region"))
        cuisine = str(r.get("cuisine_he") or r.get("cuisine"))
        rtype = str(r.get("restaurant_type_he") or r.get("restaurant_type"))
        dish = str(r.get("recommended_dish_he") or r.get("recommended_dish"))
        score = float(r.get("similarity_score", 0)) * 100

        lines.append(
            f"🍽️ {name}\n"
            f"עיר: {city}\n"
            f"אזור: {region}\n"
            f"סוג מטבח: {cuisine}\n"
            f"סוג מסעדה: {rtype}\n"
            f"דירוג: {float(r['rating']):.1f}\n"
            f"מחיר ממוצע לאדם: {float(r['avg_price_per_person']):.0f} ₪\n"
            f"מנה מומלצת: {dish}\n"
            f"ציון דמיון: {score:.0f}%"
        )

    return "\n\n".join(lines)

def answer_dish_for_restaurant(restaurant: Optional[pd.Series]) -> str:
    if restaurant is None:
        return "לא זיהיתי על איזו מסעדה שאלת. תכתבי למשל: מה המנה המומלצת של מחניודה?"

    remember_restaurant(restaurant)

    name = str(restaurant.get("name_he") or restaurant.get("name"))
    dish = str(restaurant.get("recommended_dish_he") or restaurant.get("recommended_dish"))

    if not dish or dish.lower() == "nan":
        return f"מצאתי את **{name}**, אבל אין לי מנה מומלצת עבורה בדאטה."

    return f"המנה המומלצת של **{name}** היא **{dish}** 🍽️"


def is_restaurant_related(msg: str) -> bool:
    msg = msg.lower()

    allowed_words = [
        "מסעדה", "מסעדות", "מקום לאכול", "איפה לאכול", "אוכל", "ארוחה",
        "לאכול", "רעב", "רעבה", "בא לי", "משהו טעים", "טעים", "טעימה",
        "המלצה", "המלצות", "מומלץ", "מומלצת", "מומלצים", "מומלצות",
        "ביקורת", "ביקורות", "דירוג", "דירוגים", "כוכבים",
        "מנה", "מנות", "מנה מומלצת", "מה להזמין", "כדאי להזמין",
        "מה כדאי לאכול", "תפריט", "אוכל טוב", "הכי מומלץ",

        "שעת עומס", "שעת העומס", "שעות עומס", "שעות העומס", "עומס", "העומס", "עמוס", "הכי עמוס", "מתי כדאי להגיע",

        "בן", "בת", "גיל", "טבעוני", "טבעונית", "צמחוני", "צמחונית",
        "ללא גלוטן", "בלי גלוטן",

        "איטלקי", "איטלקית", "אסייתי", "אסייתית", "יפני", "יפנית",
        "סיני", "סינית", "תאילנדי", "תאילנדית", "הודי", "הודית",
        "מקסיקני", "מקסיקנית", "יווני", "יוונית", "אמריקאי", "אמריקאית",
        "צרפתי", "צרפתית", "ישראלי", "ישראלית", "ים תיכוני", "מזרח תיכוני",
        "ערבי", "ערבית", "ברזילאי", "ברזילאית",

        "פסטה", "פיצה", "סושי", "המבורגר", "בורגר", "נודלס",
        "שווארמה", "פלאפל", "חומוס", "קבב", "סטייק", "דגים",
        "פירות ים", "שקשוקה", "סלט", "קינוח", "גלידה", "וופל", "קרפ",

        "שף", "מסעדת שף", "מסעדות שף", "בשרי", "בשרית", "בשרים",
        "חלבי", "חלבית", "בית קפה", "קפה", "יוקרתי", "יוקרתית",

        "כשר", "כשרה", "כשרות", "לא כשר", "לא כשרה", "בלי כשרות",

        "מחיר", "תקציב", "זול", "זולה", "זולות", "יקר", "יקרה",
        "עד 50", "עד 80", "עד 100", "עד 120", "עד 150", "עד 200",
        "שקל", "₪", "מחיר סביר", "בינוני", "בינונית",

        "דייט", "רומנטי", "רומנטית", "זוג", "זוגות", "משפחה", "משפחות",
        "ילדים", "חברים", "סטודנטים", "עסקי", "עסקים", "פגישה", "אירוע",

        "צפון", "בצפון", "מרכז", "במרכז", "דרום", "בדרום", "כל הארץ",

        "תל אביב", "תא", "ירושלים", "חיפה", "באר שבע", "אילת",
        "נתניה", "הרצליה", "רעננה", "כפר סבא", "רמת גן", "גבעתיים",
        "פתח תקווה", "ראשון לציון", "רחובות", "אשדוד", "אשקלון",
        "טבריה", "נצרת", "עכו", "קיסריה", "זכרון יעקב", "ראש פינה",

        "מקדונלדס", "מקדונלד'ס", "מקדונלד׳ס", "ארומה", "קפה קפה",
        "גרג", "לנדוור", "ג'פניקה", "גפניקה", "ג׳פניקה", "אגאדיר",
        "מוזס", "בלאק", "טאיזו", "מחניודה", "אבו חסן", "שילה",
        "קלארו", "אורי בורי", "גירף", "ג'ירף", "בנדיקט",
    ]

    blocked_words = [
        "מכונית", "רכב", "אוטו", "טויוטה", "יונדאי", "מאזדה", "קיה",
        "טלפון", "אייפון", "סמסונג", "מחשב", "מחשב נייד", "לפטופ",
        "קוד", "תכתוב קוד", "פייתון", "python", "java", "javascript",
        "html", "css", "sql", "mysql", "flask", "react", "github",
        "מטלה", "מבחן", "שיעורי בית", "אוניברסיטה", "מכללה",
        "כמה זה", "משוואה", "נגזרת", "אינטגרל", "חישוב",
        "ראש הממשלה", "בחירות", "כנסת", "פוליטיקה", "חדשות",
        "כדורגל", "כדורסל", "nba", "מזג אוויר", "גשם", "טמפרטורה",
        "רופא", "תרופה", "אופטלגין", "נורופן", "מחלה", "כאב",
        "סרט", "סדרה", "נטפליקס", "טיסה", "מלון", "חופשה"
    ]

    if any(word in msg for word in blocked_words):
        return False

    return any(word in msg for word in allowed_words)


def legacy_answer_free_text(message: str) -> Dict[str, str]:
    msg = message.strip().lower()

    greeting_responses = {
        "היי": [
            "היי 😊 איך אפשר לעזור?",
            "היי! בא לך שאמצא לך מסעדה טובה?",
            "אהלן 😊 מה בא לך לאכול היום?"
        ],
        "הי": [
            "היי 😊 איך אפשר לעזור?",
            "אהלן! מחפשת המלצה למסעדה?",
            "היי 😊 בא לך למצוא מקום טוב לאכול?"
        ],
        "שלום": [
            "שלום 😊 איך אפשר לעזור?",
            "שלום וברוכה הבאה ל-SmartBite 🍽️",
            "שלום 😊 מחפשת מסעדה, ביקורת או המלצה?"
        ],
        "אהלן": [
            "אהלן 😊 מה בא לך לאכול?",
            "אהלן! בא לך שאמצא לך מסעדה מתאימה?",
            "אהלן 😊 איזה סגנון מסעדה בא לך?"
        ],
        "מה קורה": [
            "הכול טוב 😊 מה איתך? בא לך שאמצא לך מסעדה?",
            "מצוין 😄 מה בא לך לאכול היום?",
            "רעב כבר 😄 רוצה המלצה למסעדה?"
        ],
        "מה נשמע": [
            "מעולה 😊 מה בא לך לאכול היום?",
            "הכול טוב 😄 מחפשת המלצה למסעדה?",
            "מצוין 😊 אפשר לעזור לך למצוא מסעדה, ביקורת או המלצה."
        ],
        "hi": [
            "Hi 😊 How can I help you find a restaurant?",
            "Hey! Looking for a restaurant recommendation?",
            "Hi 😊 What kind of food are you in the mood for?"
        ],
        "hello": [
            "Hello 😊 How can I help?",
            "Hello! Looking for a restaurant recommendation?",
            "Hi 😊 What would you like to eat today?"
        ],
        "hey": [
            "Hey 😊 How can I help?",
            "Hey! Want a restaurant recommendation?",
            "Hey 😊 What kind of restaurant are you looking for?"
        ],
    }

    if msg in greeting_responses:
        return {"message": random.choice(greeting_responses[msg])}

    if any(w in msg for w in ["אני רעב", "אני רעבה", "רעב", "רעבה"]):
        return {
            "message": "מבינה אותך 😄 איזה סגנון בא לך? איטלקי, אסייתי, בשרי, חלבי או משהו קליל?"
        }

    if any(w in msg for w in ["בא לי משהו טעים", "משהו טעים", "לא יודע", "לא יודעת"]):
        return {
            "message": "ברור 😊 תני לי כיוון קטן — עיר, תקציב או סוג אוכל — ואני אמליץ לך על משהו מתאים."
        }

    if any(w in msg for w in ["תודה", "תודה רבה", "thanks"]):
        return {"message": "בשמחה 😊"}

    if not is_restaurant_related(msg):
        return {
            "message": "אני SmartBite 🍽️ ומתמחה במסעדות בישראל. אפשר לשאול אותי על המלצות למסעדות, ביקורות, דירוגים, שעות עומס, מנות מומלצות, כשרות, מחירים, ערים וסוגי מטבח."
        }

    # בקשה חדשה להמלצה למסעדה מאפסת הקשרים קודמים.
    # חשוב: זה קורה רק אחרי שבדקנו שההודעה באמת קשורה למסעדות,
    # כדי ששאלה כמו "המלצה למכוניות" לא תפעיל שאלה על עיר.
    if (
        any(word in msg for word in ["תמליץ", "המלצה", "מסעדה", "מסעדות"])
        and not extract_params(message).get("city")
        and not extract_params(message).get("region")
    ):
        _context["awaiting_peak_city"] = False
        _context["pending_peak_restaurant"] = None
        _context["awaiting_reviews_city"] = False
        _context["pending_reviews_restaurant"] = None
        _context["awaiting_available_branch_city"] = False
        _context["pending_available_branch_restaurant"] = None
        _context["pending_available_branch_purpose"] = None
        _context["last_info_restaurant"] = None
        _context["last_info_purpose"] = None

        _context["awaiting_city"] = True
        return {
            "message": "בשמחה 😊\n\nבאיזה אזור או עיר אתה מחפש?"
        }

    # אם המשתמש שואל שאלה חדשה מפורשת על שעות עומס או ביקורות,
    # מאפסים הקשרים קודמים כדי שלא נישאר תקועים על שיחה קודמת.
    if is_peak_time_question(message) or is_reviews_question(message):
        _context["awaiting_city"] = False
        _context["awaiting_peak_city"] = False
        _context["pending_peak_restaurant"] = None
        _context["awaiting_reviews_city"] = False
        _context["pending_reviews_restaurant"] = None
        _context["awaiting_available_branch_city"] = False
        _context["pending_available_branch_restaurant"] = None
        _context["pending_available_branch_purpose"] = None
        _context["last_info_restaurant"] = None
        _context["last_info_purpose"] = None

    restaurant = find_restaurant(message)

    # אם קודם נשאלה שאלת עומס והבוט חיכה לשם מסעדה,
    # הודעה קצרה כמו "מקדונלדס" תיחשב כתשובה.
    if _context.get("pending_peak_question") and restaurant is not None:
        _context["pending_peak_question"] = False
        return {"message": answer_peak_hours(restaurant)}

    # שאלות על שעות עומס לפי orders.csv
    if is_peak_time_question(message):
        if restaurant is None:
            _context["pending_peak_question"] = True
            return {
                "message": "לא זיהיתי על איזו מסעדה שאלת. תכתבי למשל: מה שעות העומס של ארומה?"
            }

        _context["pending_peak_question"] = False
        _context["awaiting_peak_city"] = True
        _context["pending_peak_restaurant"] = restaurant

        restaurant_name = clean_name(str(restaurant.get("name_he") or restaurant.get("name")))

        return {
            "message": (
                f"מצאתי את {restaurant_name} 😊\n\n"
                "באיזו עיר או אזור תרצי לבדוק את שעות העומס?"
            )
        }

    # שאלות על דירוג וביקורות צריכות לקבל תשובה מהדאטה,
    # ולא להחזיר המלצות כלליות.
    if is_rating_question(message):
        return {"message": answer_rating_for_restaurant(restaurant)}

    if is_reviews_question(message):
        if restaurant is None:
            return {"message": answer_reviews_for_restaurant(restaurant)}

        params_for_reviews = extract_params(message)

        if not params_for_reviews.get("city") and not params_for_reviews.get("region"):
            _context["awaiting_reviews_city"] = True
            _context["pending_reviews_restaurant"] = restaurant

            restaurant_name = clean_name(str(restaurant.get("name_he") or restaurant.get("name")))

            return {
                "message": (
                    f"מצאתי את {restaurant_name} 😊\n\n"
                    "באיזו עיר או אזור תרצי לבדוק את הביקורות?"
                )
            }

        restaurant_for_city = find_restaurant_in_location(restaurant, params_for_reviews)

        if restaurant_for_city is None:
            return {
                "message": available_branches_message(
                    restaurant,
                    params_for_reviews,
                    "ביקורות"
                )
            }

        _context["last_info_restaurant"] = restaurant
        _context["last_info_purpose"] = "ביקורות"

        return {"message": answer_reviews_for_restaurant(restaurant_for_city)}

    profile_msg = detect_user_profile(message)

    # המשתמש שואל המשך עם עיר נוספת, למשל: "ובתל אביב?" / "וברמת גן?"
    # המערכת תקשר זאת לשאלה האחרונה: ביקורות או שעות עומס.
    followup_params = extract_params(message)
    if (
        _context.get("last_info_restaurant") is not None
        and _context.get("last_info_purpose") in ["ביקורות", "שעות עומס"]
        and (followup_params.get("city") or followup_params.get("region"))
        and not any(word in msg for word in ["תמליץ", "המלצה", "מסעדה", "מסעדות"])
        # אם המשתמש שואל שאלה חדשה מפורשת, לא מתייחסים אליה כהמשך לשאלה הקודמת.
        and not is_reviews_question(message)
        and not is_peak_time_question(message)
    ):
        pending_restaurant = _context.get("last_info_restaurant")
        purpose = _context.get("last_info_purpose")

        restaurant_for_city = find_restaurant_in_location(pending_restaurant, followup_params)

        if restaurant_for_city is None:
            # אם אין סניף בעיר שביקשו, מציגים רשימת סניפים זמינים,
            # אבל לא משאירים את last_info פעיל כדי שהשאלה הבאה לא תיתקע על אותה מסעדה.
            _context["last_info_restaurant"] = None
            _context["last_info_purpose"] = None

            return {
                "message": available_branches_message(
                    pending_restaurant,
                    followup_params,
                    str(purpose)
                )
            }

        if purpose == "ביקורות":
            return {"message": answer_reviews_for_restaurant(restaurant_for_city)}

        if purpose == "שעות עומס":
            return {"message": answer_peak_hours(restaurant_for_city)}

    # המשתמש בוחר עיר מתוך רשימת סניפים זמינים שהוצגה לו
    if _context.get("awaiting_available_branch_city"):
        params = extract_params(message)

        if params.get("city") or params.get("region"):
            pending_restaurant = _context.get("pending_available_branch_restaurant")
            purpose = _context.get("pending_available_branch_purpose")

            restaurant_for_city = find_restaurant_in_location(pending_restaurant, params)

            if restaurant_for_city is None:
                return {
                    "message": available_branches_message(
                        pending_restaurant,
                        params,
                        str(purpose or "מידע")
                    )
                }

            _context["awaiting_available_branch_city"] = False
            _context["pending_available_branch_restaurant"] = None
            _context["pending_available_branch_purpose"] = None

            _context["last_info_restaurant"] = pending_restaurant
            _context["last_info_purpose"] = purpose

            if purpose == "שעות עומס":
                return {"message": answer_peak_hours(restaurant_for_city)}

            if purpose == "ביקורות":
                return {"message": answer_reviews_for_restaurant(restaurant_for_city)}

            return {"message": "מצאתי את הסניף 😊 איך אפשר לעזור?"}

    # המשתמש עונה על שאלת עיר עבור שעות עומס
    if _context.get("awaiting_peak_city"):
        params = extract_params(message)

        if params.get("city") or params.get("region"):
            _context["awaiting_peak_city"] = False

            pending_restaurant = _context.get("pending_peak_restaurant")
            restaurant_for_city = find_restaurant_in_location(pending_restaurant, params)

            if restaurant_for_city is None:
                return {
                    "message": available_branches_message(
                        pending_restaurant,
                        params,
                        "שעות עומס"
                    )
                }

            _context["last_info_restaurant"] = pending_restaurant
            _context["last_info_purpose"] = "שעות עומס"

            return {"message": answer_peak_hours(restaurant_for_city)}

    # המשתמש עונה על שאלת עיר עבור ביקורות
    if _context.get("awaiting_reviews_city"):
        params = extract_params(message)

        if params.get("city") or params.get("region"):
            _context["awaiting_reviews_city"] = False

            pending_restaurant = _context.get("pending_reviews_restaurant")
            restaurant_for_city = find_restaurant_in_location(pending_restaurant, params)

            if restaurant_for_city is None:
                return {
                    "message": available_branches_message(
                        pending_restaurant,
                        params,
                        "ביקורות"
                    )
                }

            # שומרים את ההקשר כדי שאם המשתמש יכתוב אחר כך "ובתל אביב?"
            # הצ'אט יבין שעדיין מדובר בביקורות של אותה מסעדה.
            _context["last_info_restaurant"] = pending_restaurant
            _context["last_info_purpose"] = "ביקורות"

            return {"message": answer_reviews_for_restaurant(restaurant_for_city)}

    # המשתמש עונה לשאלה על עיר/אזור
    if _context.get("awaiting_city"):
        params = extract_params(message)

        if params.get("city") or params.get("region"):
            _context["awaiting_city"] = False

            params = apply_user_profile(params)
            recs = recommend_restaurants(params)

            if recs.empty:
                fallback = fallback_recommendations(params)
                return {"message": format_recommendations(fallback, params, fallback=True)}

            return {"message": format_recommendations(recs, params)}


    if (
        any(
            w in msg
            for w in [
                "מנה מומלצת",
                "מה המנה",
                "מה כדאי לאכול",
                "מנה של",
                "מנה הכי מומלצת",
                "המנה הכי מומלצת",
                "מה להזמין",
                "כדאי להזמין",
            ]
        )
        or ("מנה" in msg and "מומלצ" in msg)
    ):
        if restaurant is None:
            restaurant = get_last_restaurant()

        return {"message": answer_dish_for_restaurant(restaurant)}

    params = extract_params(message)

    # שומרים את הפרופיל בזיכרון אך לא מפעילים עדיין התאמה אישית,
    # כדי לא להחזיר המלצה אוטומטית אחרי "אני בן 16 טבעוני".
    original_params = params.copy()

    last_params = _context.get("last_params", {})

    # אם המשתמש רק נתן פרופיל כמו גיל / טבעוני / צמחוני / ללא גלוטן,
    # לא מחזירים מיד המלצה. שואלים שאלת המשך כדי שהשיחה תהיה טבעית יותר.
    # המשתמש רק סיפר על עצמו (גיל / טבעוני / צמחוני וכו')
    # ולא ביקש עדיין המלצה.
    if profile_msg and not any(
        word in msg
        for word in [
            "מסעדה",
            "מסעדות",
            "תמליץ",
            "המלצה",
            "לאכול",
            "אוכל",
            "בית קפה",
            "דייט",
        ]
    ):
        return {
            "message": "מעולה 😊\n\nאיך אוכל לעזור לך?"
        }

    # תיקון מיוחד: כל משפט עם המילה שף מפעיל חיפוש מסעדות שף
    if "שף" in msg or "chef" in msg:
        params["restaurant_type"] = "Chef Restaurant"
        recs = recommend_restaurants(params)

        if recs.empty:
            fallback = fallback_recommendations(params)
            return {"message": format_recommendations(fallback, params, fallback=True)}

        return {"message": format_recommendations(recs, params)}

    if (
        last_params
        and has_search_params(params)
        and any(w in msg for w in ["רק", "עכשיו", "ומה", "תראה", "תציג", "עד"])
    ):
        combined = {
            **last_params,
            **{k: v for k, v in params.items() if v not in [None, False, ""]},
        }
        combined = apply_user_profile(combined)

        recs = recommend_restaurants(combined)

        if recs.empty:
            fallback = fallback_recommendations(combined)
            return {"message": format_recommendations(fallback, combined, fallback=True)}

        return {"message": format_recommendations(recs, combined)}

    if has_search_params(params):
        params = apply_user_profile(params)
        recs = recommend_restaurants(params)

        if recs.empty:
            fallback = fallback_recommendations(params)
            return {"message": format_recommendations(fallback, params, fallback=True)}

        return {"message": format_recommendations(recs, params)}

    if profile_msg:
        return {
            "message": profile_msg + "\nעכשיו אפשר לבקש המלצה, למשל: תמליץ לי על מסעדה."
        }

    return {
        "message": "בשמחה 😊 תכתבי לי מה את מחפשת — למשל עיר, סוג מטבח, תקציב, כשרות, גיל, העדפה תזונתית או שעת עומס במסעדה מסוימת."
    }


# ---------------------------------------------------------------------
# ChatGPT-based SmartBite agent
# ---------------------------------------------------------------------

SMARTBITE_DOMAIN_MESSAGE_HE = (
    "אני SmartBite 🍽️ ומתמחה רק בעולם המסעדות בישראל. "
    "אפשר לשאול אותי על המלצות למסעדות, ביקורות, דירוגים, שעות עומס, "
    "מנות מומלצות, כשרות, מחירים, ערים, אזורים וסוגי מטבח."
)

SMARTBITE_DOMAIN_MESSAGE_EN = (
    "I’m SmartBite 🍽️ and I specialize only in restaurants in Israel. "
    "You can ask me about restaurant recommendations, reviews, ratings, peak hours, "
    "recommended dishes, kosher options, prices, cities, regions, and cuisines."
)



def openai_model_name() -> str:
    # אפשר לשנות ב-Render דרך Environment Variable בשם OPENAI_MODEL
    return os.environ.get("OPENAI_MODEL", os.environ.get("OPENAI_ANSWER_MODEL", "gpt-4o-mini"))


def get_openai_client() -> Optional[OpenAI]:
    """
    מחזיר לקוח OpenAI אם קיים OPENAI_API_KEY.
    אם אין מפתח, המערכת תחזור למנגנון המקומי הישן כדי שהאתר לא ייפול.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def detect_language(message: str) -> str:
    if re.search(r"[\u0590-\u05FF]", message):
        return "he"
    return "en"


def remember_conversation(user_message: str, bot_message: str) -> None:
    history = _context.setdefault("conversation_history", [])
    history.append({"user": user_message, "assistant": bot_message})
    # שומרים רק את ההודעות האחרונות כדי לא להעמיס על המודל.
    _context["conversation_history"] = history[-8:]


def json_from_model_text(text_value: str) -> Dict[str, Any]:
    """
    מנסה להוציא JSON מתשובת המודל גם אם הוא עטף אותה בטקסט.
    """
    try:
        return json.loads(text_value)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text_value, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {}


def parse_user_message_with_ai(message: str) -> Dict[str, Any]:
    """
    משתמש ב-ChatGPT כדי להבין שפה טבעית בעברית/אנגלית ולהחזיר Intent מובנה.
    המודל לא עונה למשתמש כאן, אלא רק מתרגם את השאלה לפרמטרים שהקוד יכול להריץ מול הדאטה.
    """
    client = get_openai_client()
    if client is None:
        return {}

    history = _context.get("conversation_history", [])[-6:]
    profile = {
        "user_age": _context.get("user_age"),
        "user_city": _context.get("user_city"),
        "user_region": _context.get("user_region"),
        "dietary_preference": _context.get("dietary_preference"),
        "last_restaurant_id": _context.get("last_restaurant_id"),
        "pending_ai_action": _context.get("pending_ai_action"),
    }

    system_prompt = """
You are an intent parser for SmartBite, a restaurant AI agent in Israel.
Return ONLY valid JSON. Do not answer the user.

SmartBite can answer only restaurant-related questions:
restaurant recommendations, restaurant reviews, ratings, peak hours from orders data,
recommended dishes, menus, kosher/non-kosher, prices, cities/regions in Israel,
cuisine type, similar restaurants, and general restaurant questions.

If the user replies only with a city or region and context.pending_ai_action exists, infer the intent from pending_ai_action.
If the user asks about cars, phones, coding, schoolwork, medicine, politics, weather,
movies, flights, hotels, or anything unrelated to restaurants, return intent "off_topic".

Supported intents:
- greeting
- thanks
- profile_update
- recommendation
- reviews
- rating
- peak_hours
- recommended_dish
- menu
- similar_restaurants
- general_restaurant_question
- off_topic
- unclear

Return JSON with these fields:
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
  "needs_external_search": true/false,
  "clarifying_question": string|null
}

Important:
- Normalize Hebrew city names to English values used in the database:
  Tel Aviv, Jerusalem, Haifa, Ramat Gan, Givatayim, Holon, Rishon LeZion,
  Petah Tikva, Herzliya, Netanya, Kfar Saba, Ra'anana, Rehovot, Modiin,
  Ashdod, Ashkelon, Beer Sheva, Eilat, Nazareth, Acre, Caesarea,
  Zichron Yaakov, Tiberias, Rosh Pina.
- Normalize cuisines to English values when possible: Italian, Japanese, Asian,
  Burgers, Vegan, Israeli, Mediterranean, Arab, Greek, Thai, Seafood,
  Steakhouse, Cafe, Breakfast, Pizza.
- If user says "בתל אביב", "תא", "ת״א", "Tel Aviv" => city "Tel Aviv".
- If user asks opening hours, address, phone number, current availability, live news,
  or data not present in local CSV, set needs_external_search true.
"""

    user_payload = {
        "message": message,
        "conversation_history": history,
        "known_user_profile": profile,
    }

    try:
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_INTENT_MODEL", "gpt-4.1-mini"),
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json_from_model_text(content)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def params_from_ai(parsed: Dict[str, Any]) -> Dict[str, Any]:
    params = {
        "city": parsed.get("city"),
        "region": parsed.get("region"),
        "cuisine": parsed.get("cuisine"),
        "restaurant_type": parsed.get("restaurant_type"),
        "kosher": parsed.get("kosher"),
        "price_level": parsed.get("price_level"),
        "budget": parsed.get("budget"),
        "rating_min": parsed.get("rating_min"),
        "dish": parsed.get("dish"),
        "suitable_for": parsed.get("suitable_for"),
        "known_only": False,
    }

    # ניקוי ערכים ריקים
    for key, value in list(params.items()):
        if value in ["", "null", "None"]:
            params[key] = None

    return params


def update_profile_from_ai(parsed: Dict[str, Any]) -> None:
    age = parsed.get("age")
    if isinstance(age, int):
        _context["user_age"] = age
        _context["user_age_group"] = age_to_group(age)

    city = parsed.get("city")
    if city:
        _context["user_city"] = city
        region = city_to_region(city)
        if region:
            _context["user_region"] = region

    region = parsed.get("region")
    if region:
        _context["user_region"] = region

    dietary = parsed.get("dietary_preference")
    if dietary:
        _context["dietary_preference"] = dietary


def find_restaurant_from_ai(parsed: Dict[str, Any], message: str) -> Optional[pd.Series]:
    restaurant_name = parsed.get("restaurant_name")
    if restaurant_name:
        found = find_restaurant(str(restaurant_name))
        if found is not None:
            return found

    found = find_restaurant(message)
    if found is not None:
        return found

    return get_last_restaurant()


def ask_openai_to_format_answer(user_message: str, data_answer: str, language: str) -> str:
    """
    מקבל תשובה שחושבה מהדאטה ומנסח אותה בצורה טבעית יותר.
    חשוב: אסור למודל להמציא נתונים מעבר למה שמופיע ב-data_answer.
    """
    client = get_openai_client()
    if client is None:
        return data_answer

    system_prompt = """
You are SmartBite, a warm, natural and helpful restaurant assistant.
Rewrite the provided database answer into a friendly conversational answer, like a real chat assistant.
Do NOT invent facts. Use only the provided database answer.
Keep restaurant names, prices, ratings, cities, kosher status, peak hours, and dishes exactly as provided.
If the answer is in Hebrew, answer in Hebrew. If English, answer in English.
Use a natural tone, short explanations, and helpful transitions.
Avoid sounding robotic or like a fixed template.
Keep the answer clear and not too long.
"""

    try:
        response = client.chat.completions.create(
            model=openai_model_name(),
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_message": user_message,
                            "database_answer": data_answer,
                            "language": language,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        return response.choices[0].message.content or data_answer
    except Exception:
        return data_answer




def ask_openai_natural_reply(
    user_message: str,
    language: str,
    purpose: str,
    extra_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    יוצר תשובה שיחתית טבעית דרך ChatGPT למצבים שבהם אין צורך לשלוף דאטה:
    ברכה, תודה, שאלה לא ברורה, עדכון פרופיל, או בקשה מחוץ לתחום.
    """
    client = get_openai_client()
    if client is None:
        if purpose == "greeting":
            return "היי 😊 אני SmartBite. אפשר לשאול אותי על מסעדות, המלצות, ביקורות, מנות או שעות עומס. מה בא לך למצוא היום?"
        if purpose == "thanks":
            return "בשמחה 😊"
        if purpose == "profile_update":
            return "מעולה 😊 איך אוכל לעזור לך?"
        if purpose == "off_topic":
            return SMARTBITE_DOMAIN_MESSAGE_HE if language == "he" else SMARTBITE_DOMAIN_MESSAGE_EN
        return "ספרי לי מה את מחפשת — למשל עיר, סוג מטבח, תקציב, כשרות, ביקורות או שעות עומס 😊"

    history = _context.get("conversation_history", [])[-6:]
    profile = {
        "user_age": _context.get("user_age"),
        "user_city": _context.get("user_city"),
        "user_region": _context.get("user_region"),
        "dietary_preference": _context.get("dietary_preference"),
        "last_restaurant_id": _context.get("last_restaurant_id"),
        "pending_ai_action": _context.get("pending_ai_action"),
    }

    system_prompt = """
You are SmartBite, a natural, friendly restaurant-only chat assistant for restaurants in Israel.

Rules:
- Speak naturally and conversationally, not like a fixed template.
- Keep answers short and helpful.
- Answer in Hebrew if language is "he"; answer in English if language is "en".
- SmartBite only helps with restaurants, food, menus, reviews, ratings, peak hours, kosher, prices, cities and cuisines.
- If the user asks about something outside restaurants, politely say you specialize in restaurants and invite them to ask a restaurant-related question.
- Do not invent database facts such as real ratings, prices, branches or reviews.
- For greeting, open warmly and ask how you can help with restaurants.
- For unclear restaurant questions, ask one natural clarifying question.
"""

    try:
        response = client.chat.completions.create(
            model=openai_model_name(),
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_message": user_message,
                            "purpose": purpose,
                            "language": language,
                            "conversation_history": history,
                            "known_user_profile": profile,
                            "extra_context": extra_context or {},
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        answer = response.choices[0].message.content
        if answer:
            return answer.strip()
    except Exception:
        pass

    if purpose == "greeting":
        return "היי 😊 אני SmartBite. איך אפשר לעזור לך למצוא מסעדה, מנה או המלצה?"
    if purpose == "thanks":
        return "בשמחה 😊"
    if purpose == "profile_update":
        return "מעולה 😊 איך אוכל לעזור לך?"
    if purpose == "off_topic":
        return SMARTBITE_DOMAIN_MESSAGE_HE if language == "he" else SMARTBITE_DOMAIN_MESSAGE_EN
    return "אפשר לנסח לי קצת יותר מדויק? למשל המלצה למסעדה, ביקורות, דירוג, שעות עומס או מנה מומלצת 😊"



def external_restaurant_answer(user_message: str, language: str) -> str:
    """
    fallback חיצוני כאשר המידע לא נמצא בדאטה.
    משתמש ב-OpenAI עם web_search אם זמין. אם לא זמין, מחזיר הודעה שקופה.
    """
    client = get_openai_client()
    if client is None:
        return (
            "אין לי מספיק מידע במאגר הנתונים המקומי כדי לענות על זה."
            if language == "he"
            else "I do not have enough information in the local database to answer that."
        )

    prompt = f"""
You are SmartBite, a restaurant-only assistant.
Answer ONLY if this is about restaurants/food/dining.
If the question is not restaurant-related, politely refuse.
If you use external information, say that it is based on external sources and may change.
Question: {user_message}
"""

    # ניסיון ראשון: Responses API עם web search.
    try:
        response = client.responses.create(
            model=os.environ.get("OPENAI_WEB_MODEL", "gpt-4.1-mini"),
            tools=[{"type": "web_search_preview"}],
            input=prompt,
        )
        answer = getattr(response, "output_text", None)
        if answer:
            return answer
    except Exception:
        pass

    # fallback ללא חיפוש חי
    try:
        response = client.chat.completions.create(
            model=openai_model_name(),
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are SmartBite, a restaurant-only assistant. "
                        "If the answer requires current/external information that you cannot verify, say so clearly. "
                        "Do not answer unrelated topics."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or (
            "אין לי מספיק מידע כדי לענות על זה." if language == "he" else "I do not have enough information to answer that."
        )
    except Exception:
        return (
            "אין לי מספיק מידע במאגר הנתונים המקומי כדי לענות על זה."
            if language == "he"
            else "I do not have enough information in the local database to answer that."
        )



# ---------------------------------------------------------------------
# Real AI Agent orchestration
# ---------------------------------------------------------------------

def restaurant_row_to_dict(row: pd.Series) -> Dict[str, Any]:
    return {
        "restaurant_id": int(row.get("restaurant_id", 0)),
        "name": str(row.get("name_he") or row.get("name")),
        "city": str(row.get("city_he") or row.get("city")),
        "region": str(row.get("region_he") or row.get("region")),
        "cuisine": str(row.get("cuisine_he") or row.get("cuisine")),
        "restaurant_type": str(row.get("restaurant_type_he") or row.get("restaurant_type")),
        "kosher": "כשר" if str(row.get("kosher", "")).lower() == "yes" else "לא כשר",
        "rating": float(row.get("rating", 0)),
        "avg_price_per_person": float(row.get("avg_price_per_person", 0)),
        "price_level": str(row.get("price_level_he") or row.get("price_level")),
        "recommended_dish": str(row.get("recommended_dish_he") or row.get("recommended_dish")),
        "suitable_for": str(row.get("suitable_for_he") or row.get("suitable_for")),
    }


def dataframe_to_records(df: pd.DataFrame, limit: int = 6) -> list[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    return [restaurant_row_to_dict(row) for _, row in df.head(limit).iterrows()]


def ask_ai_to_answer_from_data(
    user_message: str,
    intent: str,
    data_payload: Dict[str, Any],
    language: str,
) -> str:
    """
    Final AI response layer.
    ChatGPT receives the user's question + structured data retrieved from CSV,
    and writes the final natural answer.
    """
    client = get_openai_client()
    if client is None:
        return str(data_payload.get("fallback_answer") or "אני יכולה לעזור עם מסעדות בישראל 🍽️")

    history = _context.get("conversation_history", [])[-8:]

    system_prompt = """
You are SmartBite, a real AI Agent for restaurants in Israel.

Behavior:
- Talk naturally like a helpful human assistant.
- Answer in Hebrew if language is "he", and in English if language is "en".
- Use ONLY the data in the provided data_payload for factual restaurant details.
- Do not invent restaurants, ratings, reviews, prices, branches, peak hours, dishes, or locations.
- If the data is missing, say that the database does not include enough information and ask a useful follow-up question.
- If the user is off-topic, politely say SmartBite specializes only in restaurants and food.
- Preserve exact names, numbers, ratings, prices and cities from the data.
- Avoid sounding like a template.
- Keep answers friendly, clear and concise.
- When there are several restaurants, present them as readable recommendation cards.
- If the data_payload includes model="Cosine Similarity", mention briefly that similar restaurants were found by comparing features, not as a long technical explanation.
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
                            "intent": intent,
                            "language": language,
                            "conversation_history": history,
                            "known_user_profile": {
                                "age": _context.get("user_age"),
                                "city": _context.get("user_city"),
                                "region": _context.get("user_region"),
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

    return str(data_payload.get("fallback_answer") or "מצאתי מידע, אבל לא הצלחתי לנסח אותו כרגע.")


def build_agent_response(message: str, parsed: Dict[str, Any]) -> Dict[str, str]:
    """
    AI Agent flow:
    1. GPT understands intent.
    2. Python retrieves data from CSV/database.
    3. GPT writes a natural final response based on that data.
    """
    language = parsed.get("language") or detect_language(message)
    intent = parsed.get("intent", "unclear")

    # פרופיל משתמש נשמר בזיכרון
    detect_user_profile(message)

    if parsed.get("is_restaurant_related") is False or intent == "off_topic":
        answer = ask_ai_to_answer_from_data(
            message,
            "off_topic",
            {
                "type": "off_topic",
                "allowed_domain": "restaurants and food in Israel",
                "fallback_answer": SMARTBITE_DOMAIN_MESSAGE_HE if language == "he" else SMARTBITE_DOMAIN_MESSAGE_EN,
            },
            language,
        )
        remember_conversation(message, answer)
        return {"message": answer}

    if intent in ["greeting", "thanks", "profile_update"]:
        answer = ask_ai_to_answer_from_data(
            message,
            intent,
            {
                "type": intent,
                "agent_name": "SmartBite",
                "capabilities": [
                    "restaurant recommendations",
                    "reviews",
                    "ratings",
                    "peak hours",
                    "recommended dishes",
                    "kosher/non-kosher",
                    "prices",
                    "cities and regions",
                    "similar restaurants using Cosine Similarity",
                ],
                "fallback_answer": "היי 😊 אני SmartBite. אפשר לשאול אותי על מסעדות, מנות, ביקורות, שעות עומס או המלצות.",
            },
            language,
        )
        remember_conversation(message, answer)
        return {"message": answer}

    params = parsed_to_params(parsed)
    restaurant = find_restaurant(message)

    # אם המשתמש כתב רק שם עיר אחרי שהסוכן שאל המשך
    if _context.get("pending_ai_action") and (params.get("city") or params.get("region")):
        pending = _context.get("pending_ai_action") or {}
        previous_intent = pending.get("intent")
        pending_restaurant_id = pending.get("restaurant_id")

        if pending_restaurant_id:
            df = load_restaurants()
            found = df[df["restaurant_id"].astype(int) == int(pending_restaurant_id)]
            if not found.empty:
                restaurant = found.iloc[0]

        intent = previous_intent or intent

    # Similar restaurants with Cosine Similarity
    if intent == "similar_restaurants" or is_similarity_question(message):
        if restaurant is None:
            restaurant = get_last_restaurant()

        similar_df = recommend_similar_restaurants(restaurant, limit=5)
        fallback_answer = format_similar_restaurants(restaurant, similar_df)

        data_payload = {
            "type": "similar_restaurants",
            "model": "Cosine Similarity",
            "base_restaurant": restaurant_row_to_dict(restaurant) if restaurant is not None else None,
            "restaurants": dataframe_to_records(similar_df, limit=5),
            "fallback_answer": fallback_answer,
        }

        answer = ask_ai_to_answer_from_data(message, "similar_restaurants", data_payload, language)
        remember_conversation(message, answer)
        return {"message": answer}

    # Reviews
    if intent == "reviews":
        if restaurant is None:
            _context["pending_ai_action"] = {"intent": "reviews", "restaurant_id": None}
            answer = ask_ai_to_answer_from_data(
                message,
                "reviews",
                {
                    "type": "missing_restaurant",
                    "missing": "restaurant_name",
                    "fallback_answer": "על איזו מסעדה תרצי שאבדוק ביקורות?",
                },
                language,
            )
            remember_conversation(message, answer)
            return {"message": answer}

        if not params.get("city") and not params.get("region"):
            _context["pending_ai_action"] = {
                "intent": "reviews",
                "restaurant_id": int(restaurant["restaurant_id"]),
            }
            answer = ask_ai_to_answer_from_data(
                message,
                "reviews",
                {
                    "type": "missing_location",
                    "restaurant": restaurant_row_to_dict(restaurant),
                    "fallback_answer": f"באיזו עיר או אזור תרצי לבדוק ביקורות על {clean_name(str(restaurant.get('name_he') or restaurant.get('name')))}?",
                },
                language,
            )
            remember_conversation(message, answer)
            return {"message": answer}

        restaurant_for_city = find_restaurant_in_location(restaurant, params)
        if restaurant_for_city is None:
            fallback_answer = available_branches_message(restaurant, params, "ביקורות")
            data_payload = {
                "type": "branches_not_found",
                "requested_params": params,
                "base_restaurant": restaurant_row_to_dict(restaurant),
                "fallback_answer": fallback_answer,
            }
            answer = ask_ai_to_answer_from_data(message, "reviews", data_payload, language)
            remember_conversation(message, answer)
            return {"message": answer}

        fallback_answer = answer_reviews_for_restaurant(restaurant_for_city)
        data_payload = {
            "type": "reviews",
            "restaurant": restaurant_row_to_dict(restaurant_for_city),
            "summary": fallback_answer,
            "fallback_answer": fallback_answer,
        }
        answer = ask_ai_to_answer_from_data(message, "reviews", data_payload, language)
        remember_conversation(message, answer)
        return {"message": answer}

    # Rating
    if intent == "rating":
        if restaurant is None:
            answer = ask_ai_to_answer_from_data(
                message,
                "rating",
                {"type": "missing_restaurant", "fallback_answer": "על איזו מסעדה תרצי שאבדוק דירוג?"},
                language,
            )
            remember_conversation(message, answer)
            return {"message": answer}

        fallback_answer = answer_rating_for_restaurant(restaurant)
        answer = ask_ai_to_answer_from_data(
            message,
            "rating",
            {
                "type": "rating",
                "restaurant": restaurant_row_to_dict(restaurant),
                "summary": fallback_answer,
                "fallback_answer": fallback_answer,
            },
            language,
        )
        remember_conversation(message, answer)
        return {"message": answer}

    # Peak hours
    if intent == "peak_hours":
        if restaurant is None:
            _context["pending_ai_action"] = {"intent": "peak_hours", "restaurant_id": None}
            answer = ask_ai_to_answer_from_data(
                message,
                "peak_hours",
                {"type": "missing_restaurant", "fallback_answer": "על איזו מסעדה תרצי לבדוק שעות עומס?"},
                language,
            )
            remember_conversation(message, answer)
            return {"message": answer}

        if not params.get("city") and not params.get("region"):
            _context["pending_ai_action"] = {
                "intent": "peak_hours",
                "restaurant_id": int(restaurant["restaurant_id"]),
            }
            answer = ask_ai_to_answer_from_data(
                message,
                "peak_hours",
                {
                    "type": "missing_location",
                    "restaurant": restaurant_row_to_dict(restaurant),
                    "fallback_answer": f"באיזו עיר או אזור תרצי לבדוק את שעות העומס של {clean_name(str(restaurant.get('name_he') or restaurant.get('name')))}?",
                },
                language,
            )
            remember_conversation(message, answer)
            return {"message": answer}

        restaurant_for_city = find_restaurant_in_location(restaurant, params)
        if restaurant_for_city is None:
            fallback_answer = available_branches_message(restaurant, params, "שעות עומס")
            answer = ask_ai_to_answer_from_data(
                message,
                "peak_hours",
                {
                    "type": "branches_not_found",
                    "requested_params": params,
                    "base_restaurant": restaurant_row_to_dict(restaurant),
                    "fallback_answer": fallback_answer,
                },
                language,
            )
            remember_conversation(message, answer)
            return {"message": answer}

        fallback_answer = answer_peak_hours(restaurant_for_city)
        answer = ask_ai_to_answer_from_data(
            message,
            "peak_hours",
            {
                "type": "peak_hours",
                "restaurant": restaurant_row_to_dict(restaurant_for_city),
                "summary": fallback_answer,
                "fallback_answer": fallback_answer,
            },
            language,
        )
        remember_conversation(message, answer)
        return {"message": answer}

    # Recommended dish
    if intent == "recommended_dish":
        if restaurant is None:
            restaurant = get_last_restaurant()

        fallback_answer = answer_dish_for_restaurant(restaurant)
        answer = ask_ai_to_answer_from_data(
            message,
            "recommended_dish",
            {
                "type": "recommended_dish",
                "restaurant": restaurant_row_to_dict(restaurant) if restaurant is not None else None,
                "summary": fallback_answer,
                "fallback_answer": fallback_answer,
            },
            language,
        )
        remember_conversation(message, answer)
        return {"message": answer}

    # Recommendations
    if intent in ["recommendation", "menu", "general_restaurant_question", "unclear"]:
        params = apply_user_profile(params)

        if intent == "recommendation" and not has_search_params(params):
            _context["pending_ai_action"] = {"intent": "recommendation", "restaurant_id": None}
            answer = ask_ai_to_answer_from_data(
                message,
                "recommendation",
                {
                    "type": "missing_recommendation_params",
                    "missing": "city_or_preferences",
                    "fallback_answer": "בשמחה 😊 באיזו עיר, אזור או סגנון אוכל תרצי שאחפש?",
                },
                language,
            )
            remember_conversation(message, answer)
            return {"message": answer}

        recs = recommend_restaurants(params)
        fallback = False
        if recs.empty:
            recs = fallback_recommendations(params)
            fallback = True

        fallback_answer = format_recommendations(recs, params, fallback=fallback)

        data_payload = {
            "type": "restaurant_recommendations",
            "requested_params": params,
            "used_fallback": fallback,
            "restaurants": dataframe_to_records(recs, limit=6),
            "fallback_answer": fallback_answer,
        }

        answer = ask_ai_to_answer_from_data(message, "recommendation", data_payload, language)
        remember_conversation(message, answer)
        return {"message": answer}

    answer = ask_ai_to_answer_from_data(
        message,
        "unclear",
        {
            "type": "unclear",
            "fallback_answer": "אני רוצה לעזור 😊 אפשר לכתוב למשל: המלצה למסעדה בתל אביב, ביקורות על ארומה, שעות עומס של מחניודה או מסעדות דומות לג'פניקה.",
        },
        language,
    )
    remember_conversation(message, answer)
    return {"message": answer}


def answer_free_text(message: str) -> Dict[str, str]:
    message = (message or "").strip()
    if not message:
        return {"message": "אני כאן 😊 כתבי לי מה את מחפשת בעולם המסעדות."}

    language = detect_language(message)

    # אם אין API KEY, לא מפילים את האתר — חוזרים למנגנון המקומי.
    if get_openai_client() is None:
        return legacy_answer_free_text(message)

    parsed = parse_user_message_with_ai(message)
    if not parsed:
        return legacy_answer_free_text(message)

    return build_agent_response(message, parsed)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    message = data.get("message", "")
    return jsonify(answer_free_text(message))


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5001)),
        debug=True,
    )