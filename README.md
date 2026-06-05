# SmartBite - AI Restaurant Agent

SmartBite is a Flask web app for restaurant recommendations in Israel.

## Run locally
```bash
pip install -r requirements.txt
python3 app.py
```

Open: http://127.0.0.1:5001

## Data update
The restaurants table was updated with 12,000 restaurant records including Hebrew display/search columns and includes:
- name_he, city_he, region_he, cuisine_he, restaurant_type_he, suitable_for_he, recommended_dish_he
- region: North / Center / South / All Regions
- restaurant_type: Meat / Dairy / Fish / Mixed
- kosher: Yes / No
- price_level: Cheap / Medium / Expensive
- rating, suitable_for, recommended_dish
