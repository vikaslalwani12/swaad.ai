# Shared constants for the meal planner system

NON_VEG_KEYWORDS = [
    "chicken", "fish", "egg", "mutton", "lamb", "prawn", "shrimp", "meat", "beef", "pork", "trotter"
]

INDIAN_CUISINES = [
    "indian", "south indian", "north indian", "punjabi", "gujarati", "maharashtrian", "bengali", "rajasthani", "tamil nadu", "andhra", "udupi", "chettinad", "kerala", "hyderabadi", "goan", "mughlai", "sindhi"
]

NON_INDIAN_CUISINES = [
    "mexican", "italian", "continental", "korean", "japanese", "greek", "thai", "chinese", "french", "american", "mediterranean", "spanish", "vietnamese", "lebanese", "turkish", "german", "russian"
]

COMMON_ESSENTIALS = set([
    'salt', 'water', 'oil', 'turmeric', 'cumin', 'garlic', 'ginger', 'onion', 'tomato', 'sugar', 'black pepper',
    'mustard seeds', 'asafoetida', 'hing', 'jeera', 'lemon', 'milk', 'curd', 'bread', 'wheat flour', 'atta', 'rice',
    'paneer', 'ghee', 'kasuri methi', 'curry leaves', 'bay leaf', 'cloves', 'cardamom', 'cinnamon', 'fenugreek',
    'star anise', 'nutmeg', 'mace', 'chili', 'vinegar', 'baking powder', 'baking soda', 'cornflour', 'corn starch',
    'butter', 'cream', 'yogurt', 'soya chunks', 'beetroot', 'coriander', 'coriander leaves', 'coriander powder',
    'green peas', 'peas', 'mustard oil', 'refined oil', 'sunflower oil', 'red chili', 'red chillies',
    'green chillies', 'green chilies', 'red chilli','red chilli powder', 'green chilli', 'chakundar', 'chaat masala'
])

ROTI = {"calories": 100, "protein": 3, "carbs": 20, "fat": 0.5}
RICE = {"calories": 200, "protein": 4, "carbs": 45, "fat": 0.5}
CARB_ITEMS = ["roti", "rice", "paratha", "poha", "upma", "idli", "dosa", "thepla", "chapati", "bread"]
PROTEIN_ITEMS = ["dal", "paneer", "chana", "rajma", "sprouts", "curd", "yogurt", "tofu", "soy", "moong", "chickpea", "lentil", "sambar", "kadhi"]
SABZI_KEYWORDS = [
    "sabzi", "curry", "masala", "bhaji", "kootu", "poriyal", "thoran", "usili", "gobi", "bhindi", "aloo", "baingan", "cauliflower", "peas", "matar"
]
SNACK_KEYWORDS = ["snack", "cutlet", "pakora", "samosa", "chop", "vada", "tikki", "bhajiya", "fritter"]
CONDIMENT_KEYWORDS = ["raita", "chutney", "pickle", "papad", "salad", "buttermilk", "lassi", "tea", "coffee", "juice"]
SNACK_EXCLUDE_KEYWORDS = [
    "masala", "powder", "chutney", "pickle", "paste", "seasoning", "condiment", "spice mix", "gravy", "sauce"
]
