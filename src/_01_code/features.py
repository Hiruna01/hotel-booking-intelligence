# src/features.py
def add_features(df):
    """total_nights, total_guests, is_family, weekend_ratio, season"""
    df["total_nights"] = df["stays_in_week_nights"] + df["stays_in_weekend_nights"]
    df["total_guests"] =df["adults"] + df["babies"] + df["children"]
    df["is_family"] = ( (df["children"] > 0 ) | (df["babies"] > 0)).astype(int)
    df["weekend_ratio"] = (df["stays_in_weekend_nights"] / df["total_nights"])

    season_map = {
        "December": "winter", "January": "winter", "February": "winter",
        "March": "spring", "April": "spring", "May": "spring",
        "June": "summer", "July": "summer", "August": "summer",
        "September": "autumn", "October": "autumn", "November": "autumn"
    }

    df["season"] = df["arrival_date_month"].map(season_map)

def build_preprocessor(num_cols, cat_cols, poly_degree=None):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.preprocessing import PolynomialFeatures


    num_steps = [
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler()),
]
    if poly_degree:
        num_steps += [
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ("sc2",  StandardScaler()),
        ]
    num_pipe = Pipeline(num_steps)     # build the Pipeline last

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="infrequent_if_exist",
                            min_frequency=100, sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
        ])

    return preprocess