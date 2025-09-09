import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Project300 • Automotive Clean & Predict", layout="wide")

st.title("🚗 Project300 — Automotive Clean & Predict (Streamlit)")
st.caption("Fast > Perfect. Upload → preview → auto-flag → (optional) auto-fix → simple FastSale model.")

# ---------- Sidebar: Data source ----------
st.sidebar.header("Data Source")
use_sample = st.sidebar.checkbox("Use sample dataset (auto_sales_sample.csv)", value=True)
uploaded = st.sidebar.file_uploader("Or upload your automotive CSV", type=["csv"])

def load_data():
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state["source_name"] = uploaded.name
        return df
    elif use_sample:
        try:
            df = pd.read_csv("auto_sales_sample.csv")
            st.session_state["source_name"] = "auto_sales_sample.csv"
            return df
        except Exception as e:
            st.error(f"Couldn't load auto_sales_sample.csv: {e}")
            return None
    else:
        return None

df = load_data()
if df is None:
    st.info("Please upload a CSV or toggle 'Use sample dataset'")
    st.stop()

st.subheader("1) Preview")
st.write(f"**Source:** {st.session_state.get('source_name', 'N/A')} • **Shape:** {df.shape[0]} rows × {df.shape[1]} cols")
st.dataframe(df.head(20), use_container_width=True)

# ---------- 2) Validate (flag issues) ----------
st.subheader("2) Validate")
expected_cols = set(["VehicleID","Make","Model","Year","Mileage","ListPrice","DiscountPct","SalePrice","DaysOnLot",
                     "FuelType","Transmission","DealerRegion","CustomerSegment","ServiceVisits","WarrantyClaim","FastSale"])
missing_cols = [c for c in expected_cols if c not in df.columns]

if missing_cols:
    st.warning(f"Missing columns detected (ok for demo): {missing_cols}")

df_flag = df.copy()

# Rules
issues = []

# a) Duplicates by VehicleID
if "VehicleID" in df_flag.columns:
    dup_mask = df_flag.duplicated(subset=["VehicleID"], keep="first")
    df_flag["__dup"] = dup_mask
    issues.append(("Duplicates (VehicleID)", int(dup_mask.sum())))

# b) Year in range
if "Year" in df_flag.columns:
    yr_bad = ~df_flag["Year"].between(1990, 2026, inclusive="both")
    df_flag["__bad_year"] = yr_bad
    issues.append(("Invalid Year", int(yr_bad.sum())))

# c) Mileage non-negative & reasonable
if "Mileage" in df_flag.columns:
    mil_bad = (df_flag["Mileage"] < 0) | (df_flag["Mileage"] > 500000)
    df_flag["__bad_mileage"] = mil_bad
    issues.append(("Invalid Mileage", int(mil_bad.sum())))

# d) ListPrice non-negative
if "ListPrice" in df_flag.columns:
    lp_bad = df_flag["ListPrice"] <= 0
    df_flag["__bad_listprice"] = lp_bad
    issues.append(("Invalid ListPrice", int(lp_bad.sum())))

# e) Consistency: SalePrice ~ ListPrice * (1 - DiscountPct) within tolerance
if {"ListPrice","DiscountPct","SalePrice"}.issubset(df_flag.columns):
    expected_sale = df_flag["ListPrice"] * (1 - df_flag["DiscountPct"].clip(0, 0.9))
    sale_bad = (df_flag["SalePrice"] < expected_sale * 0.9) | (df_flag["SalePrice"] > expected_sale * 1.1)
    df_flag["__bad_saleprice"] = sale_bad
    issues.append(("Inconsistent SalePrice vs Discount", int(sale_bad.sum())))

# f) Null checks
null_counts = df_flag.isnull().sum()
null_total = int(null_counts.sum())
issues.append(("Missing values (any column)", null_total))

# Summary table
sum_df = pd.DataFrame(issues, columns=["Issue", "Count"])
st.write("**Validation summary**")
st.dataframe(sum_df, use_container_width=True)

# Show flagged rows filter
flag_cols = [c for c in df_flag.columns if c.startswith("__") and df_flag[c].dtype == bool]
if flag_cols:
    flagged = df_flag.loc[df_flag[flag_cols].any(axis=1)]
    with st.expander("Show flagged rows"):
        st.dataframe(flagged.drop(columns=flag_cols), use_container_width=True)
else:
    st.success("No issues flagged by current rules.")

# ---------- 3) Auto-fixes (optional) ----------
st.subheader("3) Auto-fix (optional)")
do_fix = st.checkbox("Apply auto-fixes (drop duplicates, fix invalids, fill nulls, recompute SalePrice)")

def auto_fix(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    # drop duplicate VehicleID
    if "VehicleID" in df.columns:
        df = df.drop_duplicates(subset=["VehicleID"], keep="first")
    # replace invalid Year
    if "Year" in df.columns:
        df.loc[~df["Year"].between(1990, 2026, inclusive="both"), "Year"] = df["Year"].median()
    # replace bad mileage/listprice
    if "Mileage" in df.columns:
        df.loc[(df["Mileage"] < 0) | (df["Mileage"] > 500000), "Mileage"] = df["Mileage"].median()
    if "ListPrice" in df.columns:
        df.loc[df["ListPrice"] <= 0, "ListPrice"] = df["ListPrice"].median()
    # fill DiscountPct
    if "DiscountPct" in df.columns:
        df["DiscountPct"] = df["DiscountPct"].fillna(df["DiscountPct"].median()).clip(0, 0.9)
    # recompute SalePrice = ListPrice * (1 - DiscountPct)
    if {"ListPrice","DiscountPct"}.issubset(df.columns):
        df["SalePrice"] = (df["ListPrice"] * (1 - df["DiscountPct"])).round(0).astype(int)
    # fill remaining nulls
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        mode = df[c].mode(dropna=True)
        df[c] = df[c].fillna(mode.iloc[0] if len(mode) else "Unknown")
    return df

if do_fix:
    df_fixed = auto_fix(df)
    st.success("Auto-fixes applied.")
else:
    df_fixed = df.copy()

with st.expander("Preview cleaned data"):
    st.dataframe(df_fixed.head(20), use_container_width=True)

# Download cleaned CSV
def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download cleaned CSV",
    data=to_csv_bytes(df_fixed),
    file_name="cleaned_auto_sales.csv",
    mime="text/csv",
    use_container_width=True
)

# ---------- 4) Simple FastSale model ----------
st.subheader("4) Predict FastSale (<30 days)")
st.caption("Quick baseline model trained on the fly (RandomForest).")

# Only run if target exists
if "FastSale" in df_fixed.columns:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Features (numeric) + simple one-hot of a few categoricals
    use_num = [c for c in ["Year","Mileage","ListPrice","SalePrice","DiscountPct","DaysOnLot","ServiceVisits","WarrantyClaim"] if c in df_fixed.columns]
    X_num = df_fixed[use_num].copy()

    cat_cols_model = [c for c in ["Make","FuelType","Transmission","DealerRegion","CustomerSegment"] if c in df_fixed.columns]
    X_cat = pd.get_dummies(df_fixed[cat_cols_model], drop_first=True) if cat_cols_model else None
    X = pd.concat([X_num, X_cat], axis=1) if X_cat is not None else X_num

    y = df_fixed["FastSale"].astype(int)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, preds)

    st.write(f"**Accuracy:** {acc:.3f}")
    # Top importances
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    st.write("**Top feature importances:**")
    st.dataframe(importances.to_frame("importance"))

    # Scoring all rows (optional)
    df_scored = df_fixed.copy()
    df_scored["FastSale_Prob"] = model.predict_proba(X)[:,1]
    with st.expander("Scored dataset preview"):
        st.dataframe(df_scored.head(20), use_container_width=True)

    # Download scored CSV
    st.download_button(
        "⬇️ Download scored CSV (with FastSale_Prob)",
        data=to_csv_bytes(df_scored),
        file_name="scored_auto_sales.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.info("Column 'FastSale' not found. Add it to enable training, or keep the cleansing-only flow.")