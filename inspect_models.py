import pickle
import pandas as pd

print("=" * 60)
print("📊 LOADING TRAINED MODELS")
print("=" * 60)

# Load Decision Tree Model
print("\n🌳 DECISION TREE MODEL (model_dt.pkl)")
print("-" * 60)
dt = pickle.load(open("model_dt.pkl", "rb"))
print(f"Model Type: {type(dt).__name__}")
print(f"Tree Depth: {dt.get_depth()}")
print(f"Number of Leaves: {dt.get_n_leaves()}")
print(f"Number of Features: {dt.n_features_in_}")
print(f"Feature Names: {dt.feature_names_in_ if hasattr(dt, 'feature_names_in_') else 'Not stored'}")
print(f"Classes: {dt.classes_}")
print(f"Feature Importances:\n{pd.Series(dt.feature_importances_, index=dt.feature_names_in_ if hasattr(dt, 'feature_names_in_') else range(dt.n_features_in_)).sort_values(ascending=False)}")

# Load KNN Model
print("\n\n👥 K-NEAREST NEIGHBORS MODEL (model_knn.pkl)")
print("-" * 60)
knn_pipeline = pickle.load(open("model_knn.pkl", "rb"))
print(f"Model Type: {type(knn_pipeline).__name__}")
print(f"Pipeline Steps: {[name for name, _ in knn_pipeline.steps]}")

# Access the KNN classifier within the pipeline
knn = knn_pipeline.named_steps["knn"]
preprocessor = knn_pipeline.named_steps["preprocessor"]

print(f"K (neighbors): {knn.n_neighbors}")
print(f"Distance Metric: {knn.metric}")
print(f"Number of Training Samples Stored: {knn.n_samples_fit_}")
print(f"Number of Features (after preprocessing): {knn.n_features_in_}")
print(f"Feature Names: {knn.feature_names_in_ if hasattr(knn, 'feature_names_in_') else 'Not stored'}")
print(f"Classes: {knn.classes_}")

# Show preprocessing details
print("\n📋 PREPROCESSING DETAILS:")
print(f"  Numerical features (StandardScaler): {preprocessor.transformers_[0][2]}")
print(f"  Categorical features (OneHotEncoder): {preprocessor.transformers_[1][2]}")
try:
    cat_features = preprocessor.transformers_[1][1].named_steps["onehot"]
    print(f"  OneHotEncoder categories: {cat_features.categories_}")
except Exception:
    pass

print("\n" + "=" * 60)
print("✅ Models loaded successfully!")
print("=" * 60)
