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
knn = pickle.load(open("model_knn.pkl", "rb"))
print(f"Model Type: {type(knn).__name__}")
print(f"K (neighbors): {knn.n_neighbors}")
print(f"Distance Metric: {knn.metric}")
print(f"Number of Training Samples Stored: {knn.n_samples_fit_}")
print(f"Number of Features: {knn.n_features_in_}")
print(f"Feature Names: {knn.feature_names_in_ if hasattr(knn, 'feature_names_in_') else 'Not stored'}")
print(f"Classes: {knn.classes_}")

print("\n" + "=" * 60)
print("✅ Models loaded successfully!")
print("=" * 60)
