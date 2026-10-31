
import joblib

data = joblib.load("best_warranty_fraud_model.joblib")

model = data["model"]
preprocessor = data["preprocessor"]

print("Model:", data["model_name"])
print("Target:", data["target_column"])

print("\nPreprocessor:")
print(preprocessor)

print("\nClassifier classes:")
print(model.classes_)

print("\nExpected input columns:")
print(preprocessor.feature_names_in_)

print("\n========== CATEGORICAL OPTIONS ==========")

categorical_pipeline = preprocessor.named_transformers_['categorical']

encoder = categorical_pipeline.named_steps['encoder']

categorical_columns = [
    'Region', 'State', 'Area', 'City',
    'Consumer_profile', 'Product_category',
    'Product_type', 'AC_1001_Issue',
    'AC_1002_Issue', 'AC_1003_Issue',
    'TV_2001_Issue', 'TV_2002_Issue',
    'TV_2003_Issue', 'Service_Centre',
    'Purchased_from', 'Purpose'
]

for column, categories in zip(categorical_columns, encoder.categories_):
    print(f"\n{column}:")
    print(categories.tolist())
