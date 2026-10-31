
import joblib

model = joblib.load("best_warranty_fraud_model.joblib")

print("MODEL TYPE:")
print(type(model))

if isinstance(model, dict):
    print("\nDICTIONARY KEYS:")
    for key, value in model.items():
        print(key, ":", type(value))

        if hasattr(value, "feature_names_in_"):
            print("Input features:", value.feature_names_in_)

else:
    print("\nMODEL DETAILS:")
    print(model)

    if hasattr(model, "feature_names_in_"):
        print("\nInput features:")
        print(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        print("\nPipeline steps:")
        print(model.named_steps)
