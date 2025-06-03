import os
import joblib
import pickle
import pandas as pd

def load_and_predict(model_path, new_data_path):
    # — 0. Sanity‐check the file  
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path!r}")
    if os.path.getsize(model_path) == 0:
        raise IOError(f"Model file is empty: {model_path!r}")

    # — 1. Try joblib first  
    try:
        obj = joblib.load(model_path)
    except Exception as e_joblib:
        # Often KeyError in pickle.dispatch_table means "not really a pickle"
        # Fall back to plain pickle
        try:
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e_pickle:
            raise RuntimeError(
                f"Could not unpickle your model file.\n"
                f"joblib.load error: {e_joblib!r}\n"
                f"pickle.load error: {e_pickle!r}\n"
                "Check that you actually saved it with joblib.dump or pickle.dump."
            )

    # — 2. Unpack your model & feature names  
    if isinstance(obj, dict):
        model = obj.get("model") or obj.get("clf") or obj.get("estimator")
        feature_names = obj.get("feature_names")
    else:
        model = obj
        feature_names = getattr(model, "feature_names", None)

    # — 3. Fallback to sklearn >=1.0 attribute  
    if feature_names is None and hasattr(model, "feature_names_in_"):
        feature_names = model.feature_names_in_.tolist()

    if feature_names is None:
        raise ValueError(
            "No feature_names found in the loaded object.  "
            "Be sure you saved them when you dumped the model."
        )

    # — 4. Read & subset your new data  
    df_new = pd.read_csv(new_data_path)
    X_new = df_new[feature_names]  # this will KeyError if any column is missing

    # — 5. Predict  
    return model.predict(X_new)


# usage
if __name__ == "__main__":
    preds = load_and_predict("runnable/z2.joblib", "runnable/preddata.csv")
    print(preds)
