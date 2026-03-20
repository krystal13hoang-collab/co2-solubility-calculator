from flask import Flask, request, jsonify, render_template
import os
import pickle
import pandas as pd

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MODEL_DIR = os.path.join(BASE_DIR, "models_out")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

MODEL_PATHS = {
    "pure_rf": os.path.join(MODEL_DIR, "pure_rf.pkl"),
    "pure_gb": os.path.join(MODEL_DIR, "pure_gb.pkl"),
    "brine_rf": os.path.join(MODEL_DIR, "brine_rf.pkl"),
    "brine_gb": os.path.join(MODEL_DIR, "brine_gb.pkl"),
}

FEATURE_COLS = ["T (K)", "P (MPa)", "Ionic Strength"]


def try_load_model(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


MODELS = {k: try_load_model(p) for k, p in MODEL_PATHS.items()}


def require_model(key: str):
    m = MODELS.get(key)
    if m is None:
        raise FileNotFoundError(
            f"Missing model file: {os.path.basename(MODEL_PATHS[key])}"
        )
    return m


def available_model_options():
    return ["Pure_Ensemble", "Brine_Ensemble"]


def parse_payload(req):
    if req.is_json:
        return req.get_json(silent=True) or {}
    return req.form.to_dict()


def make_features_df(T_K: float, P_MPa: float, I: float) -> pd.DataFrame:
    return pd.DataFrame([[T_K, P_MPa, I]], columns=FEATURE_COLS)


def compute_prediction(selected_model: str, T_K: float, P_MPa: float, I: float) -> float:
    X = make_features_df(T_K=T_K, P_MPa=P_MPa, I=I)

    if selected_model == "Pure_Ensemble":
        a = require_model("pure_rf").predict(X)[0]
        b = require_model("pure_gb").predict(X)[0]
        return float((a + b) / 2)

    if selected_model == "Brine_Ensemble":
        a = require_model("brine_rf").predict(X)[0]
        b = require_model("brine_gb").predict(X)[0]
        return float((a + b) / 2)

    raise ValueError(f"Invalid model selected: {selected_model}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "available_model_options": available_model_options(),
            "models_found": {k: (MODELS[k] is not None) for k in MODELS},
        }
    )


@app.route("/", methods=["GET", "POST"])
def home():
    ctx = dict(
        available_models=available_model_options(),
        pressure="",
        temperature="",
        ionic_strength="0",
        model="Pure_Ensemble",
        prediction=None,
        error=None,
    )

    if request.method == "GET":
        return render_template("index.html", **ctx)

    try:
        data = parse_payload(request)

        for k in ("pressure", "temperature", "model"):
            if k not in data or str(data[k]).strip() == "":
                raise ValueError(f"Missing value for '{k}'")

        selected_model = str(data["model"]).strip()
        P_MPa = float(data["pressure"])
        T_K = float(data["temperature"])

        if P_MPa <= 0 or T_K <= 0:
            raise ValueError("Pressure and Temperature must be positive.")

        if selected_model == "Pure_Ensemble":
            I = 0.0

        elif selected_model == "Brine_Ensemble":
            if "ionic_strength" not in data or str(data["ionic_strength"]).strip() == "":
                raise ValueError("Ionic Strength is required for Brine-Ensemble.")
            I = float(data["ionic_strength"])
            if I < 0:
                raise ValueError("Ionic Strength must be >= 0.")

        else:
            raise ValueError(f"Invalid model selected: {selected_model}")

        pred = round(compute_prediction(selected_model, T_K=T_K, P_MPa=P_MPa, I=I), 6)

        if request.is_json:
            return jsonify({"prediction": pred})

        ctx.update(
            pressure=P_MPa,
            temperature=T_K,
            ionic_strength=I,
            model=selected_model,
            prediction=pred,
            error=None,
        )
        return render_template("index.html", **ctx)

    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 400
        ctx["error"] = str(e)
        return render_template("index.html", **ctx), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
