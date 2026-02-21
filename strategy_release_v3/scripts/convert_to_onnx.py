import os
import joblib
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "experiments")
joblib_path = os.path.join(OUT_DIR, "block_e_xgboost_overlay.joblib")
onnx_path = os.path.join(OUT_DIR, "block_e_xgboost_overlay.onnx")

print(f"Loading joblib model from {joblib_path}...")
model = joblib.load(joblib_path)

# Rename features in the booster to f0, f1, ..., f13 to satisfy ONNX
booster = model.get_booster()
booster.feature_names = [f"f{i}" for i in range(len(booster.feature_names))]

# We have 14 features in the dataset
initial_types = [('float_input', FloatTensorType([None, 14]))]

print("Converting to ONNX...")
onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_types)

print(f"Saving ONNX model to {onnx_path}...")
onnxmltools.utils.save_model(onnx_model, onnx_path)
print("Done!")
