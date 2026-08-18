from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
from gtts import gTTS

# ---------------- APP SETUP ----------------
app = Flask(__name__)
# Resolve paths relative to this file so app works from any CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")
AUDIO_FOLDER = os.path.join(BASE_DIR, "static", "audio")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model_path = os.path.join(BASE_DIR, "CNN_trained_model.h5")
model = tf.keras.models.load_model(model_path)

_, IMG_H, IMG_W, IMG_C = model.input_shape
dummy = tf.zeros((1, IMG_H, IMG_W, IMG_C))
model(dummy)

# ---------------- LOAD CLASSES ----------------
with open(os.path.join(BASE_DIR, "class_indices.pkl"), "rb") as f:
    class_indices = pickle.load(f)

class_names = {v: k for k, v in class_indices.items()}

# ---------------- GRAD-CAM ----------------
def generate_gradcam(model, img_array, img_path):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    last_conv_index = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_index = i

    if last_conv_index is None:
        raise ValueError("No Conv2D layer found")

    with tf.GradientTape() as tape:
        x = img_tensor
        for i, layer in enumerate(model.layers):
            x = layer(x)
            if i == last_conv_index:
                conv_output = x
                tape.watch(conv_output)

        preds = x
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap ** 2.0
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.7, heatmap_color, 0.5, 0)

    gradcam_path = os.path.join(GRADCAM_FOLDER, "gradcam_result.jpg")
    cv2.imwrite(gradcam_path, superimposed)

    return gradcam_path, heatmap_resized


# ---------------- SEVERITY ANALYSIS ----------------
def analyze_severity(heatmap):
    infected_ratio = np.sum(heatmap > 0.45) / heatmap.size

    if infected_ratio < 0.15:
        return "Mild", infected_ratio
    elif infected_ratio < 0.35:
        return "Moderate", infected_ratio
    else:
        return "Severe", infected_ratio


# ---------------- AI RECOMMENDATION ----------------
def ai_generate_solution(disease, severity):

    if disease == "Tungro":
        cause = "Tungro disease is caused by Rice Tungro Virus transmitted by green leafhopper insects."
        treatment = "Remove infected plants and control insects using recommended pesticides."

    elif disease == "Blast":
        cause = "Blast disease is a fungal infection caused by Magnaporthe oryzae."
        treatment = "Apply fungicides and maintain proper plant spacing."

    elif disease == "Bacterialblight":
        cause = "Bacterial blight is caused by Xanthomonas oryzae bacteria."
        treatment = "Use disease-free seeds and avoid excess nitrogen."

    elif disease == "Brownspot":
        cause = "Brown spot occurs due to fungal infection and poor soil nutrition."
        treatment = "Improve soil nutrients and apply fungicide if needed."

    else:
        cause = "Pathogenic infection affecting crop leaves."
        treatment = "Follow proper agronomic and chemical control methods."

    severity_note = f"The infection severity is {severity}."

    return {
        "cause": cause,
        "treatment": treatment,
        "severity_note": severity_note
    }


# ---------------- AI VOICE GENERATION ----------------
def generate_voice(solution, disease, severity):

    english_text = f"""
    Disease detected is {disease}.
    Severity level is {severity}.
    Cause: {solution['cause']}
    Treatment: {solution['treatment']}
    """

    hindi_text = f"""
    पहचानी गई बीमारी {disease} है।
    बीमारी की गंभीरता {severity} है।
    कारण: {solution['cause']}
    उपचार: {solution['treatment']}
    """

    kannada_text = f"""
    ಗುರುತಿಸಲಾದ ರೋಗ {disease}.
    ರೋಗದ ತೀವ್ರತೆ ಮಟ್ಟ {severity}.
    ಕಾರಣ: {solution['cause']}
    ಚಿಕಿತ್ಸೆ: {solution['treatment']}
    """

    en_path = os.path.join(AUDIO_FOLDER, "result_en.mp3")
    hi_path = os.path.join(AUDIO_FOLDER, "result_hi.mp3")
    kn_path = os.path.join(AUDIO_FOLDER, "result_kn.mp3")

    gTTS(text=english_text, lang="en").save(en_path)
    gTTS(text=hindi_text, lang="hi").save(hi_path)
    gTTS(text=kannada_text, lang="kn").save(kn_path)

    return en_path, hi_path, kn_path


# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files["image"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_W, IMG_H))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # -------- Prediction --------
        preds = model.predict(img)
        confidence = float(np.max(preds) * 100)

        if confidence < 60:
            return render_template(
                "index.html",
                error="❌ Invalid image! Please upload a clear rice leaf image."
            )

        predicted_class = class_names[int(np.argmax(preds))]

        # -------- Grad-CAM --------
        gradcam_path, heatmap = generate_gradcam(model, img, filepath)

        # -------- Severity --------
        severity, ratio = analyze_severity(heatmap)

        # -------- Recommendation --------
        solution = ai_generate_solution(predicted_class, severity)

        # -------- Voice --------
        voice_en, voice_hi, voice_kn = generate_voice(
            solution, predicted_class, severity
        )

        return render_template(
            "result.html",
            image_path=filepath,
            gradcam_path=gradcam_path,
            prediction=predicted_class,
            confidence=round(confidence, 2),
            solution=solution,
            severity=severity,
            voice_en=voice_en,
            voice_hi=voice_hi,
            voice_kn=voice_kn
        )

    return render_template("index.html")


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
