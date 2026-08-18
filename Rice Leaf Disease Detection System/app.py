from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
import time
from gtts import gTTS
from werkzeug.utils import secure_filename

# ---------------- APP SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    template_folder=os.path.join(BASE_DIR, "templates")
)

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

# ---------------- MULTILINGUAL KNOWLEDGE BASE ----------------
DISEASE_INFO = {
    "Blast": {
        "display_name": "Rice Blast Disease",
        "en": {
            "name": "Rice Blast Disease",
            "cause": "Blast disease is a destructive fungal infection caused by Magnaporthe oryzae (Pyricularia oryzae).",
            "treatment": "Apply fungicides like Tricyclazole (0.6g/L) or Carbendazim (1g/L). Maintain balanced nitrogen fertilization and avoid excessive field wetness.",
            "speech": "Rice leaf disease analysis complete. The detected disease is Rice Blast. The infection severity level is {severity}. Cause: Blast disease is caused by Magnaporthe oryzae fungus. Treatment: Apply fungicides like Tricyclazole or Carbendazim, and maintain balanced fertilizer application."
        },
        "hi": {
            "name": "धान का ब्लास्ट रोग (Rice Blast)",
            "cause": "ब्लास्ट रोग मैग्नापोर्थे ओराइजी (कवक) के संक्रमण के कारण होता है।",
            "treatment": "ट्राइसाइक्लाज़ोल (0.6 ग्राम प्रति लीटर) या कार्बेन्डाजिम (1 ग्राम प्रति लीटर) कवकनाशी का छिड़काव करें और अत्यधिक नाइट्रोजन उर्वरक के उपयोग से बचें।",
            "speech": "धान की पत्ती की जाँच पूरी हुई। पहचानी गई बीमारी धान का ब्लास्ट रोग है। रोग की गंभीरता {severity} है। कारण: यह मैग्नापोर्थे कवक के संक्रमण से होता है। उपचार: ट्राइसाइक्लाज़ोल या कार्बेन्डाजिम कवकनाशी का छिड़काव करें और पौधों के बीच उचित दूरी रखें।"
        },
        "kn": {
            "name": "ಭತ್ತದ ಬ್ಲಾಸ್ಟ್ ರೋಗ (Rice Blast)",
            "cause": "ಬ್ಲಾಸ್ಟ್ ರೋಗವು ಮ್ಯಾಗ್ನಾಪೋರ್ತೆ ಒರೈಜೆ ಎಂಬ ಶಿಲೀಂಧ್ರ ಸೋಂಕಿನಿಂದ ಉಂಟಾಗುತ್ತದೆ.",
            "treatment": "ಟ್ರೈಸೈಕ್ಲಾಜೋಲ್ ಅಥವಾ ಕಾರ್ಬೆಂಡಾಜಿಮ್ ಶಿಲೀಂಧ್ರನಾಶಕವನ್ನು ಸಿಂಪಡಿಸಿ. ಅತಿಯಾದ ಸಾರಜನಕ ರಸಗೊಬ್ಬರ ಬಳಕೆಯನ್ನು ತಪ್ಪಿಸಿ ಮತ್ತು ಹೊಲದಲ್ಲಿ ಸರಿಯಾದ ನೀರು ನಿರ್ವಹಣೆ ಮಾಡಿ.",
            "speech": "ಭತ್ತದ ಎಲೆಯ ತಪಾಸಣೆ ಪೂರ್ಣಗೊಂಡಿದೆ. ಪತ್ತೆಯಾದ ರೋಗ ಭತ್ತದ ಬ್ಲಾಸ್ಟ್ ರೋಗ. ರೋಗದ ತೀವ್ರತೆ {severity}. ಕಾರಣ: ಈ ರೋಗವು ಮ್ಯಾಗ್ನಾಪೋರ್ತೆ ಶಿಲೀಂಧ್ರದಿಂದ ಬರುತ್ತದೆ. ಚಿಕಿತ್ಸೆ: ಟ್ರೈಸೈಕ್ಲಾಜೋಲ್ ಅಥವಾ ಕಾರ್ಬೆಂಡಾಜಿಮ್ ಶಿಲೀಂಧ್ರನಾಶಕವನ್ನು ಸಿಂಪಡಿಸಿ ಮತ್ತು ಸಾರಜನಕ ಗೊಬ್ಬರವನ್ನು ನಿಯಂತ್ರಿಸಿ."
        }
    },
    "Bacterialblight": {
        "display_name": "Bacterial Leaf Blight",
        "en": {
            "name": "Bacterial Leaf Blight",
            "cause": "Bacterial blight is caused by Xanthomonas oryzae pv. oryzae bacteria infecting the leaf vascular system.",
            "treatment": "Spray Copper Oxychloride (2.5g/L) combined with Streptocycline (0.1g/L). Use certified disease-free seeds and avoid excessive nitrogen application.",
            "speech": "Rice leaf disease analysis complete. The detected disease is Bacterial Leaf Blight. The infection severity level is {severity}. Cause: Caused by Xanthomonas oryzae bacteria infecting the leaves. Treatment: Spray Copper Oxychloride with Streptocycline and use disease-free seeds."
        },
        "hi": {
            "name": "बैक्टीरियल लीफ ब्लाइट (जीवाणु पत्ती झुलसा)",
            "cause": "यह रोग ज़ैंथोमोनास ओराइजी जीवाणु (बैक्टीरिया) के संक्रमण से होता है।",
            "treatment": "कॉपर ऑक्सीक्लोराइड (2.5 ग्राम प्रति लीटर) के साथ स्ट्रेप्टोसाइक्लिन (0.1 ग्राम प्रति लीटर) का छिड़काव करें। प्रमाणित रोगमुक्त बीजों का उपयोग करें।",
            "speech": "धान की पत्ती की जाँच पूरी हुई। पहचानी गई बीमारी बैक्टीरियल लीफ ब्लाइट है। रोग की गंभीरता {severity} है। कारण: यह ज़ैंथोमोनास जीवाणु के संक्रमण से होता है। उपचार: कॉपर ऑक्सीक्लोराइड और स्ट्रेप्टोसाइक्लिन का छिड़काव करें।"
        },
        "kn": {
            "name": "ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಲೀಫ್ ಬ್ಲೈಟ್ (ದುಂಡಾಣು ಎಲೆ ಕರಕು ರೋಗ)",
            "cause": "ಈ ರೋಗವು ಕ್ಸಾಂಥೋಮೊನಾಸ್ ಒರೈಜೆ ಎಂಬ ಬ್ಯಾಕ್ಟೀರಿಯಾದ ಸೋಂಕಿನಿಂದ ಉಂಟಾಗುತ್ತದೆ.",
            "treatment": "ಕಾಪರ್ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ ಮತ್ತು ಸ್ಟ್ರೆಪ್ಟೋಸೈಕ್ಲಿನ್ ದ್ರಾವಣವನ್ನು ಸಿಂಪಡಿಸಿ. ರೋಗಮುಕ್ತ ಬೀಜಗಳನ್ನು ಬಳಸಿ ಮತ್ತು ಅತಿಯಾದ ಸಾರಜನಕ ರಸಗೊಬ್ಬರವನ್ನು ತಪ್ಪಿಸಿ.",
            "speech": "ಭತ್ತದ ಎಲೆಯ ತಪಾಸಣೆ ಪೂರ್ಣಗೊಂಡಿದೆ. ಪತ್ತೆಯಾದ ರೋಗ ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಲೀಫ್ ಬ್ಲೈಟ್. ರೋಗದ ತೀವ್ರತೆ {severity}. ಕಾರಣ: ಕ್ಸಾಂಥೋಮೊನಾಸ್ ಬ್ಯಾಕ್ಟೀರಿಯಾದಿಂದ ಬರುತ್ತದೆ. ಚಿಕಿತ್ಸೆ: ಕಾಪರ್ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ ಮತ್ತು ಸ್ಟ್ರೆಪ್ಟೋಸೈಕ್ಲಿನ್ ಔಷಧಿಯನ್ನು ಸಿಂಪಡಿಸಿ."
        }
    },
    "Brownspot": {
        "display_name": "Brown Spot Disease",
        "en": {
            "name": "Brown Spot Disease",
            "cause": "Brown spot is a fungal disease caused by Bipolaris oryzae (Helminthosporium oryzae), often aggravated by soil nutrient deficiency.",
            "treatment": "Apply Mancozeb (2g/L) or Edifenphos. Provide balanced soil nutrition including adequate potassium and zinc fertilizers.",
            "speech": "Rice leaf disease analysis complete. The detected disease is Brown Spot disease. The infection severity level is {severity}. Cause: Caused by Bipolaris oryzae fungus and poor soil nutrition. Treatment: Apply Mancozeb fungicide and enrich soil with potassium and zinc."
        },
        "hi": {
            "name": "धान का भूरा धब्बा रोग (Brown Spot)",
            "cause": "ब्राउन स्पॉट रोग बाइपोलारिस ओराइजी कवक के संक्रमण और मिट्टी में पोषक तत्वों की कमी के कारण होता है।",
            "treatment": "मैंकोज़ेब (2 ग्राम प्रति लीटर) कवकनाशी का छिड़काव करें और मिट्टी में पोटाश तथा जिंक जैसे आवश्यक पोषक तत्व संतुलित मात्रा में दें।",
            "speech": "धान की पत्ती की जाँच पूरी हुई। पहचानी गई बीमारी भूरा धब्बा रोग यानी ब्राउन स्पॉट है। रोग की गंभीरता {severity} है। कारण: यह बाइपोलारिस कवक और पोषक तत्वों की कमी से होता है। उपचार: मैंकोज़ेब कवकनाशी का छिड़काव करें और मिट्टी में पोटाश व जिंक दें।"
        },
        "kn": {
            "name": "ಕಂದು ಚುಕ್ಕೆ ರೋಗ (Brown Spot)",
            "cause": "ಈ ರೋಗವು ಬೈಪೋಲಾರಿಸ್ ಒರೈಜೆ ಶಿಲೀಂಧ್ರ ಮತ್ತು ಮಣ್ಣಿನಲ್ಲಿ ಪೋಷಕಾಂಶಗಳ ಕೊರತೆಯಿಂದ ಉಂಟಾಗುತ್ತದೆ.",
            "treatment": "ಮ್ಯಾಂಕೋಜೆಬ್ ಶಿಲೀಂಧ್ರನಾಶಕವನ್ನು ಸಿಂಪಡಿಸಿ. ಮಣ್ಣಿಗೆ ಪೊಟ್ಯಾಶ್ ಮತ್ತು ಸತು ಒಳಗೊಂಡ ಸಮತೋಲಿತ ಪೋಷಕಾಂಶಗಳನ್ನು ಒದಗಿಸಿ.",
            "speech": "ಭತ್ತದ ಎಲೆಯ ತಪಾಸಣೆ ಪೂರ್ಣಗೊಂಡಿದೆ. ಪತ್ತೆಯಾದ ರೋಗ ಕಂದು ಚುಕ್ಕೆ ರೋಗ. ರೋಗದ ತೀವ್ರತೆ {severity}. ಕಾರಣ: ಬೈಪೋಲಾರಿಸ್ ಶಿಲೀಂಧ್ರ ಮತ್ತು ಪೋಷಕಾಂಶದ ಕೊರತೆಯಿಂದ ಉಂಟಾಗುತ್ತದೆ. ಚಿಕಿತ್ಸೆ: ಮ್ಯಾಂಕೋಜೆಬ್ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ ಹಾಗೂ ಪೊಟ್ಯಾಶ್ ಮತ್ತು ಸತು ಗೊಬ್ಬರ ನೀಡಿ."
        }
    },
    "Tungro": {
        "display_name": "Rice Tungro Disease",
        "en": {
            "name": "Rice Tungro Disease",
            "cause": "Tungro is a viral disease caused by Rice Tungro Bacilliform and Spherical Viruses, transmitted by green leafhopper insects (Nephotettix virescens).",
            "treatment": "Remove and destroy diseased plants immediately. Spray Imidacloprid (0.3ml/L) or Thiamethoxam to control green leafhopper vector insects.",
            "speech": "Rice leaf disease analysis complete. The detected disease is Rice Tungro Virus disease. The infection severity level is {severity}. Cause: Viral disease transmitted by green leafhopper insects. Treatment: Remove infected plants and spray insecticides like Imidacloprid to control leafhoppers."
        },
        "hi": {
            "name": "राइस टुंग्रो वायरस रोग (Rice Tungro)",
            "cause": "टुंग्रो एक विषाणुजनित (वायरल) रोग है जो हरी पत्ती फुदके (ग्रीन लीफहॉपर) कीड़ों द्वारा फैलता है।",
            "treatment": "संक्रमित पौधों को तुरंत उखाड़कर नष्ट करें। ग्रीन लीफहॉपर कीटों को नियंत्रित करने के लिए इमिडाक्लोप्रिड या थायमेथोक्सम कीटनाशक का छिड़काव करें।",
            "speech": "धान की पत्ती की जाँच पूरी हुई। पहचानी गई बीमारी टुंग्रो वायरस रोग है। रोग की गंभीरता {severity} है। कारण: यह हरे फुदके कीड़ों द्वारा फैलने वाला वायरस है। उपचार: रोगी पौधों को नष्ट करें और इमिडाक्लोप्रिड कीटनाशक का छिड़काव करें।"
        },
        "kn": {
            "name": "ರೈಸ್ ಟುಂಗ್ರೋ ವೈರಸ್ ರೋಗ (Rice Tungro)",
            "cause": "ಟುಂಗ್ರೋ ಒಂದು ವೈರಸ್ ರೋಗವಾಗಿದ್ದು, ಹಸಿರು ಜಿಗಿಹುಳುಗಳ (ಗ್ರೀನ್ ಲೀಫ್‌ಹಾಪರ್) ಮೂಲಕ ಹರಡುತ್ತದೆ.",
            "treatment": "ಸೋಂಕಿತ ಸಸಿಗಳನ್ನು ತಕ್ಷಣ ಕಿತ್ತು ನಾಶಮಾಡಿ. ಹಸಿರು ಜಿಗಿಹುಳುಗಳನ್ನು ನಿಯಂತ್ರಿಸಲು ಇಮಿಡಾಕ್ಲೋಪ್ರಿಡ್ ಅಥವಾ ಥಯಾಮೆಥೊಕ್ಸಾಮ್ ಕೀಟನಾಶಕವನ್ನು ಸಿಂಪಡಿಸಿ.",
            "speech": "ಭತ್ತದ ಎಲೆಯ ತಪಾಸಣೆ ಪೂರ್ಣಗೊಂಡಿದೆ. ಪತ್ತೆಯಾದ ರೋಗ ರೈಸ್ ಟುಂಗ್ರೋ ರೋಗ. ರೋಗದ ತೀವ್ರತೆ {severity}. ಕಾರಣ: ಹಸಿರು ಜಿಗಿಹುಳುಗಳಿಂದ ಹರಡುವ ವೈರಸ್. ಚಿಕಿತ್ಸೆ: ರೋಗಪೀಡಿತ ಸಸಿಗಳನ್ನು ನಾಶಪಡಿಸಿ ಮತ್ತು ಇಮಿಡಾಕ್ಲೋಪ್ರಿಡ್ ಕೀಟನಾಶಕ ಸಿಂಪಡಿಸಿ ಕೀಟಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ."
        }
    }
}

SEVERITY_TRANSLATION = {
    "Mild": {
        "en": "Mild (Early Stage)",
        "hi": "हल्का (प्रारंभिक अवस्था)",
        "kn": "ಸೌಮ್ಯ (ಆರಂಭಿಕ ಹಂತ)"
    },
    "Moderate": {
        "en": "Moderate (Spreading Stage)",
        "hi": "मध्यम (मध्यम अवस्था)",
        "kn": "ಮಧ್ಯಮ (ಹರಡುವ ಹಂತ)"
    },
    "Severe": {
        "en": "Severe (Critical Stage)",
        "hi": "गंभीर (गंभीर अवस्था)",
        "kn": "ತೀವ್ರ (ಗಂಭೀರ ಹಂತ)"
    }
}

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
    info = DISEASE_INFO.get(disease, {})
    if info:
        cause = info["en"]["cause"]
        treatment = info["en"]["treatment"]
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
    os.makedirs(AUDIO_FOLDER, exist_ok=True)

    en_path = os.path.join(AUDIO_FOLDER, "result_en.mp3")
    hi_path = os.path.join(AUDIO_FOLDER, "result_hi.mp3")
    kn_path = os.path.join(AUDIO_FOLDER, "result_kn.mp3")

    info = DISEASE_INFO.get(disease, {})
    sev_en = SEVERITY_TRANSLATION.get(severity, {}).get("en", severity)
    sev_hi = SEVERITY_TRANSLATION.get(severity, {}).get("hi", severity)
    sev_kn = SEVERITY_TRANSLATION.get(severity, {}).get("kn", severity)

    if info:
        en_text = info["en"]["speech"].format(severity=sev_en)
        hi_text = info["hi"]["speech"].format(severity=sev_hi)
        kn_text = info["kn"]["speech"].format(severity=sev_kn)
    else:
        en_text = f"Disease detected is {disease}. Severity level is {sev_en}. Cause: {solution['cause']} Treatment: {solution['treatment']}"
        hi_text = f"पहचानी गई बीमारी {disease} है। बीमारी की गंभीरता {sev_hi} है। कारण: {solution['cause']} उपचार: {solution['treatment']}"
        kn_text = f"ಗುರುತಿಸಲಾದ ರೋಗ {disease}. ರೋಗದ ತೀವ್ರತೆ {sev_kn}. ಕಾರಣ: {solution['cause']} ಚಿಕಿತ್ಸೆ: {solution['treatment']}"

    try:
        gTTS(text=en_text, lang="en", slow=False).save(en_path)
    except Exception as e:
        print(f"[Warning] Failed to generate English voice audio: {e}")

    try:
        gTTS(text=hi_text, lang="hi", slow=False).save(hi_path)
    except Exception as e:
        print(f"[Warning] Failed to generate Hindi voice audio: {e}")

    try:
        gTTS(text=kn_text, lang="kn", slow=False).save(kn_path)
    except Exception as e:
        print(f"[Warning] Failed to generate Kannada voice audio: {e}")

    return en_text, hi_text, kn_text


# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="❌ Please select an image file to upload.")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="❌ No image selected.")

        safe_name = secure_filename(file.filename) or "uploaded_leaf.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None:
            return render_template("index.html", error="❌ Unable to read the uploaded image. Please upload a valid JPG or PNG.")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_W, IMG_H))
        img_norm = img_resized.astype("float32") / 255.0
        img_input = np.expand_dims(img_norm, axis=0)

        # -------- Prediction --------
        preds = model.predict(img_input)
        confidence = float(np.max(preds) * 100)

        if confidence < 60:
            return render_template(
                "index.html",
                error="❌ Invalid image or low confidence! Please upload a clear rice leaf image."
            )

        predicted_class = class_names[int(np.argmax(preds))]

        # -------- Grad-CAM --------
        gradcam_path, heatmap = generate_gradcam(model, img_input, filepath)

        # -------- Severity --------
        severity, ratio = analyze_severity(heatmap)

        # -------- Recommendation --------
        solution = ai_generate_solution(predicted_class, severity)

        # -------- Voice --------
        speech_en, speech_hi, speech_kn = generate_voice(
            solution, predicted_class, severity
        )

        # Web URLs with cache-busting timestamp
        ts = int(time.time())
        voice_en_url = f"/static/audio/result_en.mp3?t={ts}"
        voice_hi_url = f"/static/audio/result_hi.mp3?t={ts}"
        voice_kn_url = f"/static/audio/result_kn.mp3?t={ts}"
        image_url = f"/static/uploads/{safe_name}?t={ts}"
        gradcam_url = f"/static/gradcam/gradcam_result.jpg?t={ts}"

        disease_details = DISEASE_INFO.get(predicted_class, {})

        return render_template(
            "result.html",
            image_path=image_url,
            gradcam_path=gradcam_url,
            prediction=predicted_class,
            display_name=disease_details.get("display_name", predicted_class),
            disease_info=disease_details,
            confidence=round(confidence, 2),
            solution=solution,
            severity=severity,
            severity_labels=SEVERITY_TRANSLATION.get(severity, {}),
            voice_en=voice_en_url,
            voice_hi=voice_hi_url,
            voice_kn=voice_kn_url,
            speech_en=speech_en,
            speech_hi=speech_hi,
            speech_kn=speech_kn
        )

    return render_template("index.html")


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
