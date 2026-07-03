"""
model.py — loads the trained DenseNet121 (SIPaKMeD, 5 classes) and runs inference.
"""

import io
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = Path(__file__).parent / "model" / "densenet_sipakmed_best.pt"

CLASS_NAMES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate",
]

# Clinical risk mapping — adjust to match your actual clinical labeling logic
RISK_MAP = {
    "Dyskeratotic": "High Risk",
    "Koilocytotic": "Moderate Risk",
    "Metaplastic": "Low Risk",
    "Parabasal": "Normal",
    "Superficial-Intermediate": "Normal",
}

CLINICAL_NOTES = {
    "Dyskeratotic": (
        "Dyskeratotic cell morphology detected. Nuclear atypia and abnormal "
        "keratinization patterns present. Recommend colposcopy and second-opinion review."
    ),
    "Koilocytotic": (
        "Koilocytotic changes observed, commonly associated with HPV cytopathic effect. "
        "Recommend HPV testing and follow-up cytology in 6-12 months."
    ),
    "Metaplastic": (
        "Metaplastic squamous cells identified, typically a benign reparative process "
        "in the transformation zone. Routine follow-up recommended."
    ),
    "Parabasal": (
        "Parabasal cells within normal limits. No significant atypia detected. "
        "Routine screening interval advised."
    ),
    "Superficial-Intermediate": (
        "Superficial-intermediate cells consistent with normal maturation. "
        "No dysplastic features observed. Routine screening interval advised."
    ),
}

# Plain-language educational panel content shown alongside each result.
EDUCATIONAL_INFO = {
    "Dyskeratotic": {
        "what_it_is": "Cells showing premature or abnormal keratinization, often linked to dysplastic change.",
        "why_it_matters": "Associated with a higher likelihood of precancerous or cancerous change and usually warrants closer follow-up.",
        "typical_next_step": "Colposcopy and possible biopsy, guided by a specialist.",
    },
    "Koilocytotic": {
        "what_it_is": "Cells with a clear halo around the nucleus, a classic sign of HPV infection.",
        "why_it_matters": "HPV is the primary driver of cervical cancer, so these cells flag infection status rather than cancer itself.",
        "typical_next_step": "HPV testing and repeat cytology in 6-12 months.",
    },
    "Metaplastic": {
        "what_it_is": "Cells undergoing a normal tissue-repair process in the cervix's transformation zone.",
        "why_it_matters": "Usually a benign, expected finding, not a disease process on its own.",
        "typical_next_step": "Routine screening interval, no immediate action required.",
    },
    "Parabasal": {
        "what_it_is": "Deeper, immature squamous cells, common in atrophic or low-estrogen states.",
        "why_it_matters": "Typically benign; more common post-menopause or postpartum.",
        "typical_next_step": "Routine screening interval.",
    },
    "Superficial-Intermediate": {
        "what_it_is": "Mature, well-differentiated squamous cells representing healthy cervical tissue.",
        "why_it_matters": "This is the expected, normal finding in a healthy cervix.",
        "typical_next_step": "Routine screening interval.",
    },
}

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    """Loads DenseNet121 and rebuilds the classifier head to match the checkpoint exactly."""
    global _model
    if _model is not None:
        return _model

    net = models.densenet121(weights=None)

    state_dict = torch.load(MODEL_PATH, map_location=_device, weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Detect the trained classifier head shape from the checkpoint itself,
    # instead of assuming a single nn.Linear(1024, 5).
    if "classifier.weight" in state_dict:
        # simple single Linear layer
        out_features = state_dict["classifier.weight"].shape[0]
        net.classifier = nn.Linear(net.classifier.in_features, out_features)
    elif "classifier.1.weight" in state_dict and "classifier.4.weight" in state_dict:
        # Dropout -> Linear -> ReLU -> Dropout -> Linear head
        hidden = state_dict["classifier.1.weight"].shape[0]
        out_features = state_dict["classifier.4.weight"].shape[0]
        net.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(net.classifier.in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, out_features),
        )
    else:
        raise RuntimeError(
            "Unrecognized classifier head structure in checkpoint. "
            f"Keys found: {[k for k in state_dict.keys() if k.startswith('classifier')]}"
        )

    net.load_state_dict(state_dict)
    net.to(_device)
    net.eval()
    _model = net
    return _model


def predict(image_bytes: bytes):
    """Runs inference on raw image bytes. Returns (predicted_class, confidence, all_probs dict)."""
    net = load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = net(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    probs_list = probs.cpu().tolist()
    all_probs = {CLASS_NAMES[i]: round(probs_list[i] * 100, 2) for i in range(len(CLASS_NAMES))}

    top_idx = int(torch.argmax(probs).item())
    predicted_class = CLASS_NAMES[top_idx]
    confidence = round(probs_list[top_idx] * 100, 2)

    return predicted_class, confidence, all_probs


def get_risk(cell_type: str) -> str:
    return RISK_MAP.get(cell_type, "Normal")


def get_clinical_note(cell_type: str) -> str:
    return CLINICAL_NOTES.get(cell_type, "No additional clinical notes available.")


def get_educational_info(cell_type: str) -> dict:
    return EDUCATIONAL_INFO.get(cell_type, {})


# ---------------------------------------------------------------------------
# Grad-CAM — highlights the region of the image that drove the prediction.
# Hooks DenseNet121's final conv feature map (features.norm5 output).
# ---------------------------------------------------------------------------
import numpy as np
from PIL import ImageOps

_gradcam_activations = {}
_gradcam_gradients = {}


def _register_gradcam_hooks(net):
    """
    Uses a forward hook to grab the feature map, then registers a gradient hook
    directly on that (cloned) tensor — avoids register_full_backward_hook, which
    conflicts with DenseNet's in-place ReLU operations and throws a view/inplace error.
    """
    target_layer = net.features.norm5

    def forward_hook(module, inp, out):
        activation = out.clone()
        activation.retain_grad()
        _gradcam_activations["value"] = activation

        def grad_hook(grad):
            _gradcam_gradients["value"] = grad.detach()

        activation.register_hook(grad_hook)
        # replace the layer's output with our cloned, grad-tracked tensor so the
        # rest of the forward pass (and backward pass) flows through it
        return activation

    h1 = target_layer.register_forward_hook(forward_hook)
    return (h1,)


def generate_gradcam(image_bytes: bytes, target_class_idx: int) -> Image.Image:
    """Runs a second forward+backward pass to build a Grad-CAM heatmap overlaid on the input image."""
    net = load_model()
    hooks = _register_gradcam_hooks(net)

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = _transform(image).unsqueeze(0).to(_device)
        tensor.requires_grad_(True)

        net.zero_grad()
        logits = net(tensor)
        score = logits[0, target_class_idx]
        score.backward()

        activations = _gradcam_activations["value"][0]   # [C, H, W]
        gradients = _gradcam_gradients["value"][0]        # [C, H, W]

        weights = gradients.mean(dim=(1, 2))               # [C]
        cam = torch.relu((weights[:, None, None] * activations).sum(0))  # [H, W]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.detach().cpu().numpy()

        heatmap = Image.fromarray(np.uint8(cam_np * 255)).resize(image.size, resample=Image.BILINEAR)
        heatmap_colored = ImageOps.colorize(heatmap, black="#0a1420", white="#ff4d4d", mid="#ffb020").convert("RGB")

        overlay = Image.blend(image, heatmap_colored, alpha=0.45)
        return overlay
    finally:
        for h in hooks:
            h.remove()
        _gradcam_activations.clear()
        _gradcam_gradients.clear()