"""
utils.py — Fonctions utilitaires pour la collecte, le nettoyage et la normalisation.
"""

import re
import math
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation de texte
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Nettoie et normalise un texte brut."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)           # suppr. HTML
    text = re.sub(r"http\S+", " ", text)             # suppr. URLs
    text = re.sub(r"[^\w\s\-.,;:!?()/+#@]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Détection de technologies
# ─────────────────────────────────────────────────────────────────────────────

# Ordre : clé = terme recherché (lowercase), valeur = label normalisé
TECH_PATTERNS: dict[str, str] = {
    # Frameworks DL
    r"\bpytorch\b|\btorch\b":                         "PyTorch",
    r"\btensorflow\b|\btf\b":                         "TensorFlow",
    r"\bjax\b":                                        "JAX",
    r"\bkeras\b":                                      "Keras",
    r"\bpaddle\b|\bpaddlepaddle\b":                   "PaddlePaddle",
    r"\bmxnet\b":                                      "MXNet",
    # Transformers / NLP
    r"\btransformers\b|\bhuggingface\b|\bhf\b":       "HuggingFace Transformers",
    r"\bbert\b":                                       "BERT",
    r"\bgpt\b|\bgpt-?\d":                              "GPT",
    r"\bllama\b":                                      "LLaMA",
    r"\bt5\b":                                         "T5",
    r"\blora\b|\bqlora\b":                             "LoRA",
    r"\bllm\b":                                        "LLM",
    r"\bspacy\b":                                      "spaCy",
    r"\bnltk\b":                                       "NLTK",
    # CV
    r"\bopencv\b":                                     "OpenCV",
    r"\byolo\b":                                       "YOLO",
    r"\bvit\b|vision transformer":                    "Vision Transformer",
    r"\bsegment anything\b|\bsam\b":                  "SAM",
    r"\bdiffusion\b|\bstable diffusion\b":            "Diffusion",
    r"\bgan\b":                                        "GAN",
    r"\bvae\b":                                        "VAE",
    # Audio
    r"\bwhisper\b":                                    "Whisper",
    r"\bwav2vec\b":                                    "wav2vec",
    r"\baudio\b|\bspeech\b":                           "Audio/Speech",
    # RL
    r"\breinforcement learning\b|\brl\b":             "Reinforcement Learning",
    r"\bstable.baselines\b":                           "Stable-Baselines",
    r"\bgymnasium\b|\bgym\b":                          "Gymnasium",
    # Data / ML
    r"\bscikit.learn\b|\bsklearn\b":                  "scikit-learn",
    r"\bxgboost\b":                                    "XGBoost",
    r"\blightgbm\b":                                   "LightGBM",
    r"\bpandas\b":                                     "pandas",
    r"\bnumpy\b":                                      "NumPy",
    r"\bscipy\b":                                      "SciPy",
    # Deploy / infra
    r"\bfastapi\b":                                    "FastAPI",
    r"\bflask\b":                                      "Flask",
    r"\bstreamlit\b":                                  "Streamlit",
    r"\bgradio\b":                                     "Gradio",
    r"\bdocker\b":                                     "Docker",
    r"\bkubernetes\b|\bk8s\b":                        "Kubernetes",
    r"\bonnx\b":                                       "ONNX",
    r"\btriton\b":                                     "Triton",
    r"\bmlflow\b":                                     "MLflow",
    r"\bwandb\b|weights.and.biases":                  "Weights & Biases",
    r"\bray\b|\bray tune\b":                           "Ray",
    r"\blangchain\b":                                  "LangChain",
    r"\bllamaindex\b":                                 "LlamaIndex",
    r"\bchromadb\b|\bpinecone\b|\bweaviate\b":        "Vector DB",
    # Langages
    r"\bpython\b":                                     "Python",
    r"\bjulia\b":                                      "Julia",
    r"\br\b":                                          "R",
    r"\bcuda\b":                                       "CUDA",
}

def detect_technologies(text: str) -> list[str]:
    """Détecte les technologies mentionnées dans un texte."""
    text_lower = text.lower()
    found = []
    seen_labels = set()
    for pattern, label in TECH_PATTERNS.items():
        if re.search(pattern, text_lower) and label not in seen_labels:
            found.append(label)
            seen_labels.add(label)
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Extraction de tags
# ─────────────────────────────────────────────────────────────────────────────

AI_KEYWORDS = {
    "classification", "detection", "segmentation", "generation", "translation",
    "summarization", "question-answering", "embedding", "clustering", "regression",
    "anomaly-detection", "forecasting", "recommendation", "ocr", "nlp", "cv",
    "audio", "multimodal", "self-supervised", "zero-shot", "few-shot",
    "fine-tuning", "pre-training", "distillation", "pruning", "quantization",
    "medical", "healthcare", "finance", "robotics", "autonomous", "satellite",
    "biology", "drug-discovery", "climate", "chemistry", "education",
    "speech", "text", "image", "video", "3d", "point-cloud", "graph",
    "cnn", "rnn", "lstm", "gru", "attention", "transformer", "encoder", "decoder",
}

def extract_tags_from_text(text: str, max_tags: int = 15) -> list[str]:
    """Extrait des tags IA pertinents depuis un texte libre."""
    text_lower = text.lower()
    tags = []
    for kw in AI_KEYWORDS:
        # Recherche le mot-clé ou sa variante avec tiret/underscore
        pattern = re.escape(kw).replace(r"\-", r"[-_]")
        if re.search(r"\b" + pattern + r"\b", text_lower):
            tags.append(kw)
    return tags[:max_tags]


# ─────────────────────────────────────────────────────────────────────────────
# Score de popularité normalisé
# ─────────────────────────────────────────────────────────────────────────────

def score_popularity(
    stars: int = 0,
    forks: int = 0,
    watchers: int = 0,
    downloads: int = 0,
) -> float:
    """
    Calcule un score de popularité [0.0 – 1.0] via log-normalisation.
    Poids : stars(40%) + forks(30%) + watchers(15%) + downloads(15%).
    """
    def log_norm(val: int, scale: float) -> float:
        return min(math.log1p(val) / math.log1p(scale), 1.0)

    s = log_norm(stars, 10_000)       * 0.40
    f = log_norm(forks, 3_000)        * 0.30
    w = log_norm(watchers, 5_000)     * 0.15
    d = log_norm(downloads, 100_000)  * 0.15
    return round(s + f + w + d, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Inférence de domaine
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Règles de domaine — deux niveaux de signaux
#
# Chaque domaine a :
#   "tags"  : mots cherchés dans les TAGS HuggingFace / GitHub topics
#             (termes courts, exacts, souvent des pipeline_tag)
#   "text"  : mots cherchés dans la DESCRIPTION (substring, lowercase)
#             (termes plus larges, variantes acceptées)
#
# Priorité : ordre de la liste — le premier match gagne.
# On teste d'abord les domaines les plus spécifiques/rares pour éviter
# qu'un terme générique ("text", "image") écrase un domaine précis.
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_RULES: list[dict] = [

    # ── Très spécifiques en premier ──────────────────────────────────────

    {   "domain": "Medical / Healthcare",
        "tags":   ["medical-imaging", "healthcare", "clinical-nlp", "biomedical",
                   "radiology", "pathology", "ecg", "ehr", "drug-discovery",
                   "protein", "genomics", "ophthalmology", "dermatology"],
        "text":   ["medical", "healthcare", "clinical", "cancer", "tumor",
                   "pathology", "radiology", "mri", "x-ray", "xray", "ecg",
                   "drug discovery", "patient", "diagnosis", "hospital",
                   "biomedical", "genomic", "protein folding", "ophthalmolog",
                   "dermatolog", "histolog", "microscop"],
    },
    {   "domain": "Finance",
        "tags":   ["finance", "trading", "fintech", "fraud-detection",
                   "credit-scoring", "stock-market"],
        "text":   ["finance", "trading", "stock", "market", "fraud",
                   "credit", "portfolio", "investment", "fintech",
                   "cryptocurrency", "bitcoin", "risk model"],
    },
    {   "domain": "Robotics / RL",
        "tags":   ["reinforcement-learning", "robotics", "gym", "mujoco",
                   "isaac-gym", "navigation", "control"],
        "text":   ["reinforcement learning", "robot", "autonomous",
                   "navigation", "control policy", "mujoco", "gymnasium",
                   "actor-critic", "policy gradient", "dqn", "ppo", "sac"],
    },
    {   "domain": "3D / Point Cloud",
        "tags":   ["3d", "point-cloud", "nerf", "lidar", "3d-reconstruction",
                   "mesh", "depth-estimation", "mvs"],
        "text":   ["point cloud", "3d reconstruction", "nerf", "lidar",
                   "mesh", "voxel", "depth estimation", "stereo", "mvs",
                   "3d generation", "gaussian splatting"],
    },
    {   "domain": "Time Series",
        "tags":   ["time-series", "forecasting", "anomaly-detection",
                   "signal-processing", "iot"],
        "text":   ["time series", "forecasting", "temporal", "anomaly detection",
                   "sensor", "iot", "signal processing", "sequential data",
                   "arima", "lstm forecasting"],
    },
    {   "domain": "Graph / Network",
        "tags":   ["graph-neural-network", "gnn", "knowledge-graph",
                   "link-prediction", "node-classification"],
        "text":   ["graph neural", "gnn", "knowledge graph", "node classification",
                   "link prediction", "graph embedding", "heterogeneous graph",
                   "molecular graph"],
    },

    # ── Generative AI — large filet sur les tags HF ──────────────────────

    {   "domain": "Generative AI",
        "tags":   [
            # pipeline tags HuggingFace
            "text-to-image", "image-to-image", "unconditional-image-generation",
            "text-to-video", "text-to-3d", "text-to-audio",
            "image-to-video", "video-generation",
            # topics GitHub / HF
            "stable-diffusion", "diffusion", "latent-diffusion",
            "gan", "vae", "generative-model", "image-generation",
            "diffusers", "controlnet", "lora", "dreambooth",
            "midjourney", "dall-e", "flux",
        ],
        "text":   ["stable diffusion", "diffusion model", "generative adversarial",
                   "image generation", "text-to-image", "dall-e", "midjourney",
                   "gan", "vae decoder", "latent diffusion", "controlnet",
                   "dreambooth", "inpainting", "outpainting", "image synthesis",
                   "video generation", "text-to-video", "flow matching",
                   "gaussian splatting", "generative model"],
    },

    # ── Multimodal — avant NLP et CV pour éviter absorption ──────────────

    {   "domain": "Multimodal",
        "tags":   ["visual-question-answering", "image-text-to-text",
                   "document-question-answering", "video-text-to-text",
                   "multimodal", "vision-language", "vqa", "clip",
                   "image-captioning", "grounding"],
        "text":   ["multimodal", "vision-language", "vqa", "clip",
                   "image captioning", "visual question", "image-text",
                   "grounding dino", "llava", "flamingo", "blip",
                   "document understanding", "layout"],
    },

    # ── Audio — avant NLP car "text" est dans beaucoup de TTS ────────────

    {   "domain": "Audio / Speech",
        "tags":   ["automatic-speech-recognition", "text-to-speech",
                   "audio-classification", "audio-to-audio",
                   "voice-activity-detection", "speech-synthesis",
                   "music-generation", "sound-classification"],
        "text":   ["speech recognition", "text-to-speech", "asr",
                   "tts", "whisper", "wav2vec", "audio classification",
                   "speaker", "voice", "music generation", "sound",
                   "speech synthesis", "voice cloning", "audio model"],
    },

    # ── NLP / Text ────────────────────────────────────────────────────────

    {   "domain": "NLP / Text",
        "tags":   ["text-classification", "token-classification",
                   "question-answering", "text-generation", "text2text-generation",
                   "summarization", "translation", "fill-mask",
                   "sentence-similarity", "feature-extraction",
                   "zero-shot-classification", "nlp", "llm",
                   "named-entity-recognition", "sentiment-analysis",
                   "text-mining", "information-extraction"],
        "text":   ["natural language", "nlp", "text classification",
                   "sentiment", "named entity", "question answering",
                   "summarization", "translation", "language model",
                   "bert", "gpt", "llm", "transformer", "tokenizer",
                   "embedding", "chatbot", "dialogue", "information extraction",
                   "text mining", "document classification"],
    },

    # ── Computer Vision — en dernier car "image" est partout ─────────────

    {   "domain": "Computer Vision",
        "tags":   ["image-classification", "object-detection", "image-segmentation",
                   "zero-shot-image-classification", "depth-estimation",
                   "image-feature-extraction", "video-classification",
                   "pose-estimation", "face-recognition", "ocr",
                   "action-recognition", "panoptic-segmentation"],
        "text":   ["image classification", "object detection", "segmentation",
                   "computer vision", "convolutional", "cnn", "resnet",
                   "efficientnet", "vit", "yolo", "face recognition",
                   "pose estimation", "ocr", "optical flow",
                   "video understanding", "action recognition",
                   "instance segmentation", "semantic segmentation"],
    },
]

# Pipeline tags HuggingFace → domaine direct (lookup O(1) sans regex)
_PIPELINE_TO_DOMAIN: dict[str, str] = {
    # Generative
    "text-to-image":                    "Generative AI",
    "image-to-image":                   "Generative AI",
    "unconditional-image-generation":   "Generative AI",
    "text-to-video":                    "Generative AI",
    "text-to-3d":                       "3D / Point Cloud",
    "text-to-audio":                    "Audio / Speech",
    "image-to-video":                   "Generative AI",
    # Vision
    "image-classification":             "Computer Vision",
    "object-detection":                 "Computer Vision",
    "image-segmentation":               "Computer Vision",
    "depth-estimation":                 "Computer Vision",
    "image-feature-extraction":         "Computer Vision",
    "video-classification":             "Computer Vision",
    "zero-shot-image-classification":   "Computer Vision",
    # NLP
    "text-classification":              "NLP / Text",
    "token-classification":             "NLP / Text",
    "question-answering":               "NLP / Text",
    "text-generation":                  "NLP / Text",
    "text2text-generation":             "NLP / Text",
    "summarization":                    "NLP / Text",
    "translation":                      "NLP / Text",
    "fill-mask":                        "NLP / Text",
    "sentence-similarity":              "NLP / Text",
    "feature-extraction":               "NLP / Text",
    "zero-shot-classification":         "NLP / Text",
    # Audio
    "automatic-speech-recognition":     "Audio / Speech",
    "audio-classification":             "Audio / Speech",
    "text-to-speech":                   "Audio / Speech",
    "audio-to-audio":                   "Audio / Speech",
    "voice-activity-detection":         "Audio / Speech",
    # Multimodal
    "visual-question-answering":        "Multimodal",
    "document-question-answering":      "Multimodal",
    "image-text-to-text":               "Multimodal",
    "video-text-to-text":               "Multimodal",
    # RL
    "reinforcement-learning":           "Robotics / RL",
    "robotics":                         "Robotics / RL",
    # Tabular
    "tabular-classification":           "Finance",      # majorité finance/fraud
    "tabular-regression":               "Time Series",  # majorité forecasting
}


# ─────────────────────────────────────────────────────────────────────────────
# Détection de papers ArXiv / PDF
# ─────────────────────────────────────────────────────────────────────────────

_ARXIV_PATTERNS = [
    re.compile(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', re.I),
    re.compile(r'arxiv\.org/pdf/(\d{4}\.\d{4,5})', re.I),
    re.compile(r'\barxiv[:\s]+(\d{4}\.\d{4,5})', re.I),
    re.compile(r'arxiv:(\d{4}\.\d{4,5})', re.I),
]
_PDF_PATTERN = re.compile(
    r'https?://\S+\.pdf(?:\?\S*)?', re.I
)

def extract_paper_info(text: str, tags: list[str]) -> tuple[bool, Optional[str]]:
    """
    Détecte si un projet est lié à un paper académique.
    Cherche dans le texte libre ET dans les tags HuggingFace (format "arxiv:XXXX.XXXXX").

    Retourne (has_paper: bool, paper_url: str | None)
    """
    combined = text + " " + " ".join(tags)

    # 1 — tags HuggingFace formatés "arxiv:2301.12345"
    for tag in tags:
        t = tag.lower()
        if t.startswith("arxiv:"):
            arxiv_id = tag[6:].strip()
            if re.match(r'\d{4}\.\d{4,5}', arxiv_id):
                return True, f"https://arxiv.org/abs/{arxiv_id}"

    # 2 — URLs ArXiv dans le texte
    for pattern in _ARXIV_PATTERNS:
        m = pattern.search(combined)
        if m:
            arxiv_id = m.group(1)
            return True, f"https://arxiv.org/abs/{arxiv_id}"

    # 3 — PDF générique
    m = _PDF_PATTERN.search(combined)
    if m:
        return True, m.group(0)

    # 4 — mots-clés papier sans lien
    paper_signals = ["paper", "preprint", "published in", "proceedings",
                     "neurips", "icml", "iclr", "cvpr", "eccv", "iccv",
                     "acl", "emnlp", "naacl", "aaai", "ijcai"]
    text_lower = combined.lower()
    if any(kw in text_lower for kw in paper_signals):
        return True, None

    return False, None


def infer_domain(description: str, tags: list[str],
                 pipeline_tag: str = "") -> Optional[str]:
    """
    Infère le domaine principal d'un projet IA.

    Stratégie (ordre de priorité) :
      1. pipeline_tag HuggingFace → lookup direct O(1)
      2. Tags (topics GitHub / HF) → matching exact
      3. Description → substring matching
      4. "Other" si rien ne matche
    """
    # 1 — pipeline_tag direct
    if pipeline_tag and pipeline_tag in _PIPELINE_TO_DOMAIN:
        return _PIPELINE_TO_DOMAIN[pipeline_tag]

    tags_lower = {t.lower() for t in tags}
    desc_lower = description.lower()

    for rule in DOMAIN_RULES:
        domain = rule["domain"]
        # 2 — tags
        if any(t in tags_lower for t in rule["tags"]):
            return domain
        # 3 — description
        if any(kw in desc_lower for kw in rule["text"]):
            return domain

    return "Other"