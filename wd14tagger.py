# Multi-model WD tagger (ensemble).
# Based on https://huggingface.co/spaces/SmilingWolf/wd-tagger
#
# Runs one or more WD taggers and merges their tags (max confidence per tag).
# All WD v1.4 / v2 / v3 taggers share the same ONNX preprocessing
# (pad-to-square, resize to the model's own input size, RGB->BGR, 0-255 float),
# so they can be mixed freely. Select models with PAINTS_UNDO_TAGGERS.

import os
import csv
import numpy as np
import onnxruntime as ort

from PIL import Image
from onnxruntime import InferenceSession

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hf_download', 'wd14')

# key -> (onnx_url, tags_csv_url)
TAGGER_MODELS = {
    'moat-v2': (
        'https://huggingface.co/lllyasviel/misc/resolve/main/wd-v1-4-moat-tagger-v2.onnx',
        'https://huggingface.co/lllyasviel/misc/resolve/main/wd-v1-4-moat-tagger-v2.csv',
    ),
    'vit-v3': (
        'https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/model.onnx',
        'https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/selected_tags.csv',
    ),
    'vit-large-v3': (
        'https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3/resolve/main/model.onnx',
        'https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3/resolve/main/selected_tags.csv',
    ),
    'swinv2-v3': (
        'https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.onnx',
        'https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/selected_tags.csv',
    ),
    'convnext-v3': (
        'https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3/resolve/main/model.onnx',
        'https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3/resolve/main/selected_tags.csv',
    ),
    'eva02-large-v3': (
        'https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.onnx',
        'https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv',
    ),
}

# Default ensemble: EVA02-Large (precise) + ViT-Large (broad recall).
# Override with e.g. PAINTS_UNDO_TAGGERS="eva02-large-v3" or "vit-v3,convnext-v3".
DEFAULT_TAGGERS = ['eva02-large-v3', 'vit-large-v3']

_loaded = {}  # key -> (InferenceSession, [(name, category), ...])


def _providers():
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def download_model(url, local_path):
    if os.path.exists(local_path):
        return local_path
    from torch.hub import download_url_to_file
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    temp_path = local_path + '.tmp'
    download_url_to_file(url=url, dst=temp_path)
    os.rename(temp_path, local_path)
    return local_path


def list_taggers():
    return list(TAGGER_MODELS.keys())


def resolve_models(models=None):
    from_env = False
    if models is None:
        env = os.environ.get('PAINTS_UNDO_TAGGERS', '').strip()
        from_env = bool(env)
        models = [m.strip() for m in env.split(',') if m.strip()] or list(DEFAULT_TAGGERS)
    unknown = [m for m in models if m not in TAGGER_MODELS]
    valid = [m for m in models if m in TAGGER_MODELS]
    if unknown:
        msg = f'[WARN] Ignoring unknown tagger(s) {unknown}. Options: {", ".join(TAGGER_MODELS)}'
        if from_env:
            # bad env var should never hard-crash the app — warn and continue
            print(msg)
        else:
            raise ValueError(msg)
    if not valid:
        print(f'[WARN] No valid taggers selected; using default {DEFAULT_TAGGERS}.')
        valid = list(DEFAULT_TAGGERS)
    return valid


def _load_tagger(key):
    if key in _loaded:
        return _loaded[key]
    onnx_url, csv_url = TAGGER_MODELS[key]
    onnx_path = download_model(onnx_url, os.path.join(MODEL_DIR, key, 'model.onnx'))
    csv_path = download_model(csv_url, os.path.join(MODEL_DIR, key, 'tags.csv'))
    try:
        session = InferenceSession(onnx_path, providers=_providers())
    except Exception:
        session = InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    tags = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # header: tag_id,name,category,count
        for row in reader:
            tags.append((row[1], row[2]))
    _loaded[key] = (session, tags)
    return _loaded[key]


def _preprocess(image, size):
    ratio = float(size) / max(image.size)
    new_size = tuple(int(x * ratio) for x in image.size)
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new('RGB', (size, size), (255, 255, 255))
    square.paste(image, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    arr = np.array(square).astype(np.float32)
    arr = arr[:, :, ::-1]  # RGB -> BGR
    return np.expand_dims(arr, 0)


def _run_one(key, pil_image):
    session, tags = _load_tagger(key)
    inp = session.get_inputs()[0]
    size = inp.shape[1]
    x = _preprocess(pil_image, size)
    out_name = session.get_outputs()[0].name
    probs = session.run([out_name], {inp.name: x})[0][0]
    return [(name, cat, float(p)) for (name, cat), p in zip(tags, probs)]


def default_interrogator(image, threshold=0.35, character_threshold=0.85,
                         exclude_tags="", models=None):
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert('RGB')

    models = resolve_models(models)

    merged = {}  # name -> (category, max_prob)
    for key in models:
        for name, cat, p in _run_one(key, image):
            prev = merged.get(name)
            if prev is None or p > prev[1]:
                merged[name] = (cat, p)

    general = [(n, p) for n, (c, p) in merged.items() if c == '0' and p > threshold]
    character = [(n, p) for n, (c, p) in merged.items() if c == '4' and p > character_threshold]
    character.sort(key=lambda t: t[1], reverse=True)
    general.sort(key=lambda t: t[1], reverse=True)
    selected = character + general

    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    selected = [t for t in selected if t[0] not in remove]

    res = ", ".join(
        item[0].replace("(", "\\(").replace(")", "\\)") for item in selected
    ).replace('_', ' ')
    return res
