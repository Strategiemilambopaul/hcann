"""Scènes du banc épisodique : paires jumeaux + distracteurs (images synthétiques)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def noisy_copy(img: Image.Image, seed: int, amplitude: int = 6) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img).astype(np.int16)
    arr = arr + rng.integers(-amplitude, amplitude + 1, size=arr.shape, dtype=np.int16)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def stable_seed(eid: str, salt: int = 0) -> int:
    n = 0
    for i, c in enumerate(eid):
        n = (n * 131 + ord(c) * (i + 1) + salt) % 1_000_003
    return n


def _paint_shop(draw: ImageDraw.ImageDraw, wall, door) -> None:
    draw.rectangle([36, 48, 188, 210], fill=wall, outline=(40, 40, 40), width=4)
    draw.rectangle([70, 70, 110, 110], fill=(180, 220, 255), outline=(40, 40, 40), width=2)
    draw.rectangle([114, 70, 154, 110], fill=(180, 220, 255), outline=(40, 40, 40), width=2)
    draw.rectangle([92, 140, 132, 210], fill=door, outline=(40, 40, 40), width=3)


def _paint_cat(draw: ImageDraw.ImageDraw, fur) -> None:
    draw.ellipse([70, 80, 154, 170], fill=fur, outline=(30, 30, 30), width=2)
    draw.polygon([(78, 88), (70, 48), (100, 80)], fill=fur, outline=(30, 30, 30))
    draw.polygon([(146, 88), (154, 48), (124, 80)], fill=fur, outline=(30, 30, 30))
    draw.ellipse([92, 108, 108, 124], fill=(20, 20, 20))
    draw.ellipse([116, 108, 132, 124], fill=(20, 20, 20))


def _paint_box(draw: ImageDraw.ImageDraw, cardboard) -> None:
    draw.rectangle([56, 80, 168, 188], fill=cardboard, outline=(70, 50, 20), width=3)
    draw.line([(56, 80), (112, 52), (168, 80)], fill=(70, 50, 20), width=3)
    draw.line([(112, 52), (112, 188)], fill=(70, 50, 20), width=2)


def make_base(kind: str) -> Image.Image:
    img = Image.new("RGB", (224, 224), (236, 236, 232))
    draw = ImageDraw.Draw(img)
    if kind == "shop":
        _paint_shop(draw, (248, 248, 248), (90, 60, 40))
    elif kind == "cat":
        _paint_cat(draw, (200, 170, 130))
    elif kind == "box":
        _paint_box(draw, (196, 150, 90))
    else:
        raise ValueError(kind)
    return img


# Même entité pour tout le journal fondateur. L'isolation entre entités est une étape suivante.
TRACE_BY_ID: dict[str, dict] = {
    "shop_mardi_colis": {
        "entity_id": "paul",
        "place": "white shop",
        "participants": ["paul", "Mme X"],
        "context": "errand",
        "intent": "leave a package",
        "outcome": "package left on Tuesday",
    },
    "shop_jeudi_facture": {
        "entity_id": "paul",
        "place": "white shop",
        "participants": ["paul", "Mme X"],
        "context": "errand",
        "intent": "leave an invoice",
        "outcome": "invoice left on Thursday",
    },
    "vet_momo": {
        "entity_id": "paul",
        "place": "vet clinic",
        "participants": ["paul", "Momo"],
        "context": "checkup",
        "intent": "bring Momo to the vet",
        "outcome": "Momo checked",
    },
    "vet_luna": {
        "entity_id": "paul",
        "place": "vet clinic",
        "participants": ["paul", "Luna"],
        "context": "checkup",
        "intent": "bring Luna to the vet",
        "outcome": "Luna checked",
    },
    "box_entree": {
        "entity_id": "paul",
        "place": "building entrance",
        "participants": ["paul"],
        "context": "delivery",
        "intent": "receive a parcel",
        "outcome": "box left at the entrance",
    },
    "box_cour": {
        "entity_id": "paul",
        "place": "courtyard",
        "participants": ["paul"],
        "context": "delivery",
        "intent": "receive a parcel",
        "outcome": "box left in the courtyard",
    },
    "shop_retour_cle": {
        "entity_id": "paul",
        "place": "white shop",
        "participants": ["paul"],
        "context": "pickup",
        "intent": "pick up the keys",
        "outcome": "keys collected",
    },
    "shop_retour_sac": {
        "entity_id": "paul",
        "place": "white shop",
        "participants": ["paul"],
        "context": "pickup",
        "intent": "pick up the bag",
        "outcome": "bag collected",
    },
    "distract_blue": {
        "entity_id": "paul",
        "place": "office",
        "participants": ["paul"],
        "context": "meeting",
        "intent": "attend a meeting",
        "outcome": "blue circle noted",
    },
    "distract_red": {
        "entity_id": "paul",
        "place": "kitchen",
        "participants": ["paul"],
        "context": "home",
        "intent": "note the room",
        "outcome": "red square noted",
    },
    "distract_note": {
        "entity_id": "paul",
        "place": "desk",
        "participants": ["paul"],
        "context": "reminder",
        "intent": "write a reminder",
        "outcome": "sticky note written",
    },
    "distract_park": {
        "entity_id": "paul",
        "place": "park",
        "participants": ["paul"],
        "context": "walk",
        "intent": "walk outside",
        "outcome": "afternoon walk done",
    },
}


def trace_fields(ep: dict) -> dict:
    """Champs de trace pour `extra`. La légende (`text`) n'est pas réécrite."""
    fields = dict(TRACE_BY_ID.get(ep["id"], {"entity_id": "paul"}))
    if ep.get("identity"):
        fields["identity"] = ep["identity"]
    return fields


def twin_pairs() -> list[dict]:
    shop = make_base("shop")
    cat = make_base("cat")
    box = make_base("box")
    return [
        {
            "pair_id": "white_shop",
            "a": {
                "id": "shop_mardi_colis",
                "text": "Visit to the white shop. Left an item at the counter for Mme X. Package on Tuesday.",
                "created_at": "2026-03-10T11:00:00",
                "image": noisy_copy(shop, 11, 5),
                "identity": "package Tuesday",
            },
            "b": {
                "id": "shop_jeudi_facture",
                "text": "Visit to the white shop. Left an item at the counter for Mme X. Invoice on Thursday.",
                "created_at": "2026-03-12T11:00:00",
                "image": noisy_copy(shop, 12, 5),
                "identity": "invoice Thursday",
            },
            "cue_text": "Visit to the white shop. Left an item at the counter.",
        },
        {
            "pair_id": "vet_cat",
            "a": {
                "id": "vet_momo",
                "text": "Cat at the vet clinic sitting on the table after the checkup. This is Momo.",
                "created_at": "2026-04-02T09:30:00",
                "image": noisy_copy(cat, 21, 5),
                "identity": "Momo",
            },
            "b": {
                "id": "vet_luna",
                "text": "Cat at the vet clinic sitting on the table after the checkup. This is Luna.",
                "created_at": "2026-04-09T09:30:00",
                "image": noisy_copy(cat, 22, 5),
                "identity": "Luna",
            },
            "cue_text": "Cat at the vet clinic sitting on the table.",
        },
        {
            "pair_id": "delivery_box",
            "a": {
                "id": "box_entree",
                "text": "Cardboard box left at the building after delivery. Left at the entrance.",
                "created_at": "2026-05-01T16:00:00",
                "image": noisy_copy(box, 31, 5),
                "identity": "entrance",
            },
            "b": {
                "id": "box_cour",
                "text": "Cardboard box left at the building after delivery. Left in the courtyard.",
                "created_at": "2026-05-01T18:00:00",
                "image": noisy_copy(box, 32, 5),
                "identity": "courtyard",
            },
            "cue_text": "Cardboard box left at the building after delivery.",
        },
        {
            "pair_id": "shop_return",
            "a": {
                "id": "shop_retour_cle",
                "text": "Back at the white shop. Picked something up from the counter. The keys.",
                "created_at": "2026-03-18T10:00:00",
                "image": noisy_copy(shop, 41, 7),
                "identity": "keys",
            },
            "b": {
                "id": "shop_retour_sac",
                "text": "Back at the white shop. Picked something up from the counter. The bag.",
                "created_at": "2026-03-19T10:00:00",
                "image": noisy_copy(shop, 42, 7),
                "identity": "bag",
            },
            "cue_text": "Back at the white shop. Picked something up.",
        },
    ]


def distractors() -> list[dict]:
    img = Image.new("RGB", (224, 224), (220, 230, 245))
    draw = ImageDraw.Draw(img)
    draw.ellipse([60, 60, 164, 164], fill=(40, 90, 180), outline=(20, 20, 20), width=3)
    other = Image.new("RGB", (224, 224), (245, 220, 220))
    draw2 = ImageDraw.Draw(other)
    draw2.rectangle([50, 50, 174, 174], fill=(180, 40, 40), outline=(20, 20, 20), width=3)
    return [
        {
            "id": "distract_blue",
            "text": "Blue circle in the office during a meeting.",
            "created_at": "2026-01-01T08:00:00",
            "image": img,
        },
        {
            "id": "distract_red",
            "text": "Red square in the kitchen next to the window.",
            "created_at": "2026-01-02T08:00:00",
            "image": other,
        },
        {
            "id": "distract_note",
            "text": "Wrote a reminder on a yellow sticky note.",
            "created_at": "2026-01-03T08:00:00",
            "image": Image.new("RGB", (224, 224), (255, 240, 120)),
        },
        {
            "id": "distract_park",
            "text": "Walked through the park in the late afternoon.",
            "created_at": "2026-01-04T17:00:00",
            "image": Image.new("RGB", (224, 224), (80, 140, 70)),
        },
    ]


def degraded_of(target: dict) -> Image.Image:
    return noisy_copy(target["image"], stable_seed(target["id"], 48), 48)


def crop_of(img: Image.Image) -> Image.Image:
    """Moitié gauche de l'image stockée, ramenée à 224² — amorce partielle."""
    w, h = img.size
    return img.crop((0, 0, max(1, w // 2), h)).resize((224, 224))


def distinct_visits() -> list[dict]:
    """Quatre visites visuellement séparées. Attracteurs distincts — test CA3 honnête."""
    shop = make_base("shop")
    cat = make_base("cat")
    box = make_base("box")
    park = Image.new("RGB", (224, 224), (80, 140, 70))
    return [
        {
            "id": "shop_mardi_colis",
            "text": "Visit to the white shop. Left an item at the counter for Mme X. Package on Tuesday.",
            "partial_text": "Package on Tuesday.",
            "created_at": "2026-03-10T11:00:00",
            "image": noisy_copy(shop, 11, 5),
            "identity": "package Tuesday",
        },
        {
            "id": "vet_momo",
            "text": "Cat at the vet clinic sitting on the table after the checkup. This is Momo.",
            "partial_text": "This is Momo.",
            "created_at": "2026-04-02T09:30:00",
            "image": noisy_copy(cat, 21, 5),
            "identity": "Momo",
        },
        {
            "id": "box_entree",
            "text": "Cardboard box left at the building after delivery. Left at the entrance.",
            "partial_text": "Left at the entrance.",
            "created_at": "2026-05-01T16:00:00",
            "image": noisy_copy(box, 31, 5),
            "identity": "entrance",
        },
        {
            "id": "distract_park",
            "text": "Walked through the park in the late afternoon.",
            "partial_text": "late afternoon.",
            "created_at": "2026-01-04T17:00:00",
            "image": park,
            "identity": "park",
        },
    ]


def completion_queries(ep: dict) -> list[tuple[str, str | None, Image.Image | None]]:
    return [
        ("partial_text", ep["partial_text"], None),
        ("image_crop", None, crop_of(ep["image"])),
        ("noisy_image", None, degraded_of(ep)),
    ]
