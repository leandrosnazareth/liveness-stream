"""Explicit PAD contracts, deterministic preprocessing and reference validation."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


class PadContractError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PadContract:
    model_sha256: str
    input_name: str
    output_name: str
    classes: tuple
    color_order: str
    divisor: float
    crop_scale: float
    crop_mode: str
    output_kind: str
    reference_file: str
    reference_sha256: str
    reference_source: str

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(data) != set(cls.__dataclass_fields__):
            raise PadContractError("Campos do contrato PAD invalidos")
        data["classes"] = tuple(data["classes"])
        contract = cls(**data)
        contract.validate()
        return contract

    def validate(self):
        if len(self.classes) != 3 or set(self.classes) != {"live", "print", "replay"}:
            raise PadContractError("Classes PAD invalidas")
        if self.color_order not in {"BGR", "RGB"} or self.divisor not in {1.0, 255.0}:
            raise PadContractError("Normalizacao PAD invalida")
        if not isinstance(self.crop_scale, (int, float)) or not 1 <= self.crop_scale <= 5:
            raise PadContractError("Escala PAD invalida")
        if self.crop_mode not in {"upstream_rect", "square_clip"}:
            raise PadContractError("Recorte PAD invalido")
        if self.output_kind not in {"logits", "probabilities"}:
            raise PadContractError("Tipo de saida PAD invalido")
        for digest in (self.model_sha256, self.reference_sha256):
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise PadContractError("SHA256 PAD invalido")
        if not all(isinstance(v, str) and v.strip() for v in
                   (self.input_name, self.output_name, self.reference_file, self.reference_source)):
            raise PadContractError("Contrato PAD incompleto")


def preparar_entrada_pad(frame, bbox, contract):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise PadContractError("Frame PAD invalido")
    box = np.asarray(bbox, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise PadContractError("BBox PAD invalida")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise PadContractError("BBox PAD fora da imagem")
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if contract.crop_mode == "upstream_rect":
        scale = min(contract.crop_scale, (height - 1) / bh, (width - 1) / bw)
        left, top = cx - bw * scale / 2, cy - bh * scale / 2
        right, bottom = cx + bw * scale / 2, cy + bh * scale / 2
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > width - 1:
            left -= right - width + 1
            right = width - 1
        if bottom > height - 1:
            top -= bottom - height + 1
            bottom = height - 1
        # Match the upstream inclusive right/bottom crop convention.
        crop = frame[int(top):int(bottom) + 1, int(left):int(right) + 1]
    else:
        side = max(bw, bh) * contract.crop_scale
        crop = frame[int(max(0, cy - side / 2)):int(min(height, cy + side / 2)),
                     int(max(0, cx - side / 2)):int(min(width, cx + side / 2))]
    if not crop.size:
        raise PadContractError("Recorte PAD vazio")
    image = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_LINEAR)
    if contract.color_order == "RGB":
        image = image[:, :, ::-1]
    return np.ascontiguousarray((image.astype(np.float32) / contract.divisor).transpose(2, 0, 1)[None])


def decode_output(raw, contract):
    values = np.asarray(raw)
    if values.shape != (1, 3) or not np.isfinite(values).all():
        raise PadContractError("Saida PAD invalida")
    values = values[0].astype(np.float64)
    if contract.output_kind == "logits":
        values = np.exp(values - values.max())
        values /= values.sum()
    elif (values < 0).any() or (values > 1).any() or not np.isclose(values.sum(), 1, atol=1e-5):
        raise PadContractError("Probabilidades PAD invalidas")
    return {name: float(values[i]) for i, name in enumerate(contract.classes)}


class PadAdapter:
    def __init__(self, session, contract):
        contract.validate()
        self.session, self.contract = session, contract
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != contract.input_name
                or inputs[0].type != "tensor(float)" or len(inputs[0].shape) != 4
                or list(inputs[0].shape[1:]) != [3, 80, 80]
                or (isinstance(inputs[0].shape[0], int) and inputs[0].shape[0] != 1)):
            raise PadContractError("Entrada ONNX difere do contrato")
        output = next((o for o in outputs if o.name == contract.output_name), None)
        if (output is None or output.type != "tensor(float)" or len(output.shape) != 2
                or output.shape[1] != 3):
            raise PadContractError("Saida ONNX difere do contrato")

    def predict(self, frame, bbox):
        tensor = preparar_entrada_pad(frame, bbox, self.contract)
        raw = self.session.run([self.contract.output_name], {self.contract.input_name: tensor})[0]
        return decode_output(raw, self.contract)

    def validate_reference(self, contract_path):
        reference = Path(contract_path).parent / self.contract.reference_file
        if sha256(reference) != self.contract.reference_sha256:
            raise PadContractError("SHA256 da referencia PAD diverge")
        # Reference tensors must originate in the independently reviewed upstream pipeline.
        with np.load(reference, allow_pickle=False) as fixture:
            frames, boxes = fixture["frames"], fixture["boxes"]
            inputs, expected, labels = fixture["inputs"], fixture["probabilities"], fixture["labels"]
            count = len(frames)
            if (count < 3 or frames.ndim != 4 or frames.dtype != np.uint8
                    or boxes.shape != (count, 4) or inputs.shape != (count, 3, 80, 80)
                    or expected.shape != (count, 3) or labels.shape != (count,)
                    or set(labels.tolist()) != {"live", "print", "replay"}
                    or not np.isfinite(inputs).all() or not np.isfinite(expected).all()):
                raise PadContractError("Fixture PAD incompleta")
            for i in range(count):
                tensor = preparar_entrada_pad(frames[i], boxes[i], self.contract)
                if not np.allclose(tensor[0], inputs[i], atol=1e-6, rtol=1e-5):
                    raise PadContractError("Pre-processamento diverge da referencia")
                actual = self.predict(frames[i], boxes[i])
                canonical = np.array([actual[k] for k in ("live", "print", "replay")])
                if not np.allclose(canonical, expected[i], atol=1e-4, rtol=1e-4):
                    raise PadContractError("Inferencia diverge da referencia")
                if ("live", "print", "replay")[int(canonical.argmax())] != labels[i]:
                    raise PadContractError("Classe diverge do rotulo da referencia")
