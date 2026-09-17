"""Validate an ONNX artifact against its independently generated reference fixture.

Usage: python -m liveness_app.validate_pad /path/to/model.onnx
Requires onnxruntime (CPU or GPU) and the regular capture preprocessing dependencies.
"""
import argparse

from liveness_app.core.pad import PadAdapter, PadContract, PadContractError, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    args = parser.parse_args()
    import onnxruntime as ort
    contract_path = args.model + ".json"
    contract = PadContract.load(contract_path)
    if sha256(args.model) != contract.model_sha256:
        raise PadContractError("SHA256 do modelo PAD diverge")
    adapter = PadAdapter(ort.InferenceSession(args.model, providers=["CPUExecutionProvider"]), contract)
    adapter.validate_reference(contract_path)
    print("Contrato e referencia PAD verificados. Isto nao certifica desempenho contra ataques.")


if __name__ == "__main__":
    main()
