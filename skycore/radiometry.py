"""Radiometry helpers of the PHYSICS observation model: verbatim copies of helpers of the camera's spectral response."""

from __future__ import annotations

import hashlib

import io

import json

from pathlib import Path

import numpy as np

class RadiometryError(ValueError):
    """A declaration this module refuses, naming the field."""

    def __init__(self, message, *, key=None):
        super().__init__(message)
        self.key = key

def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def declared_digest(declaration) -> str:
    """sha256 over the DECLARED block alone (inputs and provenance), never the derived block."""
    payload = {k: v for k, v in declaration.items() if k not in ("derived", "declaration_sha256")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
