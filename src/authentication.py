
import os
import time
import base64
from pathlib import Path

from dotenv import load_dotenv
from cryptography.hazmat.primitives import (
    serialization, hashes
)
from cryptography.hazmat.primitives.asymmetric import (
    padding, rsa, ed25519
)

load_dotenv()


def get_auth_headers(method, path):
    key_id = os.getenv("KALSHI_API_KEY_ID")
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")

    if not key_id or not key_path:
        raise ValueError("Missing Kalshi API credentials")

    private_key_path = Path(key_path)

    if not private_key_path.is_file():
        raise FileNotFoundError(
            f"Private key not found: {private_key_path}"
        )

    with private_key_path.open("rb") as file:
        private_key = serialization.load_pem_private_key(
            file.read(),
            password=None
        )

    timestamp = str(int(time.time() * 1000))

    # Don't include URL query parameters in the signature.
    path = path.split("?")[0]

    message = (
        timestamp + method.upper() + path
    ).encode("utf-8")

    if isinstance(private_key, rsa.RSAPrivateKey):
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )

    elif isinstance(private_key, ed25519.Ed25519PrivateKey):
        signature = private_key.sign(message)

    else:
        raise ValueError("Unsupported private key type")

    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": (
            base64.b64encode(signature).decode("utf-8")
        )
    }
