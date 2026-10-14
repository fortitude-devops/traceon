"""
Проверка владения Bitcoin-адресом по подписанному сообщению (стандарт BIP-137,
«Sign message» в Electrum, Sparrow, Trezor, Ledger, Bitcoin Core).

Клиент подписывает текст приватным ключом своего кошелька. Мы восстанавливаем из подписи
публичный ключ, вычисляем из него адрес и сравниваем с заявленным. Совпало — клиент
действительно владеет адресом. Сам приватный ключ при этом никто не видит.

Поддерживаются адреса: 1... (P2PKH), 3... (P2SH-P2WPKH), bc1q... (P2WPKH), а также их тестовые сети.
"""
from __future__ import annotations

import base64
import binascii
import hashlib

from ecdsa import SECP256k1, VerifyingKey
from ecdsa.util import sigdecode_string

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BECH32 = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
PREFIX = b"\x18Bitcoin Signed Message:\n"


def sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def ripemd160(b: bytes) -> bytes:
    try:
        return hashlib.new("ripemd160", b).digest()
    except ValueError:  # в OpenSSL 3 RIPEMD-160 может быть отключён
        from Crypto.Hash import RIPEMD160
        return RIPEMD160.new(b).digest()


def hash160(b: bytes) -> bytes:
    return ripemd160(hashlib.sha256(b).digest())


def b58check(payload: bytes) -> str:
    data = payload + sha256d(payload)[:4]
    n, s = int.from_bytes(data, "big"), ""
    while n:
        n, r = divmod(n, 58)
        s = B58[r] + s
    return "1" * (len(data) - len(data.lstrip(b"\0"))) + s


def _polymod(values) -> int:
    gen = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)
    chk = 1
    for v in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            chk ^= gen[i] if (top >> i) & 1 else 0
    return chk


def _convertbits(data: bytes, frombits=8, tobits=5) -> list[int]:
    acc = bits = 0
    out, maxv = [], (1 << tobits) - 1
    for value in data:
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            out.append((acc >> bits) & maxv)
    if bits:
        out.append((acc << (tobits - bits)) & maxv)
    return out


def bech32_v0(hrp: str, program: bytes) -> str:
    data = [0] + _convertbits(program)
    expand = [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]
    pm = _polymod(expand + data + [0] * 6) ^ 1
    checksum = [(pm >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(BECH32[d] for d in data + checksum)


def _varint(n: int) -> bytes:
    if n < 0xFD:
        return bytes([n])
    if n <= 0xFFFF:
        return b"\xfd" + n.to_bytes(2, "little")
    if n <= 0xFFFFFFFF:
        return b"\xfe" + n.to_bytes(4, "little")
    return b"\xff" + n.to_bytes(8, "little")


def message_digest(message: str) -> bytes:
    m = message.encode("utf-8")
    return sha256d(PREFIX + _varint(len(m)) + m)


def candidate_addresses(vk: VerifyingKey) -> dict[str, str]:
    """Все стандартные адреса, которые даёт этот публичный ключ."""
    out = {}
    for net, (v_pkh, v_sh, hrp) in {"основная сеть": (0x00, 0x05, "bc"), "тестовая сеть": (0x6F, 0xC4, "tb")}.items():
        h_unc = hash160(vk.to_string("uncompressed"))
        h_cmp = hash160(vk.to_string("compressed"))
        out[b58check(bytes([v_pkh]) + h_unc)] = f"P2PKH, {net}, несжатый ключ"
        out[b58check(bytes([v_pkh]) + h_cmp)] = f"P2PKH, {net}"
        out[b58check(bytes([v_sh]) + hash160(b"\x00\x14" + h_cmp))] = f"P2SH-P2WPKH, {net}"
        out[bech32_v0(hrp, h_cmp)] = f"P2WPKH, {net}"
    return out


def verify_message(address: str, message: str, signature: str) -> dict:
    address, signature = (address or "").strip(), (signature or "").strip()
    if not address or not message or not signature:
        return {"valid": False, "error": "Укажите адрес, текст сообщения и подпись."}
    try:
        sig = base64.b64decode(signature, validate=True)
    except (binascii.Error, ValueError):
        return {"valid": False, "error": "Подпись должна быть в формате Base64."}
    if len(sig) != 65:
        return {"valid": False, "error": f"Неверная длина подписи: {len(sig)} байт вместо 65."}
    if not 27 <= sig[0] <= 42:
        return {"valid": False, "error": "Неизвестный формат подписи (заголовок вне диапазона BIP-137)."}

    digest = message_digest(message)
    try:
        keys = VerifyingKey.from_public_key_recovery_with_digest(
            sig[1:], digest, curve=SECP256k1, hashfunc=hashlib.sha256, sigdecode=sigdecode_string)
    except Exception:
        return {"valid": False, "error": "Не удалось восстановить ключ из подписи."}

    target = address.lower() if address.lower().startswith(("bc1", "tb1")) else address
    for vk in keys:
        for candidate, kind in candidate_addresses(vk).items():
            if candidate == target:
                return {"valid": True, "address_type": kind,
                        "detail": "Подпись верна: клиент владеет этим адресом."}
    return {"valid": False, "error": "Подпись не соответствует адресу: сообщение подписано другим ключом "
                                      "или текст сообщения изменён."}
