/**
 * AES-256-GCM encryption for provider API keys.
 * Master key stored via `wrangler secret put PROVIDER_ENCRYPTION_KEY`.
 */

const ALGO = "AES-GCM";
const IV_BYTES = 12; // 96-bit IV for GCM

function hexToKeyMaterial(hex: string): Promise<CryptoKey> {
  const bytes = new Uint8Array(
    hex.match(/.{2}/g)!.map((b) => parseInt(b, 16))
  );
  return crypto.subtle.importKey("raw", bytes, ALGO, false, [
    "encrypt",
    "decrypt"
  ]);
}

export async function encryptApiKey(
  plaintext: string,
  masterKeyHex: string
): Promise<{ ciphertext: string; iv: string }> {
  const key = await hexToKeyMaterial(masterKeyHex);
  const iv = crypto.getRandomValues(new Uint8Array(IV_BYTES));
  const encoded = new TextEncoder().encode(plaintext);
  const encrypted = await crypto.subtle.encrypt(
    { name: ALGO, iv },
    key,
    encoded
  );
  return {
    ciphertext: bufToHex(encrypted),
    iv: bufToHex(iv.buffer)
  };
}

export async function decryptApiKey(
  ciphertext: string,
  iv: string,
  masterKeyHex: string
): Promise<string> {
  const key = await hexToKeyMaterial(masterKeyHex);
  const ctBytes = hexToBuf(ciphertext);
  const ivBytes = hexToBuf(iv);
  const decrypted = await crypto.subtle.decrypt(
    { name: ALGO, iv: ivBytes },
    key,
    ctBytes
  );
  return new TextDecoder().decode(decrypted);
}

function bufToHex(buf: ArrayBuffer): string {
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function hexToBuf(hex: string): ArrayBuffer {
  return new Uint8Array(hex.match(/.{2}/g)!.map((b) => parseInt(b, 16))).buffer as ArrayBuffer;
}
