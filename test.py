import base64, hashlib, os, urllib.parse

code_verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()
code_challenge = (
    base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
    .rstrip(b"=")
    .decode()
)

state = f"test-user:0ff7cf3b-baf4-4ab1-9cd1-41d578d8cc87:{code_verifier}"

print("state:", urllib.parse.quote(state))
print("code_challenge:", code_challenge)
