"""Authentication: pluggable sign-in providers, signed session tokens and the request principal.

passwords.py     scrypt hashing for the local provider
tokens.py        HMAC-signed session tokens (issued by the API, provider-independent)
providers.py     Identity, password providers, complete_sign_in (identity -> User)
dependencies.py  get_principal: X-API-Key (services) or Bearer token (users)
"""
