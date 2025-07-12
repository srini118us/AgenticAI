# test_email_config.py
import os
from dotenv import load_dotenv

load_dotenv()

print("=== EMAIL CONFIG TEST ===")
print(f"EMAIL_ENABLED: {os.getenv('EMAIL_ENABLED')}")
print(f"GMAIL_EMAIL: {os.getenv('GMAIL_EMAIL')}")
print(f"GMAIL_APP_PASSWORD: {os.getenv('GMAIL_APP_PASSWORD')}")

# Test email enabled check
email_enabled = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
gmail_email = os.getenv("GMAIL_EMAIL")
gmail_password = os.getenv("GMAIL_APP_PASSWORD")

print(f"\nParsed values:")
print(f"email_enabled: {email_enabled}")
print(f"gmail_email exists: {bool(gmail_email)}")
print(f"gmail_password exists: {bool(gmail_password)}")
print(f"All good: {email_enabled and gmail_email and gmail_password}")