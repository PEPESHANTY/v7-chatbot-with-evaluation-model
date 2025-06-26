import os
import boto3
from dotenv import load_dotenv

# --- Load .env variables ---
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET = os.getenv("S3_BUCKET_NAME")
CHUNK_ID_FILE = "last_chunk_id.txt"

# --- Setup S3 client with credentials ---
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

# --- Read current value from S3 ---
try:
    obj = s3.get_object(Bucket=BUCKET, Key=CHUNK_ID_FILE)
    current_id = int(obj["Body"].read().decode("utf-8").strip())
    print(f"📥 Current last_chunk_id.txt = {current_id}")
except Exception as e:
    print(f"❌ Error reading {CHUNK_ID_FILE}: {e}")
    current_id = None

# --- Set this ONLY if you want to update it ---
new_id = None  # 🔁 Replace with e.g. 24000 if you want to change it
#new_id = 23114  # 🔁 Replace with e.g. 24000 if you want to change it

if new_id is not None and current_id is not None:
    try:
        s3.put_object(Bucket=BUCKET, Key=CHUNK_ID_FILE, Body=str(new_id).encode("utf-8"))
        print(f"✅ Updated last_chunk_id.txt to: {new_id}")
    except Exception as e:
        print(f"❌ Failed to update chunk ID: {e}")
else:
    print(f"ℹ️ No changes made. Current last_chunk_id.txt = {current_id}")

