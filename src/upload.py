import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError

def upload_to_r2(file_name, bucket_name, object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_name)

    # --- Manual input for R2 credentials in Colab ---
    # You will need to replace these placeholders with your actual credentials.
    # For better security, consider using Colab's secret manager or environment variables.
    endpoint_url = os.environ.get("R2_ENDPOINT_URL") # e.g., "https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
    aws_access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")

    if not all([endpoint_url, aws_access_key_id, aws_secret_access_key]):
        print("[FEHLER] Cloudflare R2 credentials are not set as environment variables. Upload skipped.")
        print("Please set R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY as environment variables.")
        return False
    # --- End Manual Input ---

    s3_client = boto3.client(
        service_name='s3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name='auto'
    )

    print(f"-> Starte Upload von '{file_name}' nach R2 Bucket '{bucket_name}' als '{object_name}'...")
    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
        print(f"   [SUCCESS] Upload von '{file_name}' erfolgreich.")
    except FileNotFoundError:
        print(f"   [FEHLER] Die Datei '{file_name}' wurde nicht gefunden.")
        return False
    except NoCredentialsError:
        print("   [FEHLER] Credentials nicht gefunden. Bitte überprüfe die manuell eingegebenen Secrets.")
        return False
    except ClientError as e:
        print(f"   [FEHLER] Ein Client-Fehler ist aufgetreten: {e}")
        return False
    return True

def run_upload():
    print("\n--- Starte den Upload-Prozess zu Cloudflare R2 ---")

    R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")

    if not R2_BUCKET_NAME:
        print("[FEHLER] R2_BUCKET_NAME environment variable is not set. Upload skipped.")
    else:
        files_to_upload = {
            "eisbach_predictions.csv": "eisbach_predictions.csv",
            "eisbach_plot.html": "eisbach_plot.html",
            # "eisbach_new.png": "eisbach_new.png" # PNG not generated in new version yet, but could be added if needed
        }
        for local_path, cloud_object_name in files_to_upload.items():
            if os.path.exists(local_path):
                upload_to_r2(file_name=local_path, bucket_name=R2_BUCKET_NAME, object_name=cloud_object_name)
            else:
                print(f"[WARNUNG] Lokale Datei '{local_path}' nicht gefunden. Upload übersprungen.")

    print("\n--- Upload-Prozess abgeschlossen. ---")
