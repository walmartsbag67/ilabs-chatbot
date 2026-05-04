import google.auth

try:
    # This function looks for the "Application Default Credentials" you just set up
    credentials, project = google.auth.default()
    
    print("✅ Success! Google Cloud is connected.")
    print(f"Project ID: {project}")
    print(f"Credential Type: {type(credentials).__name__}")

except Exception as e:
    print("❌ Connection failed.")
    print(f"Error: {e}")
