from google import genai

# Using your project ID
PROJECT_ID = "project-67f295a8-2e8f-45e2-a81"

client = genai.Client(
    vertexai=True, 
    project=PROJECT_ID, 
    location="us-central1"
)
    
try:
    response = client.models.generate_content(
        # gemini-2.5-flash is the stable workhorse in 2026
        # gemini-3.1-flash-lite-preview is also an option for speed
        model="gemini-2.5-flash", 
        contents="Give me one tip for 3D printing with PETG."
    )
    print("\n--- AI Response ---")
    print(response.text)
except Exception as e:
    print(f"\n❌ Error: {e}")