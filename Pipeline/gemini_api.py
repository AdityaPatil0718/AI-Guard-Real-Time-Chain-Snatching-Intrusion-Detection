import traceback
from PIL import Image
import google.generativeai as genai

class GeminiAPI:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash-latest"):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"[✓] Gemini model initialized: {model_name}")

    def generate_analysis_from_image(self, image_path: str, user_query: str) -> str:
        try:
            image = Image.open(image_path)
        except Exception as e:
            return f"ERROR: Could not open image {image_path}: {e}"

        prompt_parts = [
            "You are analyzing a CCTV frame for suspicious or theft-related activity.",
            f"**User Question:** {user_query}",
            "**Instructions:** Answer succinctly yes/no and explain what you see."
        ]

        try:
            response = self.model.generate_content([image] + prompt_parts)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                return f"ERROR: Blocked by policy: {response.prompt_feedback.block_reason}"
            return response.text.strip()
        except Exception as e:
            traceback.print_exc()
            return f"ERROR: Failed to generate analysis: {type(e).__name__}: {e}"

# if __name__ == "__main__":
#     # 🔒 Directly set your API key here
#     api_key = "AIzaSyA_YUyQB6HXSHPWDNpxAFJ2ZxvJwrLUj4U"  # <--- Replace with your actual Gemini API key

#     # 🔍 Image and query
#     image_path = "best_frame_snatching_confidence_0.9429.jpg"
#     user_question = ("Is there any suspicious or theft-related activity in this image?",
#                     "Describe the person performing it — clothing color (upper and lower), helmet/headgear, etc")

#     # ⚙️ Run Gemini analysis
#     gemini = GeminiAPI(api_key)
#     result = gemini.generate_analysis_from_image(image_path, user_question)
#     print("\n=== Gemini Analysis ===")
#     print(result)
