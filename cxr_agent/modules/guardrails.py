from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def moderate_content(text: str) -> dict:
    """
    Use OpenAI Moderation API to detect harmful or unsafe content.

    Returns:
        {"safe": True/False, "reason": "..."}
    """
    try:
        result = client.moderations.create(
            model="omni-moderation-latest",
            input=text,
        )

        flagged = result["results"][0]["flagged"]
        categories = result["results"][0]["categories"]

        if flagged:
            return {
                "safe": False,
                "reason": f"Flagged categories: {', '.join([k for k,v in categories.items() if v])}"
            }
        else:
            return {"safe": True, "reason": ""}
    except Exception as e:
        return {"safe": True, "reason": f"Moderation unavailable: {e}"}
