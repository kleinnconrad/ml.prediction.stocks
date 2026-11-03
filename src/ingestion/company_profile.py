import os
import json
import logging
from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env variables (including GEMINI_API_KEY)
load_dotenv()

CACHE_FILE = os.path.join('data', 'processed', 'company_profiles_cache.json')

# Descriptions returned when no profile could be generated; never cached
FALLBACK_DESCRIPTIONS = (
    "No description available.",
    "Error fetching description.",
    "Failed to parse description.",
)

class CompanyProfile(BaseModel):
    full_name: str
    description: str

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load company profile cache: {e}")
    return {}

def save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save company profile cache: {e}")

def fetch_company_profile(ticker: str, company_name: str, model: str) -> dict:
    """
    Fetches a brief description using Gemini given the ticker and known company name.
    Utilizes a local cache to avoid redundant API calls.

    Args:
        ticker (str): Ticker symbol.
        company_name (str): Known company name.
        model (str): Gemini model id (`gemini_model` in config/settings.yaml).

    Returns:
        dict: {"full_name": str, "description": str}
    """
    cache = load_cache()
    if ticker in cache:
        return cache[ticker]
        
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        logger.warning(f"GEMINI_API_KEY not found. Cannot fetch profile for {ticker}.")
        return {"full_name": company_name, "description": "No description available."}
        
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"Provide a 1-sentence brief description for the German stock '{company_name}' (Ticker symbol '{ticker}')."
        
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': CompanyProfile,
            },
        )
        
        try:
            profile_data = json.loads(response.text)
            profile_data['full_name'] = company_name # Enforce the known correct name
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Gemini response for {ticker}: {response.text}")
            return {"full_name": company_name, "description": "Failed to parse description."}
            
        # Cache and save
        cache[ticker] = profile_data
        save_cache(cache)
        
        return profile_data
        
    except Exception as e:
        logger.error(f"Error fetching company profile for {ticker} from Gemini: {e}")
        return {"full_name": company_name, "description": "Error fetching description."}
