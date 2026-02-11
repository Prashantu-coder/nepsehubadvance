from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI(title="Symbol Data API")

# Enable CORS for all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base Google Script URL
BASE_URL = "https://script.google.com/macros/s/AKfycbytdROXOtfIeISL0NFbH6obQJX_Hugvn6nPOkpjQdJbdzK1SJjXuDK5Q6nj2lrFeX_9/exec"


@app.get("/api/symbol-data")
async def get_symbol_data(symbol: str = Query(..., description="Company symbol")):
    """
    Fetch data from Google Apps Script using symbol from frontend
    """

    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(BASE_URL, params={"symbol": symbol}, follow_redirects=True)

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Error fetching data from external API"
            )

        data = response.json()

        return {
            "success": True,
            "symbol": symbol,
            "data": data
        }

    except httpx.RequestError:
        raise HTTPException(status_code=500, detail="External API request failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
