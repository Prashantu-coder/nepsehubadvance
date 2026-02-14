from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

app = FastAPI(title="Symbol Data API with POI")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URL = "https://script.google.com/macros/s/AKfycbxteVvHon6igrKGV7KCyUO4m09tz9Q1FEG5nDv924zUPP2LARxmkQaX30yTPJrrFwItlg/exec"

# ---------- POI Detection Functions (adapted from previous answer) ----------
def calculate_indicators(df, volume_ma_period=20, atr_period=14):
    """Add necessary columns for POI detection."""
    df = df.copy()
    df['volume_ma'] = df['volume'].rolling(window=volume_ma_period).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Average True Range
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift()),
                                     abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(window=atr_period).mean()
    df['range'] = df['high'] - df['low']
    df['range_ratio'] = df['range'] / df['atr']
    
    # Body and wicks
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open','close']].max(axis=1)
    df['lower_wick'] = df[['open','close']].min(axis=1) - df['low']
    
    return df

def find_swing_points(df, order=5):
    highs_idx = argrelextrema(df['close'].values, np.greater, order=order)[0]
    lows_idx = argrelextrema(df['close'].values, np.less, order=order)[0]
    swing_highs = df.iloc[highs_idx][['close']].rename(columns={'close':'price'})
    swing_lows = df.iloc[lows_idx][['close']].rename(columns={'close':'price'})
    return swing_highs, swing_lows

def detect_poi_levels(df, volume_threshold=2.0, wick_threshold=0.6, range_threshold=0.5):
    """Main POI detection – returns list of POI dicts."""
    df = df.copy()
    pois = []
    
    # High Volume Nodes
    high_volume_bars = df[df['volume_ratio'] > volume_threshold]
    for idx, row in high_volume_bars.iterrows():
        pois.append({'price': row['low'], 'type': 'support', 'strength': row['volume_ratio'],
                     'description': 'High Volume Node (low)'})
        pois.append({'price': row['high'], 'type': 'resistance', 'strength': row['volume_ratio'],
                     'description': 'High Volume Node (high)'})
    
    # Reversal wicks
    bullish_rej = df[(df['lower_wick'] > wick_threshold * df['range']) & (df['volume_ratio'] > 1.5)]
    for idx, row in bullish_rej.iterrows():
        pois.append({'price': row['low'], 'type': 'support', 'strength': row['volume_ratio'],
                     'description': 'Bullish rejection wick'})
    
    bearish_rej = df[(df['upper_wick'] > wick_threshold * df['range']) & (df['volume_ratio'] > 1.5)]
    for idx, row in bearish_rej.iterrows():
        pois.append({'price': row['high'], 'type': 'resistance', 'strength': row['volume_ratio'],
                     'description': 'Bearish rejection wick'})
    
    # Absorption bars
    absorption = df[(df['range_ratio'] < range_threshold) & (df['volume_ratio'] > volume_threshold)]
    for idx, row in absorption.iterrows():
        pois.append({'price': row['low'], 'type': 'support', 'strength': row['volume_ratio'],
                     'description': 'Absorption bar (low)'})
        pois.append({'price': row['high'], 'type': 'resistance', 'strength': row['volume_ratio'],
                     'description': 'Absorption bar (high)'})
    
    # Breakout levels using swing points
    swing_highs, swing_lows = find_swing_points(df)
    for idx, row in swing_highs.iterrows():
        future = df.loc[idx:]
        breakout = future[future['close'] > row['price']]
        if not breakout.empty and breakout.iloc[0]['volume_ratio'] > volume_threshold:
            pois.append({'price': row['price'], 'type': 'support',
                         'strength': breakout.iloc[0]['volume_ratio'],
                         'description': 'Broken resistance (now support)'})
    for idx, row in swing_lows.iterrows():
        future = df.loc[idx:]
        breakout = future[future['close'] < row['price']]
        if not breakout.empty and breakout.iloc[0]['volume_ratio'] > volume_threshold:
            pois.append({'price': row['price'], 'type': 'resistance',
                         'strength': breakout.iloc[0]['volume_ratio'],
                         'description': 'Broken support (now resistance)'})
    
    # Climax patterns (simplified)
    for i in range(1, len(df)-1):
        prev, curr, nxt = df.iloc[i-1], df.iloc[i], df.iloc[i+1]
        if (prev['close'] > prev['open'] and prev['volume_ratio'] > 1.8 and
            curr['close'] < curr['open'] and curr['volume'] > prev['volume'] and
            curr['low'] < prev['low']):
            pois.append({'price': prev['high'], 'type': 'resistance',
                         'strength': curr['volume_ratio'],
                         'description': 'Buying climax'})
        if (prev['close'] < prev['open'] and prev['volume_ratio'] > 1.8 and
            curr['close'] > curr['open'] and curr['volume'] > prev['volume'] and
            curr['high'] > prev['high']):
            pois.append({'price': prev['low'], 'type': 'support',
                         'strength': curr['volume_ratio'],
                         'description': 'Selling climax'})
    
    # Convert to DataFrame for deduplication/clustering
    if pois:
        pois_df = pd.DataFrame(pois)
        pois_df['price'] = pois_df['price'].round(2)  # adjust rounding as needed
        pois_df = pois_df.sort_values('strength', ascending=False).drop_duplicates(subset=['price', 'type'])
        return pois_df.to_dict(orient='records')
    else:
        return []

# ---------- FastAPI Endpoint ----------
@app.get("/api/symbol-data")
async def get_symbol_data(symbol: str = Query(..., description="Company symbol")):
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

        raw_data = response.json()

        # Convert the raw data to a pandas DataFrame
        # Adjust the field names to match what the Google Script returns
        # Assuming it's a list of dicts with keys: date, open, high, low, close, volume
        df = pd.DataFrame(raw_data)
        # If date is a string, convert to datetime for sorting
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
        
        # Ensure numeric columns
        for col in ['open','high','low','close','volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['open','high','low','close','volume'], inplace=True)

        # Calculate indicators and detect POIs
        df_with_indicators = calculate_indicators(df)
        pois = detect_poi_levels(df_with_indicators)

        return {
            "success": True,
            "symbol": symbol,
            "data": raw_data,          # original data from Google Script
            "pois": pois                # detected points of interest
        }

    except httpx.RequestError:
        raise HTTPException(status_code=500, detail="External API request failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))