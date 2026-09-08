import os
import sys
import requests
from datetime import datetime
from fastapi import FastAPI, status, HTTPException, Security, Depends, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader

# Importy do Surprise i analizy danych
import pandas as pd
import numpy as np
from surprise import KNNBasic, SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# 1. Pobranie klucza z pamięci RAM kontenera
MVT_API_KEY = os.getenv("MVT_API_KEY")

# Bezpieczeństwo przede wszystkim: jeśli klucza brak, wyłączamy aplikację
if not MVT_API_KEY:
    print("BŁĄD KRYTYCZNY: Zmienna środowiskowa MVT_API_KEY nie została ustawiona!")
    sys.exit(1)

app = FastAPI(title="MVT Recommendation API")

# Definiujemy, że klucz ma być przekazywany w nagłówku HTTP o nazwie X-API-Key
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Funkcja weryfikująca poprawność klucza
async def verify_api_key(header_value: str = Security(api_key_header)):
    if header_value == MVT_API_KEY:
        return header_value
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Brak lub niepoprawny klucz API (X-API-Key)"
    )

# --- KLASA DO MODELI REKOMENDACJI ---

class RecommendationModelManager:
    """Menedżer modeli rekomendacji z Surprise"""
    
    def __init__(self):
        self.cf_model = None  # Collaborative Filtering model
        self.cb_model = None  # Content-Based Filtering model
        # Cache w pamięci (szkielet) - klucz: "recommendations_{user_id}"
        self._cache: dict = {}

    def build_collaborative_filtering_model(self, trainset):
        """Buduje model collaborative filtering (KNNBasic + SVD)"""
        try:
            knn = KNNBasic(
                min_k=20,
                n_factors=50,
                n_epochs=20,
                random_state=42
            )
            knn.fit(trainset)
            self.cf_model = knn
            print("Model collaborative filtering wybudowany")
        except Exception as e:
            print(f"Błąd podczas budowania modelu CF: {e}")
    
    def build_content_based_filtering_model(self, trainset):
        """Buduje model content-based (SVD)"""
        try:
            svd = SVD(
                n_factors=50,
                n_epochs=20,
                random_state=42
            )
            svd.fit(trainset)
            self.cb_model = svd
            print("Model content-based wybudowany")
        except Exception as e:
            print(f"Błąd podczas budowania modelu CB: {e}")

    # --- CACHE TYMCZASOWY (DO ZMIANY!!!) ---
    #
    # UWAGA: poniższe metody są TYLKO szkieletem (placeholder).
    # Backend nie ma jeszcze gotowej tabeli do zapisu rekomendacji (frontend
    # nie jest jeszcze po stronie gotowy). Zamiast pisać do bazy, cache jest
    # przechowywany tymczasowo w pamięci RAM (dict) z prostym expires-at.
    #
    # GDD: gdyby istniał endpoint do zapisu, miałby być albo pusty, albo zakomentowany.
    # Tutaj zapis w ogóle nie jest wywoływany (zob. endpoint get_recommendations).

    def get_cached_recommendation(self, cache_key):
        """SZKIELET odczytu z cache (pamięć RAM). Zwraca listę lub None."""
        if cache_key not in self._cache:
            return None
        entry = self._cache[cache_key]
        if entry is None:
            return None
        # proste wygaszanie (szkielet)
        expires_at = entry.get("expires_at")
        if expires_at is not None and datetime.now().timestamp() > expires_at:
            self._cache[cache_key] = None
            return None
        return entry.get("recommendations") if entry else None

    def save_cached_recommendation(self, user_id, recommendations, cache_key):
        """SZKIELET zapisu do cache (pamięć RAM).
        NIE zapisuje jeszcze do bazy (brak gotowej tabeli po stronie frontendu).
        """
        expires_at = datetime.now().timestamp() + CACHE_EXPIRATION_MINUTES * 60
        self._cache[cache_key] = {
            "user_id": user_id,
            "recommendations": recommendations,
            "expires_at": expires_at,
        }

# --- KLASA DO POŁĄCZENIA Z CLOUDFLARE D1 ---

class CloudflareDBConnection:
    """Klasa do interfejsu z bazą danych w Cloudflare D1"""

    def __init__(self):
        self.account_id = os.getenv("ACCOUNT_ID")
        self.database_id = os.getenv("DATABASE_ID")
        self.api_token = os.getenv("API_TOKEN")
        
        # Budujemy URL do REST API Cloudflare
        self.endpoint_url = "https://api.cloudflare.com/client/v4/accounts"

    def fetch_user_ratings(self, user_id: int) -> pd.DataFrame | None:
        """
        Pobieramy oceny użytkownika z bazy Cloudflare D1.
        RETURN: DataFrame z kolumnami (rating, movie_id) lub None w razie błędu
        """
        try:
            if not self.account_id or not self.database_id or not self.api_token:
                print("Brak połączenia z Cloudflare - brak kluczy środowiskowych")
                return None
            
            # Budujemy URL do REST API
            endpoint = f"{self.endpoint_url}/{self.account_id}/databases/{self.database_id}/query"
            
            # Zapytanie SQL - pobieramy rating i movie_id z reviews dla użytkownika
            query = """
                SELECT r.rating, r.movie_id
                FROM reviews r
                WHERE r.user_id = ?
                ORDER BY r.created_at DESC
                LIMIT 100
            """
            
            # Parametr SQL (bez iniekcji!)
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            payload = {"query": query}
            
            response = requests.post(endpoint, json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json().get("result", [])
                
                # Konwertujemy JSON do DataFrame z rating i movie_id
                return pd.DataFrame(data, columns=["rating", "movie_id"])
            else:
                print(f"Błąd API Cloudflare (status: {response.status_code})")
                return None
                
        except Exception as e:
            print(f"Błąd podczas pobierania danych użytkownika z Cloudflare: {e}")
            return None

# Inicjalizacja menedżera modeli i bazy Cloudflare
model_manager = RecommendationModelManager()
cloudflare_db = CloudflareDBConnection()

# Endpoint publiczny (np. dla Cloudflare do sprawdzania czy kontener żywe)
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

CACHE_EXPIRATION_MINUTES = 30



@app.get("/movies/recommendations", dependencies=[Depends(verify_api_key)])
async def get_recommendations(user_id: int):
    """Endpoint rekomendacji filmów z Surprise (Content-Based + Collaborative Filtering)"""
    
    # 1. Ładujemy dane użytkownika z Cloudflare D1
    user_ratings_df = cloudflare_db.fetch_user_ratings(user_id)
    
    if user_ratings_df is None:
        return {
            "status": "error",
            "message": "Nie można pobrać danych użytkownika z bazy Cloudflare"
        }
    
    # 2. Przygotujemy Surprise dataset i trenujmy modele
    dataset = Dataset.from_df(
        user_ratings_df,
        id_columns=['movie_id'],
        rcolumns=['rating']
    )
    
    try:
        # Podziel dane na trening/test
        trainset, testset = train_test_split(
            dataset, 
            test_size=0.25,       # 25% do testów
            allow_same_user=True   # Pozwala na tego samego użytkownika w training/test
        )
        
        # Budujmy modele (tylko jeśli jeszcze nie istnieją)
        if not model_manager.cf_model:
            model_manager.build_collaborative_filtering_model(trainset)
        if not model_manager.cb_model:
            model_manager.build_content_based_filtering_model(trainset)
            
    except Exception as e:
        print(f"Błąd podczas trenowania modeli: {e}")
    
    # 3. Generujemy rekomendacje SVD
    recommendations = []
    if model_manager.cf_model and user_id in model_manager.cf_model.get_all_users():
        try:
            # Pobierzmy podobne filmy dla użytkownika
            similar_items = model_manager.cf_model.similar_items(user_id)
            
            for movie_id, data in similar_items:
                try:
                    prediction = model_manager.cf_model.predict(user_id, movie_id)
                    
                    # Sprawdź czy użytkownik nie ocenił już tego filmu
                    if user_id not in data or str(data[user_id]) != "nan":
                        recommendations.append({
                            "movie_id": movie_id,
                            "rating": round(prediction.estimation, 2),
                            "confidence_interval": list(prediction.confidence_interval)
                        })
                except:
                    continue
            
            # Sortuj według przewidywanych ocen (najwyższe pierwsze)
            recommendations.sort(key=lambda x: x["rating"], reverse=True)
            
        except Exception as e:
            print(f"✗ Błąd podczas generowania rekomendacji: {e}")
    
    # 4. Zwróćmy wyniki z cache jeśli aktualne
    cache_key = f"recommendations_{user_id}"
    cached_recommendations = model_manager.get_cached_recommendation(cache_key)
    
    if cached_recommendations and len(cached_recommendations) > 0:
        return {
            "status": "success",
            "source": "cache",
            "user_id": user_id,
            "type": "collaborative_filtering",
            "count": len(cached_recommendations),
            "recommendations": cached_recommendations
        }
    
    # 5. Zapiszmy cache jeśli mamy rekomendacje
    if recommendations:
        model_manager.save_cached_recommendation(
            user_id=user_id, 
            recommendations=recommendations[:10],
            cache_key=cache_key
        )
    
    return {
        "status": "success",
        "source": "computed",
        "user_id": user_id,
        "type": "collaborative_filtering",
        "count": len(recommendations),
        "recommendations": recommendations[:10]  # Zwróćmy max 10 rekomendacji
    }


@app.get("/movies/recommendations/force-recalculate")
async def force_recommendation_calculation(user_id: int):
    """Endpoint do wymuszenia ponownego obliczania (bez cache)"""
    
    # 1. Ładujemy dane użytkownika z Cloudflare D1
    user_ratings_df = cloudflare_db.fetch_user_ratings(user_id)
    
    if user_ratings_df is None:
        return {
            "status": "error",
            "message": "Nie można pobrać danych użytkownika z bazy Cloudflare"
        }
    
    # 2. Przygotujmy Surprise dataset
    dataset = Dataset.from_df(
        user_ratings_df,
        id_columns=['movie_id'],
        rcolumns=['rating']
    )
    
    try:
        # Podziel dane na trening/test
        trainset, _ = train_test_split(
            dataset, 
            test_size=0.25,
            allow_same_user=True
        )
        
        # Trenujmy modele od nowa
        model_manager.build_collaborative_filtering_model(trainset)
        model_manager.build_content_based_filtering_model(trainset)
        
    except Exception as e:
        print(f"✗ Błąd podczas trenowania modeli (force): {e}")
        return {
            "status": "error",
            "message": f"Błąd podczas trenowania modeli: {e}"
        }
    
    # 3. Generujemy rekomendacje bez cache
    recommendations = []
    if model_manager.cf_model and user_id in model_manager.cf_model.get_all_users():
        try:
            similar_items = model_manager.cf_model.similar_items(user_id)
            
            for movie_id, data in similar_items:
                try:
                    prediction = model_manager.cf_model.predict(user_id, movie_id)
                    
                    if user_id not in data or str(data[user_id]) != "nan":
                        recommendations.append({
                            "movie_id": movie_id,
                            "rating": round(prediction.estimation, 2),
                            "confidence_interval": list(prediction.confidence_interval)
                        })
                except:
                    continue
            
            recommendations.sort(key=lambda x: x["rating"], reverse=True)
            
        except Exception as e:
            print(f"✗ Błąd podczas generowania rekomendacji (force): {e}")
    
    # 4. Zwróćmy wyniki
    return {
        "status": "success",
        "source": "forced-recalculation",
        "user_id": user_id,
        "type": "collaborative_filtering",
        "count": len(recommendations),
        "recommendations": recommendations[:10]
    }


# --- ENDPOINTY DO REKOMENDACJI DLA FRONTENDU ---

@app.get("/movies/user/{user_id}/recommendations", dependencies=[Depends(verify_api_key)])
async def get_user_recommendations(user_id: int):
    """
    Endpoint pobierania rekomendacji filmów dla użytkownika.
    Zwraca JSON z listą rekomendowanych filmów.
    Frontend woła ten endpoint, aby pokazać rekomendacje na dashboardzie.

    Uwaga: to jest to samo logiczne zapytanie co /movies/recommendations,
    tylko z innym pathem dla wygody frontendu.
    """
    return await get_recommendations(user_id)