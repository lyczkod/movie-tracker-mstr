import os
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from surprise import Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity

from services.model_manager import RecommendationModelManager, tracker
from database.db_manager import DatabaseManager

router = APIRouter(prefix="/movies/user", tags=["Recommendations"])
model_manager = RecommendationModelManager()
db = DatabaseManager()

API_KEY = os.getenv("MVT_API_KEY")
CACHE_MINUTES = 7 * 24 * 60  # 7 dni w minutach
REC_TYPE_CF = "collaborative_filtering"
REC_TYPE_CB = "content_based"

async def verify_api_key(header: str = Security(APIKeyHeader(name="X-API-Key", auto_error=False))):
    if not header:
        raise HTTPException(status_code=403, detail="Brak klucza API w nagłówku")
    safe_api_key = API_KEY or ""
    clean_header = header.strip()
    clean_api_key = safe_api_key.strip().strip('"').strip("'")
    
    if clean_header == clean_api_key and clean_api_key != "": 
        return header
    raise HTTPException(status_code=403, detail="Niepoprawny klucz API")

def compute_recommendations(user_id: int) -> dict:
    results = {
        "status": "success",
        "user_id": user_id,
        "collaborative": {"source": "database_cache", "data": []},
        "content_based": {"source": "database_cache", "data": []}
    }
    
    # 1. POBRANIE DANYCH Z BAZY
    user_ratings = db.fetch_user_ratings(user_id)
    watched_movie_ids = set(db.fetch_watched_movies(user_id))
    onboarding_movie_ids = set(db.fetch_onboarding_movies(user_id))
    
    # Sprawdzamy czy są jakiekolwiek dane do generowania poleceń
    if (user_ratings is None or user_ratings.empty) and not onboarding_movie_ids:
        return {"status": "error", "message": "Brak ocen i filmów startowych użytkownika"}
        
    if user_ratings is not None and not user_ratings.empty:
        rated_movie_ids = set(user_ratings['movie_id'].tolist())
    else:
        rated_movie_ids = set()

    # Zbiór wszystkich filmów, których nie chcemy już polecać
    known_movie_ids = rated_movie_ids.union(watched_movie_ids).union(onboarding_movie_ids)
    
    # --- COLLABORATIVE FILTERING (CF) ---
    cf_cached = db.get_cached_recommendations(user_id, REC_TYPE_CF)
    if cf_cached:
        results["collaborative"]["data"] = cf_cached
        tracker.log("cf", "CF", f"Pobrano {len(cf_cached)} rekomendacji z pamięci podręcznej D1 dla user_id={user_id}.")
    else:
        results["collaborative"]["source"] = "computed"
        
        # CF wymaga prawdziwych ocen
        if user_ratings is not None and not user_ratings.empty:
            df_cf = user_ratings.copy()
            df_cf['user_id'] = user_id
            df_cf = df_cf[['user_id', 'movie_id', 'rating']]
            
            reader = Reader(rating_scale=(1, 5))
            trainset = Dataset.load_from_df(df_cf, reader).build_full_trainset()
            
            if not model_manager.cf_model: 
                model_manager.build_collaborative_filtering_model(trainset, user_id)
                
            movies_df = db.fetch_movies_metadata()
            if model_manager.cf_model is not None and movies_df is not None and not movies_df.empty:
                cf_recs = []
                for m_id in movies_df['id'].tolist():
                    if m_id not in known_movie_ids:
                        pred = model_manager.cf_model.predict(user_id, m_id)
                        cf_recs.append({
                            "movie_id": m_id, 
                            "rating": round(pred.est, 2),
                            "confidence_interval": [0.0, 0.0]
                        })
                cf_recs.sort(key=lambda x: x["rating"], reverse=True)
                top_cf = cf_recs[:10]
                db.save_cached_recommendations(user_id, top_cf, CACHE_MINUTES, REC_TYPE_CF)
                tracker.record_count("cf", len(top_cf))
                tracker.log("cf", "CF", f"Wygenerowano {len(top_cf)} rekomendacji dla user_id={user_id}. Top ID: {[r['movie_id'] for r in top_cf[:3]]}")
            else:
                results["collaborative"]["message"] = "Brak modelu CF lub lokalnej bazy."
        else:
            results["collaborative"]["message"] = "Brak ocen użytkownika do wyliczenia profilu (CF)."

    # --- CONTENT-BASED FILTERING (CB) ---
    cb_cached = db.get_cached_recommendations(user_id, REC_TYPE_CB)
    if cb_cached:
        results["content_based"]["data"] = cb_cached
        tracker.log("cb", "CB", f"Pobrano {len(cb_cached)} rekomendacji z pamięci podręcznej D1 dla user_id={user_id}.")
    else:
        results["content_based"]["source"] = "computed"
        
        # Bierzemy dobre oceny (>= 4)
        good_movies = []
        if user_ratings is not None and not user_ratings.empty:
            good_movies = user_ratings[user_ratings['rating'] >= 4]['movie_id'].tolist()
        
        # FALLBACK ONBOARDINGU (COLD START)
        if not good_movies and onboarding_movie_ids:
            good_movies = list(onboarding_movie_ids)
            tracker.log("cb", "CB", f"Cold start: Użyto {len(good_movies)} filmów z onboardingu dla user_id={user_id}.")
            
        if not good_movies:
            results["content_based"]["message"] = "Brak pozytywnych ocen lub filmów startowych do profilowania CB"
        else:
            if model_manager.tfidf_matrix is None:
                movies_features_df = db.fetch_movies_features()
                if movies_features_df is not None and not movies_features_df.empty:
                    model_manager.build_content_based_model(movies_features_df, user_id)
            
            if model_manager.tfidf_matrix is not None and model_manager.movie_id_map is not None and model_manager.movie_indices is not None:
                similar_scores = {}
                for m_id in good_movies:
                    if m_id in model_manager.movie_indices:
                        idx = model_manager.movie_indices[m_id]
                        cosine_sim = cosine_similarity(model_manager.tfidf_matrix[idx], model_manager.tfidf_matrix).flatten()
                        
                        for sim_idx, score in enumerate(cosine_sim):
                            if score > 0.01:
                                sim_movie_id = int(model_manager.movie_id_map[sim_idx])
                                if sim_movie_id not in known_movie_ids:
                                    similar_scores[sim_movie_id] = similar_scores.get(sim_movie_id, 0) + score
                    
                if similar_scores:
                    sorted_sims = sorted(similar_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                    max_score = sorted_sims[0][1] if sorted_sims else 1
                    cb_recs = []
                    
                    for m_id, score in sorted_sims:
                        normalized_rating = 3.5 + (score / max_score) * 1.5
                        cb_recs.append({
                            "movie_id": m_id,
                            "rating": round(normalized_rating, 2),
                            "confidence_interval": [0.0, 0.0]
                        })
                        
                    db.save_cached_recommendations(user_id, cb_recs, CACHE_MINUTES, REC_TYPE_CB)
                    tracker.record_count("cb", len(cb_recs))
                    tracker.log("cb", "CB", f"Wygenerowano {len(cb_recs)} rekomendacji dla user_id={user_id}. Top ID: {[r['movie_id'] for r in cb_recs[:3]]}")
                    results["content_based"]["data"] = cb_recs

    return results

@router.get("/{user_id}/recommendations", dependencies=[Depends(verify_api_key)])
def get_recommendations(user_id: int):
    return compute_recommendations(user_id)

@router.post("/{user_id}/recommendations/force-recalculate", dependencies=[Depends(verify_api_key)])
def force_recalculate(user_id: int):
    results = {
        "status": "success",
        "user_id": user_id,
        "collaborative": {"source": "forced-recalculation", "data": []},
        "content_based": {"source": "forced-recalculation", "data": []}
    }
    
    user_ratings = db.fetch_user_ratings(user_id)
    watched_movie_ids = set(db.fetch_watched_movies(user_id))
    onboarding_movie_ids = set(db.fetch_onboarding_movies(user_id))
    
    if (user_ratings is None or user_ratings.empty) and not onboarding_movie_ids:
        return {"status": "error", "message": "Brak ocen i filmów startowych użytkownika"}
        
    if user_ratings is not None and not user_ratings.empty:
        rated_movie_ids = set(user_ratings['movie_id'].tolist())
    else:
        rated_movie_ids = set()

    known_movie_ids = rated_movie_ids.union(watched_movie_ids).union(onboarding_movie_ids)
    
    # --- COLLABORATIVE FILTERING ---
    if user_ratings is not None and not user_ratings.empty:
        df_cf = user_ratings.copy()
        df_cf['user_id'] = user_id
        df_cf = df_cf[['user_id', 'movie_id', 'rating']]
        
        reader = Reader(rating_scale=(1, 5))
        trainset = Dataset.load_from_df(df_cf, reader).build_full_trainset()
        model_manager.build_collaborative_filtering_model(trainset, user_id)
            
        movies_df = db.fetch_movies_metadata()
        if model_manager.cf_model is not None and movies_df is not None and not movies_df.empty:
            cf_recs = []
            for m_id in movies_df['id'].tolist():
                if m_id not in known_movie_ids:
                    pred = model_manager.cf_model.predict(user_id, m_id)
                    cf_recs.append({
                        "movie_id": m_id, 
                        "rating": round(pred.est, 2),
                        "confidence_interval": [0.0, 0.0]
                    })
            cf_recs.sort(key=lambda x: x["rating"], reverse=True)
            top_cf_recs = cf_recs[:10]
            db.save_cached_recommendations(user_id, top_cf_recs, CACHE_MINUTES, REC_TYPE_CF)
            tracker.record_count("cf", len(top_cf_recs))
            tracker.log("cf", "CF", f"Wymuszono przeliczenie. Zapisano {len(top_cf_recs)} rec (CF) dla user_id={user_id}.")
            results["collaborative"]["data"] = top_cf_recs
        else:
            results["collaborative"]["message"] = "Brak modelu CF lub lokalnej bazy."
    else:
        results["collaborative"]["message"] = "Brak ocen do wymuszenia przeliczenia CF."

    # --- CONTENT-BASED FILTERING ---
    good_movies = []
    if user_ratings is not None and not user_ratings.empty:
        good_movies = user_ratings[user_ratings['rating'] >= 4]['movie_id'].tolist()
    
    if not good_movies and onboarding_movie_ids:
        good_movies = list(onboarding_movie_ids)
        
    if not good_movies:
        results["content_based"]["message"] = "Brak pozytywnych ocen lub filmów startowych do profilowania CB"
    else:
        movies_features_df = db.fetch_movies_features()
        if movies_features_df is not None and not movies_features_df.empty:
            model_manager.build_content_based_model(movies_features_df, user_id)
            
        if model_manager.tfidf_matrix is not None and model_manager.movie_id_map is not None and model_manager.movie_indices is not None:
            similar_scores = {}
            for m_id in good_movies:
                if m_id in model_manager.movie_indices:
                    idx = model_manager.movie_indices[m_id]
                    cosine_sim = cosine_similarity(model_manager.tfidf_matrix[idx], model_manager.tfidf_matrix).flatten()
                    
                    for sim_idx, score in enumerate(cosine_sim):
                        if score > 0.01:
                            sim_movie_id = int(model_manager.movie_id_map[sim_idx])
                            if sim_movie_id not in known_movie_ids:
                                similar_scores[sim_movie_id] = similar_scores.get(sim_movie_id, 0) + score
                                
            if similar_scores:
                sorted_sims = sorted(similar_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                max_score = sorted_sims[0][1] if sorted_sims else 1
                cb_recs = []
                
                for m_id, score in sorted_sims:
                    normalized_rating = 3.5 + (score / max_score) * 1.5
                    cb_recs.append({
                        "movie_id": m_id,
                        "rating": round(normalized_rating, 2),
                        "confidence_interval": [0.0, 0.0]
                    })
                    
                db.save_cached_recommendations(user_id, cb_recs, CACHE_MINUTES, REC_TYPE_CB)
                tracker.record_count("cb", len(cb_recs))
                tracker.log("cb", "CB", f"Wymuszono przeliczenie. Zapisano {len(cb_recs)} rec (CB) dla user_id={user_id}.")
                results["content_based"]["data"] = cb_recs

    return results
