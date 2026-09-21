// Pomocnicza funkcja dekodowania tokenu
async function getUserIdFromRequest(request) {
  const authHeader = request.headers.get('Authorization');
  if (!authHeader || !authHeader.startsWith('Bearer ')) return null;
  try {
    const token = authHeader.substring(7);
    const payload = JSON.parse(atob(token));
    if (payload.exp < Date.now()) return null;
    return payload.userId;
  } catch {
    return null;
  }
}

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

// POST - Przenosi filmy z onboardingu do obejrzanych
export async function onRequestPost(context) {
  const { request, env } = context;

  const userId = await getUserIdFromRequest(request);
  if (!userId) {
    return new Response(JSON.stringify({ error: 'Not authenticated' }), {
      status: 401,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }

  try {
    const body = await request.json();
    const movieIds = Array.isArray(body.movieIds) ? body.movieIds : [];

    if (movieIds.length === 0) {
      return new Response(JSON.stringify({ success: true, message: 'Brak filmów' }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    const statements = [];
    
    // 1. Dodajemy do tabeli watched (dzisiejsza data, status watched)
    const insertWatchedStmt = env.db.prepare(
      `INSERT OR IGNORE INTO watched (user_id, movie_id, watched_date, status) 
       VALUES (?, ?, date('now'), 'watched')`
    );
    
    // 2. Usuwamy z tabeli user_onboarding_movies
    const deleteOnboardingStmt = env.db.prepare(
      `DELETE FROM user_onboarding_movies WHERE user_id = ? AND movie_id = ?`
    );

    // Pakujemy to w jedną transakcję batch() dla każdego ID
    for (const movieId of movieIds) {
      statements.push(insertWatchedStmt.bind(userId, movieId));
      statements.push(deleteOnboardingStmt.bind(userId, movieId));
    }

    await env.db.batch(statements);

    return new Response(JSON.stringify({ success: true }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
}

export async function onRequestOptions() {
  return new Response(null, { headers: corsHeaders });
}
