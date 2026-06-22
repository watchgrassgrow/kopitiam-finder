import { useState, useCallback } from 'react'

// ── Injected at build time by GitHub Actions from repository secret ───────────
const GEMINI_KEY = import.meta.env.VITE_GEMINI_API_KEY || ''
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_KEY}`

// ── Colours ───────────────────────────────────────────────────────────────────
const C = {
  bg: '#0f0e0c', surface: '#1a1815', card: '#201e1b', border: '#2e2b26',
  accent: '#e8a045', accent2: '#c45c2a', text: '#f0ead8', muted: '#7a7060',
  green: '#5a9e6f', red: '#c45c2a', blue: '#4a90b8',
}

// ── Global styles ─────────────────────────────────────────────────────────────
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: ${C.bg}; color: ${C.text}; font-family: 'DM Mono', monospace; }
  #root { min-height: 100vh; }
  @keyframes fadeUp   { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
  @keyframes spin     { to { transform: rotate(360deg); } }
  @keyframes dotBeat  { 0%,80%,100% { transform:scale(0.6); opacity:0.4; } 40% { transform:scale(1); opacity:1; } }
  .fade-card   { animation: fadeUp 0.4s ease both; }
  .spinner     { display:inline-block; width:10px; height:10px; border:2px solid currentColor; border-top-color:transparent; border-radius:50%; animation:spin 0.8s linear infinite; }
  .dot-loader span { display:inline-block; width:7px; height:7px; background:${C.accent}; border-radius:50%; margin:0 3px; animation:dotBeat 1.2s infinite; }
  .dot-loader span:nth-child(2) { animation-delay:0.2s; }
  .dot-loader span:nth-child(3) { animation-delay:0.4s; }
  input, select, button { font-family: inherit; }
  input:focus { outline: none; border-color: ${C.accent} !important; }
  select:focus { outline: none; }
  a { color: ${C.accent}; }
`

// ── SVY21 → WGS84 ─────────────────────────────────────────────────────────────
function svy21ToLatLng(N, E) {
  const a=6378137, f=1/298.257223563, e2=2*f-f*f, e_2=e2/(1-e2)
  const n0=38744.572, E0=28001.642, k0=1, oLat=1.366666, oLon=103.833333
  const oLatR = oLat * Math.PI / 180
  const n=f/(2-f), n2=n*n, n3=n2*n, n4=n3*n
  const A = (a/(1+n)) * (1 + n2/4 + n4/64)
  const M0 = A*(oLatR + (-3*n/2+9*n3/16)*Math.sin(2*oLatR) + (15*n2/16-15*n4/32)*Math.sin(4*oLatR) + (-35*n3/48)*Math.sin(6*oLatR) + (315*n4/512)*Math.sin(8*oLatR))
  const Np=(N-n0)+k0*M0, Ep=(E-E0)/k0, pp=Np/(A*k0)
  const p1=pp + (3*n/2-27*n3/32)*Math.sin(2*pp) + (21*n2/16-55*n4/32)*Math.sin(4*pp) + (151*n3/96)*Math.sin(6*pp) + (1097*n4/512)*Math.sin(8*pp)
  const nu = a/Math.sqrt(1-e2*Math.sin(p1)*Math.sin(p1))
  const t=Math.tan(p1), t2=t*t, ep2=e_2*Math.cos(p1)*Math.cos(p1)
  const x=Ep/(nu*k0), x2=x*x
  const lat = p1 - (nu*t/(a*a/(1-e2*Math.sin(p1)*Math.sin(p1)))) * (x2/2 - x2*x2*(5+3*t2+ep2-9*t2*ep2)/24)
  const lon = oLon*Math.PI/180 + (x - x2*x*(1+2*t2+ep2)/6) / Math.cos(p1)
  return { lat: lat*180/Math.PI, lng: lon*180/Math.PI }
}

function haversine(a, b, c, d) {
  const R=6371000, dL=(c-a)*Math.PI/180, dN=(d-b)*Math.PI/180
  const e = Math.sin(dL/2)**2 + Math.cos(a*Math.PI/180)*Math.cos(c*Math.PI/180)*Math.sin(dN/2)**2
  return R * 2 * Math.atan2(Math.sqrt(e), Math.sqrt(1-e))
}

// ── Gemini API call with Google Search grounding ───────────────────────────────
async function callGemini(prompt, useSearch = true) {
  const body = {
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    generationConfig: { temperature: 0.1, maxOutputTokens: 4096 },
  }
  if (useSearch) {
    body.tools = [{ google_search: {} }]
  }
  const resp = await fetch(GEMINI_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = await resp.json()
  if (data.error) throw new Error(`Gemini error: ${data.error.message}`)
  const text = data.candidates?.[0]?.content?.parts?.map(p => p.text || '').join('') || ''
  return text
}

// ── Parse JSON from Gemini response (strips markdown fences) ──────────────────
function parseJSON(text) {
  const clean = text.replace(/```json|```/gi, '').trim()
  const match = clean.match(/\{[\s\S]*\}|\[[\s\S]*\]/)
  if (!match) throw new Error('No JSON found in response')
  return JSON.parse(match[0])
}

// ── Step 1: Geocode via Gemini + Google Search → OneMap ───────────────────────
async function geocodeLocation(query) {
  const isPostal = /^\d{6}$/.test(query.trim())
  const prompt = `You are a Singapore geocoding assistant.

${isPostal
  ? `The user entered Singapore postal code: ${query}
Use Google Search to look up: "Singapore postal code ${query} address location"
Then also search for the OneMap result: site:onemap.gov.sg "${query}" OR "postal ${query} singapore coordinates"`
  : `The user entered Singapore location: "${query}"
Use Google Search to find the precise latitude and longitude of "${query}" in Singapore.`}

Based on your search results, return ONLY this JSON (no markdown, no explanation):
{"lat": 1.3521, "lng": 103.8198, "address": "full address string", "postal": "${isPostal ? query : 'null or postal code if found'}"}

Requirements:
- lat/lng must be precise Singapore coordinates (lat between 1.1 and 1.5, lng between 103.5 and 104.1)
- address should be the full street address
- If postal code entered, resolve to the actual block/street (e.g. 541338 → "BLK 338 ANG MO KIO AVE 3")`

  const text = await callGemini(prompt, true)
  const result = parseJSON(text)
  if (!result.lat || !result.lng) throw new Error('Could not resolve location coordinates')
  if (result.lat < 1.1 || result.lat > 1.5) throw new Error(`Invalid coordinates: ${result.lat}, ${result.lng}`)
  return result
}

// ── Step 2: Fetch HDB carparks via Gemini + Google Search ────────────────────
async function fetchCarparks(lat, lng, radiusM) {
  const prompt = `You are a Singapore HDB carpark data assistant.

Use Google Search to fetch live data from these URLs:
1. https://api.data.gov.sg/v1/transport/carpark-availability
2. https://data.gov.sg/api/action/datastore_search?resource_id=139a3035-e624-4f56-b63f-89ae28d4ae4c&limit=2000

Search location: latitude ${lat}, longitude ${lng}
Find all HDB carparks within ${radiusM}m of this point.

The carpark info dataset uses SVY21 coordinates (x_coord, y_coord). To convert to WGS84 approximately:
- lng ≈ (x_coord / 111320) + 103.0  
- lat ≈ (y_coord / 111320) + 1.1

Filter for carparks within ${radiusM}m. Return the 8 closest ones, merged with live availability from dataset 1.

Return ONLY a JSON array (no markdown):
[
  {
    "car_park_no": "SK1",
    "address": "BLK 123 SENGKANG EAST AVE 1",
    "car_park_type": "SURFACE CAR PARK",
    "free_parking": "SUN & PH FR 7AM-10:30PM",
    "night_parking": "YES",
    "gantry_height": "2.1",
    "distM": 150,
    "availLots": 45,
    "totalLots": 200
  }
]`

  const text = await callGemini(prompt, true)
  try {
    const result = parseJSON(text)
    return Array.isArray(result) ? result : result.carparks || []
  } catch {
    return []
  }
}

// ── Step 3: AI place matching (no search needed) ──────────────────────────────
async function fetchPlaces(address, lat, lng, carparks, radiusM) {
  const cpList = carparks.slice(0, 8).map((cp, i) =>
    `#${i+1} ${cp.car_park_no}: ${cp.address}, type=${cp.car_park_type}, free=${cp.free_parking}, night=${cp.night_parking}, gantry=${cp.gantry_height||'?'}m, dist=${cp.distM}m, lots=${cp.availLots??'?'}/${cp.totalLots??'?'}`
  ).join('\n')

  const prompt = `You are a Singapore local food guide.

Search location: "${address}" (${lat}, ${lng})
Nearby HDB carparks found:
${cpList || 'No carparks found in dataset — use general area knowledge'}

Suggest 6 realistic affordable eating places (coffeeshops, hawker centres, food courts) near this Singapore location. Each must be paired with one carpark from the list above.

Return ONLY a JSON array (no markdown):
[
  {
    "name": "Name of eating place",
    "type": "Coffeeshop",
    "address": "BLK 123 Street Name",
    "distance_m": 120,
    "avg_spend": "$3–5 per pax",
    "famous_for": "Kaya toast, Wonton mee, Teh tarik",
    "open_hours": "6am–10pm",
    "crowd_level": "Moderate",
    "emoji": "☕",
    "tips": "One useful local tip",
    "carpark_no": "SK1",
    "carpark_walk_m": 80
  }
]`

  const text = await callGemini(prompt, false)
  const result = parseJSON(text)
  return Array.isArray(result) ? result : result.places || []
}

// ── UI Components ─────────────────────────────────────────────────────────────

function PipeStep({ icon, label, detail, state }) {
  const bg = state==='active' ? 'rgba(232,160,69,0.07)' : state==='done' ? 'rgba(90,158,111,0.07)' : state==='error' ? 'rgba(196,92,42,0.07)' : 'transparent'
  const col = state==='active' ? C.accent : state==='done' ? C.green : state==='error' ? C.red : C.muted
  return (
    <div style={{ background: bg, color: col, padding: '9px 14px', borderBottom: `1px solid ${C.border}`, display: 'flex', alignItems: 'center', gap: 10, transition: 'all 0.3s', fontSize: 10, letterSpacing: 1, textTransform: 'uppercase' }}>
      <span style={{ fontSize: 14, flexShrink: 0 }}>{icon}</span>
      <div style={{ flex: 1 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {label} {state === 'active' && <span className="spinner" />}
          {state === 'done' && <span style={{ fontSize: 10 }}>✓</span>}
          {state === 'error' && <span style={{ fontSize: 10 }}>✗</span>}
        </div>
        <div style={{ fontSize: 9, opacity: 0.7, marginTop: 1 }}>{detail}</div>
      </div>
    </div>
  )
}

function Badge({ color, children }) {
  const styles = {
    green: { bg: 'rgba(90,158,111,0.15)', col: C.green, b: '1px solid rgba(90,158,111,0.3)' },
    orange: { bg: 'rgba(232,160,69,0.12)', col: C.accent, b: '1px solid rgba(232,160,69,0.25)' },
    blue:   { bg: 'rgba(74,144,184,0.12)', col: C.blue,   b: '1px solid rgba(74,144,184,0.25)' },
    gray:   { bg: 'rgba(122,112,96,0.15)', col: C.muted,  b: `1px solid ${C.border}` },
  }
  const s = styles[color] || styles.gray
  return <span style={{ background: s.bg, color: s.col, border: s.b, fontSize: 9, letterSpacing: 1, textTransform: 'uppercase', padding: '2px 6px', borderRadius: 2 }}>{children}</span>
}

function PlaceCard({ p, i }) {
  const cp = p.cp
  const avail = cp?.availLots ?? null
  const total = cp?.totalLots ?? null
  const pct = (avail != null && total) ? Math.round(avail / total * 100) : 0
  const barCol = pct > 40 ? C.green : pct > 15 ? C.accent : C.red
  const mapUrl = `https://www.google.com/maps/search/${encodeURIComponent(p.name + ' ' + p.address + ' Singapore')}`
  const cpUrl = cp ? `https://www.google.com/maps/search/${encodeURIComponent(cp.address + ' Singapore carpark')}` : ''

  return (
    <div className="fade-card" style={{ animationDelay: `${i * 0.07}s`, background: C.card, border: `1px solid ${C.border}`, borderRadius: 6, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ padding: '13px 15px 11px', borderBottom: `1px solid ${C.border}`, display: 'flex', alignItems: 'flex-start', gap: 10 }}>
        <div style={{ fontSize: 24, flexShrink: 0, marginTop: 2 }}>{p.emoji || '🍽'}</div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 700, fontSize: 13, color: C.text, lineHeight: 1.3, marginBottom: 3 }}>{p.name}</div>
          <span style={{ fontSize: 9, letterSpacing: 2, textTransform: 'uppercase', color: C.accent2, border: `1px solid ${C.accent2}`, borderRadius: 2, padding: '1px 5px' }}>{p.type}</span>
        </div>
        <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 700, fontSize: 14, color: C.green, flexShrink: 0 }}>{(p.avg_spend || '').split(' ')[0]}</div>
      </div>

      {/* Body */}
      <div style={{ padding: '11px 15px' }}>
        {[
          { icon: '📍', val: p.address },
          { icon: '🍽', val: p.famous_for },
          { icon: '🕐', val: `${p.open_hours} · ${p.crowd_level === 'Quiet' ? '😌' : p.crowd_level === 'Busy' ? '🔥' : '👥'} ${p.crowd_level}` },
        ].map((r, idx) => (
          <div key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: 7, marginBottom: 8, fontSize: 11, lineHeight: 1.5 }}>
            <span style={{ fontSize: 12, flexShrink: 0, marginTop: 1 }}>{r.icon}</span>
            <span style={{ color: C.text, flex: 1 }}>{r.val}</span>
          </div>
        ))}

        {/* Carpark row */}
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 7, marginBottom: 8, fontSize: 11, lineHeight: 1.5 }}>
          <span style={{ fontSize: 12, flexShrink: 0, marginTop: 1 }}>🅿</span>
          {cp ? (
            <div style={{ color: C.text, flex: 1 }}>
              <strong>{cp.car_park_no}</strong> · {cp.address}
              <div style={{ color: C.muted, fontSize: 10 }}>{cp.car_park_type} · ~{p.carpark_walk_m}m walk · {cp.distM}m from search</div>
              {avail != null && total ? (
                <div style={{ marginTop: 5 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: C.muted, marginBottom: 3 }}>
                    <span>{avail}/{total} lots free ({pct}%)</span>
                    {cp.night_parking === 'YES' && <span style={{ color: C.blue }}>🌙 Night OK</span>}
                  </div>
                  <div style={{ height: 3, background: C.border, borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${pct}%`, background: barCol, borderRadius: 2, transition: 'width 0.8s ease' }} />
                  </div>
                </div>
              ) : (
                <div style={{ fontSize: 10, color: C.muted, marginTop: 3 }}>
                  {cp.night_parking === 'YES' ? '🌙 Night parking OK · ' : ''}Availability not available
                </div>
              )}
            </div>
          ) : (
            <span style={{ color: C.muted, fontSize: 10 }}>No HDB carpark matched — check street parking</span>
          )}
        </div>

        {p.tips && (
          <div style={{ display: 'flex', gap: 7, marginBottom: 8, fontSize: 10, color: C.muted, fontStyle: 'italic', lineHeight: 1.5 }}>
            <span>💡</span><span>{p.tips}</span>
          </div>
        )}

        {/* Badges */}
        <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginTop: 9 }}>
          {cp?.car_park_type?.includes('SURFACE') && <Badge color="green">🅿 Surface</Badge>}
          {cp?.car_park_type?.includes('MULTI')   && <Badge color="blue">🏢 Multi-storey</Badge>}
          {cp?.car_park_type?.includes('BASEMENT') && <Badge color="gray">🏠 Basement</Badge>}
          {cp?.free_parking && cp.free_parking !== 'NO' && <Badge color="green">💚 Free</Badge>}
          {cp?.night_parking === 'YES' && <Badge color="blue">🌙 Night</Badge>}
          {cp?.gantry_height && <Badge color="gray">↕{cp.gantry_height}m</Badge>}
          <Badge color="orange">~{p.avg_spend}</Badge>
        </div>
      </div>

      {/* Footer */}
      <div style={{ padding: '9px 15px', borderTop: `1px solid ${C.border}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ fontSize: 10, color: C.muted }}>~<strong style={{ color: C.accent }}>{p.distance_m}m</strong> walk</div>
        <div style={{ display: 'flex', gap: 5 }}>
          {cp && (
            <button onClick={() => window.open(cpUrl, '_blank')}
              style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, letterSpacing: 1, color: C.accent, background: 'transparent', border: `1px solid rgba(232,160,69,0.3)`, padding: '4px 9px', borderRadius: 3, cursor: 'pointer', textTransform: 'uppercase' }}>
              🅿 Park
            </button>
          )}
          <button onClick={() => window.open(mapUrl, '_blank')}
            style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, letterSpacing: 1, color: C.accent, background: 'transparent', border: `1px solid rgba(232,160,69,0.3)`, padding: '4px 9px', borderRadius: 3, cursor: 'pointer', textTransform: 'uppercase' }}>
            📍 Maps
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [query, setQuery]       = useState('')
  const [radius, setRadius]     = useState(600)
  const [loading, setLoading]   = useState(false)
  const [allResults, setAllResults] = useState([])
  const [locLabel, setLocLabel] = useState('')
  const [error, setError]       = useState('')
  const [activeFilters, setActiveFilters] = useState(new Set(['all']))
  const [pipe, setPipe] = useState({
    s1: { state: '', detail: 'Google Search → OneMap for exact coordinates' },
    s2: { state: '', detail: 'Google Search → data.gov.sg live lot availability' },
    s3: { state: '', detail: 'AI matches eating places to carparks' },
    s4: { state: '', detail: 'All done' },
  })

  const updPipe = (key, state, detail) =>
    setPipe(p => ({ ...p, [key]: { state, detail: detail ?? p[key].detail } }))

  const toggleFilter = (f) => setActiveFilters(prev => {
    const n = new Set(prev)
    if (f === 'all') return new Set(['all'])
    n.delete('all')
    if (n.has(f)) n.delete(f); else n.add(f)
    return n.size === 0 ? new Set(['all']) : n
  })

  const filtered = allResults.filter(p => {
    if (activeFilters.has('all')) return true
    if (activeFilters.has('coffeeshop') && p.type === 'Coffeeshop') return true
    if (activeFilters.has('hawker')     && p.type === 'Hawker Centre') return true
    if (activeFilters.has('foodcourt')  && p.type === 'Food Court') return true
    if (activeFilters.has('surface')    && p.cp?.car_park_type?.includes('SURFACE')) return true
    if (activeFilters.has('free')       && p.cp?.free_parking && p.cp.free_parking !== 'NO') return true
    if (activeFilters.has('available')  && (p.cp?.availLots ?? 0) > 10) return true
    return false
  })

  const doSearch = useCallback(async () => {
    const q = query.trim()
    if (!q) { setError('Please enter a location or postal code.'); return }
    if (!GEMINI_KEY) { setError('Gemini API key not configured. Check VITE_GEMINI_API_KEY environment variable.'); return }

    setError(''); setLocLabel(''); setAllResults([])
    setLoading(true)
    setPipe({
      s1: { state: 'active', detail: 'Searching OneMap for exact coordinates…' },
      s2: { state: '',       detail: 'Google Search → data.gov.sg live lot availability' },
      s3: { state: '',       detail: 'AI matches eating places to carparks' },
      s4: { state: '',       detail: 'All done' },
    })

    try {
      // ── Step 1: Geocode ──────────────────────────────────────────────────
      let loc
      try {
        loc = await geocodeLocation(q)
        updPipe('s1', 'done', `${loc.address} (${Number(loc.lat).toFixed(5)}, ${Number(loc.lng).toFixed(5)})`)
        setLocLabel(`${loc.address}${loc.postal && loc.postal !== 'null' ? ' · ' + loc.postal : ''}`)
      } catch (e) {
        updPipe('s1', 'error', e.message)
        throw new Error('Geocode failed: ' + e.message)
      }

      // ── Step 2: Carparks ─────────────────────────────────────────────────
      updPipe('s2', 'active', 'Fetching HDB carpark dataset + live availability…')
      let carparks = []
      try {
        carparks = await fetchCarparks(loc.lat, loc.lng, radius)
        updPipe('s2', 'done', `${carparks.length} carparks found within ${radius}m`)
        if (carparks.length === 0) {
          setError(`No HDB carparks found within ${radius}m. Try a larger radius.`)
        }
      } catch (e) {
        updPipe('s2', 'error', 'Carpark fetch failed — AI will estimate')
        carparks = []
      }

      // ── Step 3: Places ───────────────────────────────────────────────────
      updPipe('s3', 'active', 'Matching eating places to carparks…')
      let places = []
      try {
        places = await fetchPlaces(loc.address, loc.lat, loc.lng, carparks, radius)
        updPipe('s3', 'done', `${places.length} eating places matched`)
      } catch (e) {
        updPipe('s3', 'error', e.message)
        throw new Error('Place matching failed: ' + e.message)
      }

      // ── Merge & display ──────────────────────────────────────────────────
      const cpMap = {}
      carparks.forEach(cp => { cpMap[cp.car_park_no] = cp })
      const merged = places
        .map(p => ({ ...p, cp: cpMap[p.carpark_no] || null }))
        .sort((a, b) => a.distance_m - b.distance_m)

      setAllResults(merged)
      updPipe('s4', 'done', `Showing ${merged.length} results`)

    } catch (e) {
      console.error(e)
      setError(e.message)
    }

    setLoading(false)
  }, [query, radius])

  const FILTERS = [
    { k: 'all',        l: 'All' },
    { k: 'coffeeshop', l: '☕ Coffeeshop' },
    { k: 'hawker',     l: '🍜 Hawker' },
    { k: 'foodcourt',  l: '🏪 Food Court' },
    { k: 'surface',    l: '🅿 Surface Lot' },
    { k: 'free',       l: '💚 Free Parking' },
    { k: 'available',  l: '✅ Lots Free' },
  ]

  return (
    <>
      <style>{GLOBAL_CSS}</style>
      <div style={{ minHeight: '100vh', background: C.bg, color: C.text }}>

        {/* ── Header ── */}
        <div style={{ padding: '28px 20px 14px', borderBottom: `1px solid ${C.border}` }}>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10, marginBottom: 4 }}>
            <span style={{ fontSize: 28 }}>☕</span>
            <h1 style={{ fontFamily: "'Syne', sans-serif", fontSize: 'clamp(22px,6vw,40px)', fontWeight: 800, color: C.accent, letterSpacing: -1, lineHeight: 1 }}>
              Kopitiam Finder
            </h1>
          </div>
          <div style={{ fontSize: 10, color: C.muted, letterSpacing: 3, textTransform: 'uppercase' }}>
            Accurate geocoding · Real HDB carpark data · Singapore
          </div>
        </div>

        {/* ── Search bar ── */}
        <div style={{ padding: '18px 20px 0' }}>
          <div style={{ fontSize: 10, letterSpacing: 3, textTransform: 'uppercase', color: C.muted, marginBottom: 8 }}>
            Location · Postal Code · Address
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <div style={{ flex: 1, minWidth: 180, position: 'relative' }}>
              <span style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: C.muted, fontSize: 14, pointerEvents: 'none' }}>📍</span>
              <input
                type="text" value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !loading && doSearch()}
                placeholder="541338 · Sengkang · Tampines MRT..."
                style={{ width: '100%', background: C.surface, border: `1px solid ${C.border}`, color: C.text, fontSize: 14, padding: '12px 12px 12px 36px', borderRadius: 4 }}
              />
            </div>
            <select value={radius} onChange={e => setRadius(Number(e.target.value))}
              style={{ background: C.surface, border: `1px solid ${C.border}`, color: C.text, fontSize: 12, padding: '12px 8px', borderRadius: 4, cursor: 'pointer' }}>
              {[400, 600, 800, 1000, 1500].map(r => (
                <option key={r} value={r}>{r >= 1000 ? r/1000 + 'km' : r + 'm'}</option>
              ))}
            </select>
            <button onClick={doSearch} disabled={loading}
              style={{ background: loading ? C.muted : C.accent, color: '#0f0e0c', border: 'none', fontFamily: "'Syne', sans-serif", fontWeight: 700, fontSize: 13, letterSpacing: 1, padding: '12px 20px', borderRadius: 4, cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, whiteSpace: 'nowrap' }}>
              {loading ? 'Searching…' : 'Search'}
            </button>
          </div>
        </div>

        {/* ── Pipeline status ── */}
        <div style={{ margin: '14px 20px 0', border: `1px solid ${C.border}`, borderRadius: 4, overflow: 'hidden' }}>
          <PipeStep icon="🗺" label="1 · Geocode"       detail={pipe.s1.detail} state={pipe.s1.state} />
          <PipeStep icon="🅿" label="2 · HDB Carparks"  detail={pipe.s2.detail} state={pipe.s2.state} />
          <PipeStep icon="🤖" label="3 · Place Match"   detail={pipe.s3.detail} state={pipe.s3.state} />
          <PipeStep icon="✅" label="4 · Results Ready" detail={pipe.s4.detail} state={pipe.s4.state} />
        </div>

        {/* ── Filters ── */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', margin: '12px 20px 0' }}>
          {FILTERS.map(f => (
            <button key={f.k} onClick={() => toggleFilter(f.k)}
              style={{ background: activeFilters.has(f.k) ? 'rgba(232,160,69,0.08)' : 'transparent', border: `1px solid ${activeFilters.has(f.k) ? C.accent : C.border}`, color: activeFilters.has(f.k) ? C.accent : C.muted, fontSize: 10, letterSpacing: 1, padding: '5px 10px', borderRadius: 100, cursor: 'pointer', textTransform: 'uppercase' }}>
              {f.l}
            </button>
          ))}
        </div>

        {/* ── Info bars ── */}
        {locLabel && (
          <div style={{ margin: '10px 20px 0', background: 'rgba(74,144,184,0.08)', border: `1px solid rgba(74,144,184,0.25)`, borderRadius: 4, padding: '9px 12px', fontSize: 11, color: C.blue }}>
            📌 <strong>{locLabel}</strong>
          </div>
        )}
        {error && (
          <div style={{ margin: '10px 20px 0', background: 'rgba(196,92,42,0.1)', border: `1px solid rgba(196,92,42,0.3)`, borderRadius: 4, padding: '9px 12px', fontSize: 11, color: '#e07050' }}>
            ⚠ {error}
          </div>
        )}

        {/* ── Results ── */}
        <div style={{ padding: '16px 20px 48px' }}>
          {loading ? (
            <div style={{ textAlign: 'center', padding: '48px 20px', color: C.muted }}>
              <div className="dot-loader"><span /><span /><span /></div>
              <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 16, color: C.text, marginTop: 14, marginBottom: 8 }}>Searching Singapore…</div>
              <div style={{ fontSize: 11, lineHeight: 2.0 }}>
                Step 1 · Gemini + Google Search → OneMap geocode<br />
                Step 2 · Live HDB carpark availability<br />
                Step 3 · AI place matching<br />
                <span style={{ color: C.accent2, fontSize: 10 }}>Takes ~20–30 seconds</span>
              </div>
            </div>
          ) : filtered.length > 0 ? (
            <>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: 12, borderBottom: `1px solid ${C.border}`, marginBottom: 14 }}>
                <div style={{ fontSize: 11, color: C.muted, letterSpacing: 2, textTransform: 'uppercase' }}>
                  <span style={{ color: C.accent }}>{filtered.length}</span> places near {query}
                </div>
                <div style={{ fontSize: 10, color: C.muted }}>sorted by distance</div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
                {filtered.map((p, i) => <PlaceCard key={i} p={p} i={i} />)}
              </div>
            </>
          ) : !loading && allResults.length === 0 && (
            <div style={{ textAlign: 'center', padding: '48px 20px', color: C.muted }}>
              <div style={{ fontSize: 40, marginBottom: 12 }}>🗺️</div>
              <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 16, color: C.text, marginBottom: 10 }}>Find your next kopi session</div>
              <div style={{ fontSize: 11, lineHeight: 2.0 }}>
                Enter a postal code, MRT, or area name<br />
                <strong style={{ color: C.accent }}>541338</strong> &nbsp;·&nbsp; <strong style={{ color: C.accent }}>Sengkang</strong> &nbsp;·&nbsp; <strong style={{ color: C.accent }}>Tampines MRT</strong><br />
                <span style={{ fontSize: 10 }}>Powered by Gemini + Google Search grounding</span>
              </div>
            </div>
          )}
        </div>

        {/* ── Footer ── */}
        <div style={{ fontSize: 10, color: C.muted, textAlign: 'center', padding: 10, borderTop: `1px solid ${C.border}`, letterSpacing: 1 }}>
          <span style={{ color: C.accent2 }}>Gemini 2.0 Flash</span> &nbsp;·&nbsp;
          <span style={{ color: C.accent2 }}>Google Search Grounding</span> &nbsp;·&nbsp;
          <span style={{ color: C.accent2 }}>HDB Carpark data.gov.sg</span> &nbsp;·&nbsp;
          <span style={{ color: C.accent2 }}>OneMap Singapore</span>
        </div>
      </div>
    </>
  )
}
