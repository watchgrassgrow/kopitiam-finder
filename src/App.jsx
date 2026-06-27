import { useState, useCallback } from 'react'

// ── API keys injected at build time from GitHub Secrets ───────────────────────
const GEMINI_KEY = import.meta.env.VITE_GEMINI_API_KEY || ''
const GROQ_KEY   = import.meta.env.VITE_GROQ_API_KEY   || ''
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${GEMINI_KEY}`
const GROQ_URL   = 'https://api.groq.com/openai/v1/chat/completions'

const C = {
  bg:'#0f0e0c', surface:'#1a1815', card:'#201e1b', border:'#2e2b26',
  accent:'#e8a045', accent2:'#c45c2a', text:'#f0ead8', muted:'#7a7060',
  green:'#5a9e6f', red:'#c45c2a', blue:'#4a90b8',
}

const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
  html,body{background:#0f0e0c;color:#f0ead8;font-family:'DM Mono',monospace;}
  #root{min-height:100vh;}
  @keyframes fadeUp{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
  @keyframes spin{to{transform:rotate(360deg);}}
  @keyframes dotBeat{0%,80%,100%{transform:scale(0.6);opacity:0.4;}40%{transform:scale(1);opacity:1;}}
  .fade-card{animation:fadeUp 0.4s ease both;}
  .spinner{display:inline-block;width:10px;height:10px;border:2px solid currentColor;border-top-color:transparent;border-radius:50%;animation:spin 0.8s linear infinite;}
  .dot-loader span{display:inline-block;width:7px;height:7px;background:#e8a045;border-radius:50%;margin:0 3px;animation:dotBeat 1.2s infinite;}
  .dot-loader span:nth-child(2){animation-delay:0.2s;}
  .dot-loader span:nth-child(3){animation-delay:0.4s;}
  input,select,button{font-family:inherit;}
  input:focus{outline:none;border-color:#e8a045 !important;}
  select:focus{outline:none;}
`

function parseJSON(text) {
  const clean = text.replace(/```json|```/gi,'').trim()
  const match = clean.match(/\{[\s\S]*\}|\[[\s\S]*\]/)
  if (!match) throw new Error('No JSON found in response')
  return JSON.parse(match[0])
}

function isRateLimit(msg='') {
  const m = msg.toLowerCase()
  return m.includes('high demand')||m.includes('quota')||m.includes('rate limit')||m.includes('429')||m.includes('overloaded')
}

async function callGroq(prompt) {
  if (!GROQ_KEY) throw new Error('No Groq API key')
  const resp = await fetch(GROQ_URL,{
    method:'POST',
    headers:{'Content-Type':'application/json','Authorization':`Bearer ${GROQ_KEY}`},
    body:JSON.stringify({model:'llama-3.3-70b-versatile',messages:[{role:'user',content:prompt}],temperature:0.1,max_tokens:4096})
  })
  const data = await resp.json()
  if (data.error) throw new Error(`Groq error: ${data.error.message}`)
  return data.choices?.[0]?.message?.content || ''
}

async function callAI(prompt, useSearch=true, label='') {
  if (GEMINI_KEY) {
    for (let attempt=1; attempt<=2; attempt++) {
      try {
        const body = {contents:[{role:'user',parts:[{text:prompt}]}],generationConfig:{temperature:0.1,maxOutputTokens:4096}}
        if (useSearch) body.tools = [{google_search:{}}]
        const resp = await fetch(GEMINI_URL,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
        const data = await resp.json()
        if (data.error) {
          if (isRateLimit(data.error.message) && attempt===1) { await new Promise(r=>setTimeout(r,4000)); continue }
          throw new Error(data.error.message)
        }
        const text = data.candidates?.[0]?.content?.parts?.map(p=>p.text||'').join('')||''
        if (text) return {text, provider:'Gemini'}
      } catch(e) {
        if (isRateLimit(e.message) && attempt===1) { await new Promise(r=>setTimeout(r,4000)); continue }
        console.warn(`Gemini failed (${label}): ${e.message}`)
        break
      }
    }
  }
  console.log(`Groq fallback for ${label}`)
  const text = await callGroq(prompt)
  return {text, provider:'Groq'}
}

async function geocodeLocation(query) {
  const isPostal = /^\d{6}$/.test(query.trim())
  const prompt = `You are a Singapore geocoding assistant with deep knowledge of Singapore addresses.

${isPostal
  ? `Resolve Singapore postal code: ${query}
Use Google Search to find the exact address for this postal code.
Also search OneMap: https://www.onemap.gov.sg/api/common/elastic/search?searchVal=${query}&returnGeom=Y&getAddrDetails=Y`
  : `Find coordinates for: "${query}" in Singapore using Google Search.`}

Return ONLY this JSON (no markdown):
{"lat":1.3521,"lng":103.8198,"address":"full address","postal":"${isPostal?query:'postal if found'}"}

lat must be 1.1-1.5, lng must be 103.5-104.1`

  const {text, provider} = await callAI(prompt, true, 'geocode')
  const result = parseJSON(text)
  if (!result.lat||!result.lng) throw new Error('Could not resolve location')
  if (result.lat<1.1||result.lat>1.5) throw new Error(`Invalid coordinates: ${result.lat},${result.lng}`)
  return {...result, provider}
}

function svy21ToLatLng(N, E) {
  const a=6378137,f=1/298.257223563,e2=2*f-f*f,e_2=e2/(1-e2)
  const n0=38744.572,E0=28001.642,k0=1,oLat=1.366666,oLon=103.833333,oLatR=oLat*Math.PI/180
  const n=f/(2-f),n2=n*n,n3=n2*n,n4=n3*n
  const A=(a/(1+n))*(1+n2/4+n4/64)
  const M0=A*(oLatR+(-3*n/2+9*n3/16)*Math.sin(2*oLatR)+(15*n2/16-15*n4/32)*Math.sin(4*oLatR)+(-35*n3/48)*Math.sin(6*oLatR)+(315*n4/512)*Math.sin(8*oLatR))
  const Np=(N-n0)+k0*M0,Ep=(E-E0)/k0,pp=Np/(A*k0)
  const p1=pp+(3*n/2-27*n3/32)*Math.sin(2*pp)+(21*n2/16-55*n4/32)*Math.sin(4*pp)+(151*n3/96)*Math.sin(6*pp)+(1097*n4/512)*Math.sin(8*pp)
  const nu=a/Math.sqrt(1-e2*Math.sin(p1)*Math.sin(p1))
  const t=Math.tan(p1),t2=t*t,ep2=e_2*Math.cos(p1)*Math.cos(p1)
  const x=Ep/(nu*k0),x2=x*x
  const lat=p1-(nu*t/(a*a/(1-e2*Math.sin(p1)*Math.sin(p1))))*(x2/2-x2*x2*(5+3*t2+ep2-9*t2*ep2)/24)
  const lon=oLon*Math.PI/180+(x-x2*x*(1+2*t2+ep2)/6)/Math.cos(p1)
  return{lat:lat*180/Math.PI,lng:lon*180/Math.PI}
}
function haversine(a,b,c,d){
  const R=6371000,dL=(c-a)*Math.PI/180,dN=(d-b)*Math.PI/180
  const e=Math.sin(dL/2)**2+Math.cos(a*Math.PI/180)*Math.cos(c*Math.PI/180)*Math.sin(dN/2)**2
  return R*2*Math.atan2(Math.sqrt(e),Math.sqrt(1-e))
}

async function fetchCarparks(lat, lng, radiusM) {
  // Fetch directly from browser — no CORS issues on GitHub Pages
  let availMap = {}
  try {
    const r = await fetch('https://api.data.gov.sg/v1/transport/carpark-availability')
    const d = await r.json()
    if (d.items?.[0]) {
      d.items[0].carpark_data.forEach(cp => {
        const total = cp.carpark_info.reduce((s,i)=>s+parseInt(i.total_lots||0),0)
        const avail = cp.carpark_info.reduce((s,i)=>s+parseInt(i.lots_available||0),0)
        availMap[cp.carpark_number] = {total, avail}
      })
      console.log(`[Carparks] Availability loaded: ${Object.keys(availMap).length} carparks`)
    }
  } catch(e) { console.warn('[Carparks] Availability fetch failed:', e) }

  let cpList = []
  try {
    const r = await fetch('https://data.gov.sg/api/action/datastore_search?resource_id=139a3035-e624-4f56-b63f-89ae28d4ae4c&limit=10000')
    const d = await r.json()
    if (d.result?.records) {
      cpList = d.result.records
      console.log(`[Carparks] Info loaded: ${cpList.length} records`)
    }
  } catch(e) { console.warn('[Carparks] Info fetch failed:', e) }

  if (cpList.length === 0) {
    // Fallback: ask AI to estimate carparks
    const prompt = `List 6 realistic HDB carparks within ${radiusM}m of Singapore coordinates (${lat}, ${lng}). Use your knowledge of Singapore carpark locations.
Return ONLY a JSON array: [{"car_park_no":"SK1","address":"BLK 123 STREET","car_park_type":"SURFACE CAR PARK","free_parking":"SUN & PH FR 7AM-10:30PM","night_parking":"YES","gantry_height":"2.1","distM":150,"availLots":null,"totalLots":null}]`
    const {text, provider} = await callAI(prompt, false, 'carparks-fallback')
    try {
      const result = parseJSON(text)
      return {carparks: Array.isArray(result)?result:[], provider:'AI estimate'}
    } catch { return {carparks:[], provider:'AI estimate'} }
  }

  // Filter by radius using SVY21 conversion
  const nearby = []
  for (const cp of cpList) {
    const x=parseFloat(cp.x_coord), y=parseFloat(cp.y_coord)
    if (isNaN(x)||isNaN(y)||x===0) continue
    const coords = svy21ToLatLng(y, x)
    const dist = haversine(lat, lng, coords.lat, coords.lng)
    if (dist <= radiusM) {
      const av = availMap[cp.car_park_no] || null
      nearby.push({
        car_park_no: cp.car_park_no,
        address: cp.address,
        car_park_type: cp.car_park_type,
        free_parking: cp.free_parking,
        night_parking: cp.night_parking,
        gantry_height: cp.gantry_height,
        distM: Math.round(dist),
        availLots: av?.avail ?? null,
        totalLots: av?.total ?? null,
      })
    }
  }
  nearby.sort((a,b)=>a.distM-b.distM)
  return {carparks: nearby.slice(0,10), provider:'data.gov.sg'}
}

async function fetchPlaces(address, lat, lng, carparks) {
  const cpSummary = carparks.slice(0,8).map((cp,i)=>
    `#${i+1} carpark_no="${cp.car_park_no}" address="${cp.address}" type="${cp.car_park_type}" free="${cp.free_parking}" night="${cp.night_parking}" distM=${cp.distM} availLots=${cp.availLots??'?'} totalLots=${cp.totalLots??'?'}`
  ).join('\n')

  const validNos = carparks.slice(0,8).map(cp=>cp.car_park_no).join(', ')

  const prompt = `You are a Singapore local food guide.

Location: "${address}" (${lat}, ${lng})

Real HDB carparks found nearby (use EXACTLY these carpark_no values):
${cpSummary||'No real carparks found — set carpark_no to null'}

Valid carpark_no values you MUST use: ${validNos||'none'}

Suggest 6 realistic affordable eating places (coffeeshops, hawker centres, food courts) near "${address}". 
For each place, assign the closest carpark from the list above using its EXACT carpark_no value.
Do NOT invent new carpark numbers. Only use: ${validNos||'null'}.

Return ONLY a JSON array (no markdown):
[{"name":"Name","type":"Coffeeshop","address":"BLK 123 Street","distance_m":120,"avg_spend":"$3-5 per pax","famous_for":"Kaya toast, Wonton mee","open_hours":"6am-10pm","crowd_level":"Moderate","emoji":"☕","tips":"Local tip","carpark_no":"EXACT_NO_FROM_LIST","carpark_walk_m":80}]`

  const {text, provider} = await callAI(prompt, false, 'places')
  const result = parseJSON(text)
  const places = Array.isArray(result)?result:result.places||[]
  // Validate carpark_no — must exist in our real carpark list
  const validSet = new Set(carparks.map(cp=>cp.car_park_no))
  places.forEach(p => {
    if (p.carpark_no && !validSet.has(p.carpark_no)) {
      console.warn(`[Places] AI used invalid carpark_no: ${p.carpark_no}, clearing`)
      p.carpark_no = carparks[0]?.car_park_no || null
    }
  })
  return {places, provider}
}

function PipeStep({icon,label,detail,state}) {
  const bg  = state==='active'?'rgba(232,160,69,0.07)':state==='done'?'rgba(90,158,111,0.07)':state==='error'?'rgba(196,92,42,0.07)':'transparent'
  const col = state==='active'?C.accent:state==='done'?C.green:state==='error'?C.red:C.muted
  return (
    <div style={{background:bg,color:col,padding:'9px 14px',borderBottom:`1px solid ${C.border}`,display:'flex',alignItems:'center',gap:10,transition:'all 0.3s',fontSize:10,letterSpacing:1,textTransform:'uppercase'}}>
      <span style={{fontSize:14,flexShrink:0}}>{icon}</span>
      <div style={{flex:1}}>
        <div style={{display:'flex',alignItems:'center',gap:8}}>{label}{state==='active'&&<span className="spinner"/>}{state==='done'&&<span>✓</span>}{state==='error'&&<span>✗</span>}</div>
        <div style={{fontSize:9,opacity:0.7,marginTop:1}}>{detail}</div>
      </div>
    </div>
  )
}

function Badge({color,children}) {
  const s={green:{bg:'rgba(90,158,111,0.15)',col:C.green,b:'1px solid rgba(90,158,111,0.3)'},orange:{bg:'rgba(232,160,69,0.12)',col:C.accent,b:'1px solid rgba(232,160,69,0.25)'},blue:{bg:'rgba(74,144,184,0.12)',col:C.blue,b:'1px solid rgba(74,144,184,0.25)'},gray:{bg:'rgba(122,112,96,0.15)',col:C.muted,b:`1px solid ${C.border}`}}[color]||{bg:'rgba(122,112,96,0.15)',col:C.muted,b:`1px solid ${C.border}`}
  return <span style={{background:s.bg,color:s.col,border:s.b,fontSize:9,letterSpacing:1,textTransform:'uppercase',padding:'2px 6px',borderRadius:2}}>{children}</span>
}

function PlaceCard({p,i}) {
  const cp=p.cp
  const avail=cp?.availLots??null, total=cp?.totalLots??null
  const pct=(avail!=null&&total)?Math.round(avail/total*100):0
  const barCol=pct>40?C.green:pct>15?C.accent:C.red
  const mapUrl=`https://www.google.com/maps/search/${encodeURIComponent(p.name+' '+p.address+' Singapore')}`
  const cpUrl=cp?`https://www.google.com/maps/search/${encodeURIComponent(cp.address+' Singapore carpark')}`:''
  const btn={fontFamily:"'DM Mono',monospace",fontSize:10,letterSpacing:1,color:C.accent,background:'transparent',border:`1px solid rgba(232,160,69,0.3)`,padding:'4px 9px',borderRadius:3,cursor:'pointer',textTransform:'uppercase'}
  return (
    <div className="fade-card" style={{animationDelay:`${i*0.07}s`,background:C.card,border:`1px solid ${C.border}`,borderRadius:6,overflow:'hidden'}}>
      <div style={{padding:'13px 15px 11px',borderBottom:`1px solid ${C.border}`,display:'flex',alignItems:'flex-start',gap:10}}>
        <div style={{fontSize:24,flexShrink:0,marginTop:2}}>{p.emoji||'🍽'}</div>
        <div style={{flex:1,minWidth:0}}>
          <div style={{fontFamily:"'Syne',sans-serif",fontWeight:700,fontSize:13,color:C.text,lineHeight:1.3,marginBottom:3}}>{p.name}</div>
          <span style={{fontSize:9,letterSpacing:2,textTransform:'uppercase',color:C.accent2,border:`1px solid ${C.accent2}`,borderRadius:2,padding:'1px 5px'}}>{p.type}</span>
        </div>
        <div style={{fontFamily:"'Syne',sans-serif",fontWeight:700,fontSize:14,color:C.green,flexShrink:0}}>{(p.avg_spend||'').split(' ')[0]}</div>
      </div>
      <div style={{padding:'11px 15px'}}>
        {[{icon:'📍',val:p.address},{icon:'🍽',val:p.famous_for},{icon:'🕐',val:`${p.open_hours} · ${p.crowd_level==='Quiet'?'😌':p.crowd_level==='Busy'?'🔥':'👥'} ${p.crowd_level}`}].map((r,idx)=>(
          <div key={idx} style={{display:'flex',alignItems:'flex-start',gap:7,marginBottom:8,fontSize:11,lineHeight:1.5}}>
            <span style={{fontSize:12,flexShrink:0,marginTop:1}}>{r.icon}</span>
            <span style={{color:C.text,flex:1}}>{r.val}</span>
          </div>
        ))}
        <div style={{display:'flex',alignItems:'flex-start',gap:7,marginBottom:8,fontSize:11,lineHeight:1.5}}>
          <span style={{fontSize:12,flexShrink:0,marginTop:1}}>🅿</span>
          {cp?(
            <div style={{color:C.text,flex:1}}>
              <strong>{cp.car_park_no}</strong> · {cp.address}
              <div style={{color:C.muted,fontSize:10}}>{cp.car_park_type} · ~{p.carpark_walk_m}m walk · {cp.distM}m from search</div>
              {avail!=null&&total?(
                <div style={{marginTop:5}}>
                  <div style={{display:'flex',justifyContent:'space-between',fontSize:10,color:C.muted,marginBottom:3}}>
                    <span>{avail}/{total} lots free ({pct}%)</span>
                    {cp.night_parking==='YES'&&<span style={{color:C.blue}}>🌙 Night OK</span>}
                  </div>
                  <div style={{height:3,background:C.border,borderRadius:2,overflow:'hidden'}}>
                    <div style={{height:'100%',width:`${pct}%`,background:barCol,borderRadius:2,transition:'width 0.8s ease'}}/>
                  </div>
                </div>
              ):<div style={{fontSize:10,color:C.muted,marginTop:3}}>{cp.night_parking==='YES'?'🌙 Night OK · ':''}Availability not in dataset</div>}
            </div>
          ):<span style={{color:C.muted,fontSize:10}}>No HDB carpark matched nearby</span>}
        </div>
        {p.tips&&<div style={{display:'flex',gap:7,marginBottom:8,fontSize:10,color:C.muted,fontStyle:'italic',lineHeight:1.5}}><span>💡</span><span>{p.tips}</span></div>}
        <div style={{display:'flex',gap:5,flexWrap:'wrap',marginTop:9}}>
          {cp?.car_park_type?.includes('SURFACE')&&<Badge color="green">🅿 Surface</Badge>}
          {cp?.car_park_type?.includes('MULTI')&&<Badge color="blue">🏢 Multi-storey</Badge>}
          {cp?.car_park_type?.includes('BASEMENT')&&<Badge color="gray">🏠 Basement</Badge>}
          {cp?.free_parking&&cp.free_parking!=='NO'&&<Badge color="green">💚 Free</Badge>}
          {cp?.night_parking==='YES'&&<Badge color="blue">🌙 Night</Badge>}
          {cp?.gantry_height&&<Badge color="gray">↕{cp.gantry_height}m</Badge>}
          <Badge color="orange">~{p.avg_spend}</Badge>
        </div>
      </div>
      <div style={{padding:'9px 15px',borderTop:`1px solid ${C.border}`,display:'flex',alignItems:'center',justifyContent:'space-between'}}>
        <div style={{fontSize:10,color:C.muted}}>~<strong style={{color:C.accent}}>{p.distance_m}m</strong> walk</div>
        <div style={{display:'flex',gap:5}}>
          {cp&&<button onClick={()=>window.open(cpUrl,'_blank')} style={btn}>🅿 Park</button>}
          <button onClick={()=>window.open(mapUrl,'_blank')} style={btn}>📍 Maps</button>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [query,setQuery]=useState('')
  const [radius,setRadius]=useState(600)
  const [loading,setLoading]=useState(false)
  const [allResults,setAllResults]=useState([])
  const [locLabel,setLocLabel]=useState('')
  const [providers,setProviders]=useState({})
  const [error,setError]=useState('')
  const [activeFilters,setActiveFilters]=useState(new Set(['all']))
  const [pipe,setPipe]=useState({
    s1:{state:'',detail:'Gemini + Google Search → OneMap, Groq fallback'},
    s2:{state:'',detail:'Live HDB carpark availability'},
    s3:{state:'',detail:'AI matches eating places to carparks'},
    s4:{state:'',detail:'All done'},
  })

  const updPipe=(key,state,detail)=>setPipe(p=>({...p,[key]:{state,detail:detail??p[key].detail}}))

  const toggleFilter=(f)=>setActiveFilters(prev=>{
    const n=new Set(prev)
    if(f==='all') return new Set(['all'])
    n.delete('all')
    if(n.has(f)) n.delete(f); else n.add(f)
    return n.size===0?new Set(['all']):n
  })

  const filtered=allResults.filter(p=>{
    if(activeFilters.has('all')) return true
    if(activeFilters.has('coffeeshop')&&p.type==='Coffeeshop') return true
    if(activeFilters.has('hawker')&&p.type==='Hawker Centre') return true
    if(activeFilters.has('foodcourt')&&p.type==='Food Court') return true
    if(activeFilters.has('surface')&&p.cp?.car_park_type?.includes('SURFACE')) return true
    if(activeFilters.has('free')&&p.cp?.free_parking&&p.cp.free_parking!=='NO') return true
    if(activeFilters.has('available')&&(p.cp?.availLots??0)>10) return true
    return false
  })

  const doSearch=useCallback(async()=>{
    const q=query.trim()
    if(!q){setError('Please enter a location or postal code.');return}
    if(!GEMINI_KEY&&!GROQ_KEY){setError('No API keys configured. Add VITE_GEMINI_API_KEY and/or VITE_GROQ_API_KEY in GitHub Secrets.');return}
    setError('');setLocLabel('');setAllResults([]);setProviders({})
    setLoading(true)
    setPipe({
      s1:{state:'active',detail:'Resolving coordinates…'},
      s2:{state:'',detail:'Live HDB carpark availability'},
      s3:{state:'',detail:'AI matches eating places to carparks'},
      s4:{state:'',detail:'All done'},
    })
    try {
      let loc
      try {
        loc=await geocodeLocation(q)
        updPipe('s1','done',`${loc.address} (${Number(loc.lat).toFixed(5)}, ${Number(loc.lng).toFixed(5)}) · ${loc.provider}`)
        setLocLabel(`${loc.address}${loc.postal&&loc.postal!=='null'?' · '+loc.postal:''}`)
        setProviders(p=>({...p,geocode:loc.provider}))
      } catch(e) { updPipe('s1','error',e.message); throw new Error('Geocode failed: '+e.message) }

      updPipe('s2','active','Fetching nearby HDB carparks…')
      let carparks=[]
      try {
        const res=await fetchCarparks(loc.lat,loc.lng,radius)
        carparks=res.carparks
        updPipe('s2','done',`${carparks.length} carparks found · ${res.provider}`)
        setProviders(p=>({...p,carparks:res.provider}))
        if(!carparks.length) setError(`No HDB carparks found within ${radius}m. Try a larger radius.`)
      } catch(e) { updPipe('s2','error','Carpark fetch failed — AI will estimate'); carparks=[] }

      updPipe('s3','active','Matching eating places to carparks…')
      let places=[]
      try {
        const res=await fetchPlaces(loc.address,loc.lat,loc.lng,carparks)
        places=res.places
        updPipe('s3','done',`${places.length} places matched · ${res.provider}`)
        setProviders(p=>({...p,places:res.provider}))
      } catch(e) { updPipe('s3','error',e.message); throw new Error('Place matching failed: '+e.message) }

      const cpMap={}
      carparks.forEach(cp=>{cpMap[cp.car_park_no]=cp})
      const merged=places.map(p=>({...p,cp:cpMap[p.carpark_no]||null})).sort((a,b)=>a.distance_m-b.distance_m)
      setAllResults(merged)
      updPipe('s4','done',`Showing ${merged.length} results`)
    } catch(e) { console.error(e); setError(e.message) }
    setLoading(false)
  },[query,radius])

  const FILTERS=[
    {k:'all',l:'All'},{k:'coffeeshop',l:'☕ Coffeeshop'},{k:'hawker',l:'🍜 Hawker'},
    {k:'foodcourt',l:'🏪 Food Court'},{k:'surface',l:'🅿 Surface Lot'},
    {k:'free',l:'💚 Free Parking'},{k:'available',l:'✅ Lots Free'},
  ]
  const usingGroq=Object.values(providers).includes('Groq')

  return (
    <>
      <style>{GLOBAL_CSS}</style>
      <div style={{minHeight:'100vh',background:C.bg,color:C.text}}>
        <div style={{padding:'28px 20px 14px',borderBottom:`1px solid ${C.border}`}}>
          <div style={{display:'flex',alignItems:'flex-end',gap:10,marginBottom:4}}>
            <span style={{fontSize:28}}>☕</span>
            <h1 style={{fontFamily:"'Syne',sans-serif",fontSize:'clamp(22px,6vw,40px)',fontWeight:800,color:C.accent,letterSpacing:-1,lineHeight:1}}>Kopitiam Finder</h1>
          </div>
          <div style={{fontSize:10,color:C.muted,letterSpacing:3,textTransform:'uppercase'}}>Accurate geocoding · Real HDB carpark data · Singapore</div>
        </div>

        <div style={{padding:'18px 20px 0'}}>
          <div style={{fontSize:10,letterSpacing:3,textTransform:'uppercase',color:C.muted,marginBottom:8}}>Location · Postal Code · Address</div>
          <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
            <div style={{flex:1,minWidth:180,position:'relative'}}>
              <span style={{position:'absolute',left:10,top:'50%',transform:'translateY(-50%)',color:C.muted,fontSize:14,pointerEvents:'none'}}>📍</span>
              <input type="text" value={query} onChange={e=>setQuery(e.target.value)} onKeyDown={e=>e.key==='Enter'&&!loading&&doSearch()}
                placeholder="550212 · Sengkang · Tampines MRT..."
                style={{width:'100%',background:C.surface,border:`1px solid ${C.border}`,color:C.text,fontSize:14,padding:'12px 12px 12px 36px',borderRadius:4}}/>
            </div>
            <select value={radius} onChange={e=>setRadius(Number(e.target.value))}
              style={{background:C.surface,border:`1px solid ${C.border}`,color:C.text,fontSize:12,padding:'12px 8px',borderRadius:4,cursor:'pointer'}}>
              {[400,600,800,1000,1500].map(r=><option key={r} value={r}>{r>=1000?r/1000+'km':r+'m'}</option>)}
            </select>
            <button onClick={doSearch} disabled={loading}
              style={{background:loading?C.muted:C.accent,color:'#0f0e0c',border:'none',fontFamily:"'Syne',sans-serif",fontWeight:700,fontSize:13,letterSpacing:1,padding:'12px 20px',borderRadius:4,cursor:loading?'not-allowed':'pointer',opacity:loading?0.7:1,whiteSpace:'nowrap'}}>
              {loading?'Searching…':'Search'}
            </button>
          </div>
        </div>

        <div style={{margin:'14px 20px 0',border:`1px solid ${C.border}`,borderRadius:4,overflow:'hidden'}}>
          <PipeStep icon="🗺" label="1 · Geocode"      detail={pipe.s1.detail} state={pipe.s1.state}/>
          <PipeStep icon="🅿" label="2 · HDB Carparks" detail={pipe.s2.detail} state={pipe.s2.state}/>
          <PipeStep icon="🤖" label="3 · Place Match"  detail={pipe.s3.detail} state={pipe.s3.state}/>
          <PipeStep icon="✅" label="4 · Results"      detail={pipe.s4.detail} state={pipe.s4.state}/>
        </div>

        <div style={{display:'flex',gap:6,flexWrap:'wrap',margin:'12px 20px 0'}}>
          {FILTERS.map(f=>(
            <button key={f.k} onClick={()=>toggleFilter(f.k)}
              style={{background:activeFilters.has(f.k)?'rgba(232,160,69,0.08)':'transparent',border:`1px solid ${activeFilters.has(f.k)?C.accent:C.border}`,color:activeFilters.has(f.k)?C.accent:C.muted,fontSize:10,letterSpacing:1,padding:'5px 10px',borderRadius:100,cursor:'pointer',textTransform:'uppercase'}}>
              {f.l}
            </button>
          ))}
        </div>

        {locLabel&&<div style={{margin:'10px 20px 0',background:'rgba(74,144,184,0.08)',border:`1px solid rgba(74,144,184,0.25)`,borderRadius:4,padding:'9px 12px',fontSize:11,color:C.blue}}>
          📌 <strong>{locLabel}</strong>
          {usingGroq&&<span style={{marginLeft:10,fontSize:10,color:C.muted}}>· Groq fallback active</span>}
        </div>}
        {error&&<div style={{margin:'10px 20px 0',background:'rgba(196,92,42,0.1)',border:`1px solid rgba(196,92,42,0.3)`,borderRadius:4,padding:'9px 12px',fontSize:11,color:'#e07050'}}>⚠ {error}</div>}

        <div style={{padding:'16px 20px 48px'}}>
          {loading?(
            <div style={{textAlign:'center',padding:'48px 20px',color:C.muted}}>
              <div className="dot-loader"><span/><span/><span/></div>
              <div style={{fontFamily:"'Syne',sans-serif",fontSize:16,color:C.text,marginTop:14,marginBottom:8}}>Searching Singapore…</div>
              <div style={{fontSize:11,lineHeight:2.0}}>
                Gemini 2.5 Flash Lite + Google Search (primary)<br/>
                Groq Llama 3.3 70B (auto-fallback if rate limited)<br/>
                <span style={{color:C.accent2,fontSize:10}}>~15–25 seconds</span>
              </div>
            </div>
          ):filtered.length>0?(
            <>
              <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',paddingBottom:12,borderBottom:`1px solid ${C.border}`,marginBottom:14}}>
                <div style={{fontSize:11,color:C.muted,letterSpacing:2,textTransform:'uppercase'}}><span style={{color:C.accent}}>{filtered.length}</span> places near {query}</div>
                <div style={{fontSize:10,color:C.muted}}>sorted by distance</div>
              </div>
              <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(280px,1fr))',gap:12}}>
                {filtered.map((p,i)=><PlaceCard key={i} p={p} i={i}/>)}
              </div>
            </>
          ):!loading&&allResults.length===0&&(
            <div style={{textAlign:'center',padding:'48px 20px',color:C.muted}}>
              <div style={{fontSize:40,marginBottom:12}}>🗺️</div>
              <div style={{fontFamily:"'Syne',sans-serif",fontSize:16,color:C.text,marginBottom:10}}>Find your next kopi session</div>
              <div style={{fontSize:11,lineHeight:2.0}}>
                Enter a postal code, MRT, or area name<br/>
                <strong style={{color:C.accent}}>550212</strong> · <strong style={{color:C.accent}}>Sengkang</strong> · <strong style={{color:C.accent}}>Tampines MRT</strong><br/>
                <span style={{fontSize:10}}>Gemini primary · Groq fallback · Always on</span>
              </div>
            </div>
          )}
        </div>

        <div style={{fontSize:10,color:C.muted,textAlign:'center',padding:10,borderTop:`1px solid ${C.border}`,letterSpacing:1}}>
          <span style={{color:C.accent2}}>Gemini 2.5 Flash Lite</span> · <span style={{color:C.accent2}}>Groq Llama 3.3 70B</span> · <span style={{color:C.accent2}}>HDB data.gov.sg</span> · <span style={{color:C.accent2}}>OneMap SG</span>
        </div>
      </div>
    </>
  )
}
