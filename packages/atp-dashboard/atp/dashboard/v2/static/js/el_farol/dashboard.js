// ATP · El Farol · Cards dashboard (from docs/mockups/el-farol/cards.html).
//
// Expects window.ATP prepared by data_helpers.js and the match DOM
// skeleton from templates/ui/match_detail.html. Runs immediately on
// load; event listeners are attached to elements in the skeleton.
//
// Source: LABS-97 mockup. The only production adaptations are:
//   * IIFE wrap (no module-level globals except window.togglePlay /
//     step / jump / openDrawer / closeDrawer that the inline onclick
//     handlers need).
//   * Per-match localStorage keys (scrubber position + pinned agents).
//   * Day clamp uses NUM_DAYS instead of the mockup's 100 constant.
(function () {
// ATP · El Farol · Cards dashboard
const { AGENTS, NUM_SLOTS, CAPACITY, NUM_DAYS, WEEK_LEN, DATA, cumPayoff, cumSeries, leaderboard, nonEmptyIntervals } = window.ATP;

let currentDay = Math.min(42, typeof NUM_DAYS === 'number' ? NUM_DAYS : 42);
let playing = false; let playTimer = null;
let playSpeed = 380;
let heatMode = 'slot-day';
let pinnedCompare = [];
let hoverCompare = null;
let identityColorsOn = true;

const $ = (id) => document.getElementById(id);
function agentColor(i) { return identityColorsOn ? AGENTS[i].color : '#9aa3b2'; }

const LS_KEY_DAY = 'atp-day-' + (window.__ATP_MATCH__ ? window.__ATP_MATCH__.match_id : 'default');
const LS_KEY_PINS = 'atp-pinned-' + (window.__ATP_MATCH__ ? window.__ATP_MATCH__.match_id : 'default');
try { const d = parseInt(localStorage.getItem(LS_KEY_DAY)||'1', 10); if (d>=1 && d<=NUM_DAYS) currentDay = d; } catch(e){}
try { const p = JSON.parse(localStorage.getItem(LS_KEY_PINS)||'[]'); if (Array.isArray(p)) pinnedCompare = p.filter(x => x>=0 && x<AGENTS.length); } catch(e){}

/* ---------- playback ---------- */
function togglePlay() {
  playing = !playing;
  const btn = $('playBtn');
  if (playing) { btn.textContent = '⏸ Pause'; playTimer = setInterval(() => { if (currentDay < NUM_DAYS) { currentDay++; renderAll(); } else togglePlay(); }, playSpeed); }
  else { btn.textContent = '▶ Play'; clearInterval(playTimer); }
}
function step(d) { currentDay = Math.max(1, Math.min(NUM_DAYS, currentDay + d)); renderAll(); }
function jump(d) { currentDay = d; renderAll(); }
window.togglePlay = togglePlay; window.step = step; window.jump = jump;
$('scrubber').addEventListener('input', e => { currentDay = parseInt(e.target.value); renderAll(); });
$('speedSel').addEventListener('change', e => { playSpeed = parseInt(e.target.value); if (playing) { clearInterval(playTimer); playing = false; togglePlay(); } });

/* ---------- popovers ---------- */
$('rulesBtn').addEventListener('click', e => { e.stopPropagation(); $('rulesPop').classList.toggle('open'); });
$('tweaksBtn').addEventListener('click', e => { e.stopPropagation(); $('tweaksPop').classList.toggle('open'); });
document.addEventListener('click', (e) => {
  if (!$('rulesPop').contains(e.target) && e.target !== $('rulesBtn')) $('rulesPop').classList.remove('open');
  if (!$('tweaksPop').contains(e.target) && e.target !== $('tweaksBtn')) $('tweaksPop').classList.remove('open');
});

function renderRulesDiagram() {
  const rd = DATA[currentDay-1];
  $('rulesDayNum').textContent = currentDay;
  const el = $('rulesDiagram'); el.innerHTML = '';
  for (let s = 0; s < NUM_SLOTS; s++) {
    const att = rd.slotAttendance[s]; const over = att > CAPACITY; const empty = att === 0;
    const div = document.createElement('div');
    div.className = 's ' + (empty ? '' : (over ? 'over' : 'ok'));
    div.textContent = att;
    el.appendChild(div);
  }
}

/* ---------- tweaks ---------- */
$('tweakDensity').addEventListener('change', e => { const root = document.getElementById('atp-el-farol'); root.classList.remove('roomy','dense'); root.classList.add(e.target.value); renderAll(); });
$('tweakColors').addEventListener('click', e => { identityColorsOn = !identityColorsOn; e.target.classList.toggle('on', identityColorsOn); renderAll(); });
$('tweakRules').addEventListener('click', e => { e.target.classList.toggle('on'); $('rulesBtn').style.display = e.target.classList.contains('on') ? '' : 'none'; });

/* ---------- heatmap segmented control ---------- */
$('heatSeg').querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
  heatMode = b.dataset.m;
  $('heatSeg').querySelectorAll('button').forEach(x => x.classList.toggle('on', x===b));
  const subs = {
    'slot-day': 'slot × day · darker red = over capacity · click to jump',
    'agent-day': 'agent × day · green = good day, red = bad · click a cell to inspect',
    'agent-slot': 'agent × slot · how often each agent picks each slot'
  };
  $('heatSub').textContent = subs[heatMode];
  renderHeatmap();
}));

/* ---------- KPI strip ---------- */
function renderKPIs() {
  const rd = DATA[currentDay-1];
  const lb = leaderboard(currentDay);
  $('kpiLeader').textContent = lb[0].id;
  $('kpiLeaderSub').innerHTML = `+${lb[0].payoff} payoff`;
  $('kpiSpread').textContent = (lb[0].payoff - lb[lb.length-1].payoff);
  $('kpiOver').textContent = rd.overSlots + ' / 16';
  $('kpiAtt').textContent = rd.slotAttendance.reduce((a,b)=>a+b,0);
  let bestV = -999, bestA = null, bestD = 0;
  for (let r = 0; r < currentDay; r++) for (let ai = 0; ai < AGENTS.length; ai++) {
    const p = DATA[r].decisions[ai].payoff; if (p > bestV) { bestV = p; bestA = ai; bestD = r+1; }
  }
  $('kpiBest').textContent = '+' + bestV; $('kpiBestSub').textContent = `${AGENTS[bestA].id} · d${bestD}`;
  $('dayLabel').textContent = currentDay; $('weekLabel').textContent = 'w' + Math.ceil(currentDay/WEEK_LEN);
  $('scrubber').value = currentDay;
  try { localStorage.setItem(LS_KEY_DAY, String(currentDay)); } catch(e){}
}

/* ---------- sparkline ---------- */
function sparklineSVG(series, w, h, color, yMin, yMax) {
  const pad = 3;
  const max = yMax !== undefined ? yMax : Math.max(...series, 1);
  const min = yMin !== undefined ? yMin : Math.min(...series, 0);
  const range = (max - min) || 1;
  const n = series.length;
  const pts = series.map((v, i) => {
    const x = pad + (i/(n-1 || 1)) * (w - pad*2);
    const y = pad + (1 - (v - min)/range) * (h - pad*2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  let extra = '';
  if (min < 0 && max > 0) {
    const zy = pad + (1 - (0 - min)/range) * (h - pad*2);
    extra += `<line x1="${pad}" x2="${w-pad}" y1="${zy}" y2="${zy}" stroke="#2a2f39" stroke-width="0.5" stroke-dasharray="2,2"/>`;
  }
  const endXY = pts[pts.length-1].split(',');
  return `<svg class="acard-chart" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">${extra}<polyline points="${pts.join(' ')}" fill="none" stroke="${color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><circle cx="${endXY[0]}" cy="${endXY[1]}" r="2.5" fill="${color}"/></svg>`;
}

/* ---------- cards ---------- */
function renderCards() {
  const lb = leaderboard(currentDay);
  const rd = DATA[currentDay-1];
  const prev = currentDay > 1 ? leaderboard(currentDay-1) : null;
  const prevRank = prev ? Object.fromEntries(prev.map(e => [e.idx, e.rank])) : {};
  // global y-range so sparklines compare fairly
  let gMin = 0, gMax = 0;
  for (let ai = 0; ai < AGENTS.length; ai++) {
    const s = cumSeries(ai);
    for (let r = 0; r < currentDay; r++) { if (s[r] > gMax) gMax = s[r]; if (s[r] < gMin) gMin = s[r]; }
  }
  const html = lb.map(e => {
    const d = rd.decisions[e.idx];
    const delta = prev ? prevRank[e.idx] - e.rank : 0;
    const sym = delta > 0 ? `▲${delta}` : delta < 0 ? `▼${-delta}` : '─';
    const todayCls = d.payoff > 0 ? 'good' : d.payoff < 0 ? 'bad' : '';
    const series = cumSeries(e.idx).slice(0, currentDay);
    const spark = sparklineSVG(series, 200, 56, agentColor(e.idx), gMin, gMax);
    let pips = '';
    for (let s = 0; s < NUM_SLOTS; s++) {
      const picked = d.picks.includes(s);
      const att = rd.slotAttendance[s];
      let c = 'c';
      if (picked) c += (att > CAPACITY ? ' b' : ' g');
      pips += `<div class="${c}" title="slot ${s}: ${att}/${AGENTS.length}${picked?' · picked':''}"></div>`;
    }
    const pinned = pinnedCompare.includes(e.idx);
    return `<div class="acard ${e.rank===1?'rank1':''}" data-agent="${e.idx}">
      <button class="acard-compare ${pinned?'pinned':''}" data-pin="${e.idx}" title="${pinned?'Unpin':'Pin to compare'}">${pinned?'✓':'+'}</button>
      <div class="acard-head">
        <div class="name"><span class="sw" style="background:${agentColor(e.idx)}"></span><span class="id">${e.id}</span></div>
        <span class="rank-pill">#${e.rank} ${sym}</span>
      </div>
      <div class="payoff ${e.payoff>0?'good':e.payoff<0?'bad':''}">${e.payoff>0?'+':''}${e.payoff}</div>
      <div class="profile">user · ${e.user}</div>
      ${spark}
      <div class="acard-stats">
        <span>today <b class="${todayCls}">${d.payoff>0?'+':''}${d.payoff}</b></span>
        <span>visits/d <b>${e.visits.toFixed(2)}</b></span>
      </div>
      <div class="acard-today">${pips}</div>
    </div>`;
  }).join('');
  $('cards').innerHTML = html;

  // wire card interactions
  $('cards').querySelectorAll('.acard').forEach(c => {
    const ai = parseInt(c.dataset.agent);
    c.addEventListener('click', e => {
      if (e.target.closest('.acard-compare')) return;
      openDrawer(ai, currentDay);
    });
    c.addEventListener('mouseenter', () => { hoverCompare = ai; renderCompare(); });
    c.addEventListener('mouseleave', () => { hoverCompare = null; renderCompare(); });
  });
  $('cards').querySelectorAll('.acard-compare').forEach(b => {
    b.addEventListener('click', e => {
      e.stopPropagation();
      const ai = parseInt(b.dataset.pin);
      const i = pinnedCompare.indexOf(ai);
      if (i >= 0) pinnedCompare.splice(i, 1);
      else { pinnedCompare.push(ai); if (pinnedCompare.length > 4) pinnedCompare.shift(); }
      try { localStorage.setItem(LS_KEY_PINS, JSON.stringify(pinnedCompare)); } catch(e){}
      renderCards();
      renderCompare();
    });
  });
}

/* ---------- heatmap ---------- */
function heatColor(att) {
  if (att === 0) return '#13151a';
  if (att <= CAPACITY) { const i = att/CAPACITY; return `rgba(74,222,128,${0.15 + i*0.55})`; }
  const over = att - CAPACITY;
  const i = Math.min(over/3, 1);
  return `rgba(248,113,113,${0.45 + i*0.5})`;
}

function renderHeatmap() {
  const svg = $('heatmap');
  const W = 900, H = 340;
  const PAD_L = 78, PAD_R = 14, PAD_T = 14, PAD_B = 32;
  const plotW = W - PAD_L - PAD_R, plotH = H - PAD_T - PAD_B;
  let out = '';

  if (heatMode === 'slot-day') {
    const cw = plotW / NUM_DAYS, ch = plotH / NUM_SLOTS;
    for (let s = 0; s < NUM_SLOTS; s++)
      out += `<text x="${PAD_L-6}" y="${PAD_T + s*ch + ch/2 + 3}" fill="#9aa3b2" font-size="9" text-anchor="end" font-family="JetBrains Mono">slot ${s}</text>`;
    for (let d = 0; d < NUM_DAYS; d++) {
      for (let s = 0; s < NUM_SLOTS; s++) {
        const att = DATA[d].slotAttendance[s];
        out += `<rect x="${PAD_L + d*cw}" y="${PAD_T + s*ch}" width="${cw-0.3}" height="${ch-0.3}" fill="${heatColor(att)}" data-day="${d+1}" class="hmcell-sd"><title>day ${d+1} · slot ${s} · ${att}/${AGENTS.length}${att>CAPACITY?' (over cap)':''}</title></rect>`;
      }
    }
    for (let d = 1; d <= NUM_DAYS; d++)
      if (d===1 || d%10===0) out += `<text x="${PAD_L + (d-1)*cw + cw/2}" y="${H-PAD_B+14}" fill="#6b7280" font-size="9" text-anchor="middle" font-family="JetBrains Mono">d${d}</text>`;
    const phX = PAD_L + (currentDay-1)*cw + cw/2;
    out += `<line x1="${phX}" x2="${phX}" y1="${PAD_T-2}" y2="${H-PAD_B+2}" stroke="#7aa7ff" stroke-width="1.5"/>`;
  } else if (heatMode === 'agent-day') {
    const cw = plotW / NUM_DAYS, ch = plotH / AGENTS.length;
    AGENTS.forEach((a, ai) => {
      out += `<text x="${PAD_L-6}" y="${PAD_T + ai*ch + ch/2 + 3}" fill="${agentColor(ai)}" font-size="10" text-anchor="end" font-family="JetBrains Mono">${a.id}</text>`;
    });
    for (let d = 0; d < NUM_DAYS; d++) {
      for (let ai = 0; ai < AGENTS.length; ai++) {
        const dec = DATA[d].decisions[ai];
        let c = '#13151a';
        if (dec.payoff > 0) c = `rgba(74,222,128,${0.2 + Math.min(dec.payoff,6)/8})`;
        else if (dec.payoff < 0) c = `rgba(248,113,113,${0.2 + Math.min(Math.abs(dec.payoff),6)/8})`;
        out += `<rect x="${PAD_L + d*cw}" y="${PAD_T + ai*ch}" width="${cw-0.3}" height="${ch-0.3}" fill="${c}" data-day="${d+1}" data-agent="${ai}" class="hmcell-ad"><title>${AGENTS[ai].id} · day ${d+1} · ${dec.payoff>0?'+':''}${dec.payoff}</title></rect>`;
      }
    }
    for (let d = 1; d <= NUM_DAYS; d++)
      if (d===1 || d%10===0) out += `<text x="${PAD_L + (d-1)*cw + cw/2}" y="${H-PAD_B+14}" fill="#6b7280" font-size="9" text-anchor="middle" font-family="JetBrains Mono">d${d}</text>`;
    const phX = PAD_L + (currentDay-1)*cw + cw/2;
    out += `<line x1="${phX}" x2="${phX}" y1="${PAD_T-2}" y2="${H-PAD_B+2}" stroke="#7aa7ff" stroke-width="1.5"/>`;
  } else { // agent-slot
    const cw = plotW / NUM_SLOTS, ch = plotH / AGENTS.length;
    // frequency agent × slot through currentDay
    const freq = AGENTS.map(() => new Array(NUM_SLOTS).fill(0));
    for (let d = 0; d < currentDay; d++) {
      for (let ai = 0; ai < AGENTS.length; ai++) {
        DATA[d].decisions[ai].picks.forEach(s => freq[ai][s]++);
      }
    }
    AGENTS.forEach((a, ai) => {
      out += `<text x="${PAD_L-6}" y="${PAD_T + ai*ch + ch/2 + 3}" fill="${agentColor(ai)}" font-size="10" text-anchor="end" font-family="JetBrains Mono">${a.id}</text>`;
    });
    for (let s = 0; s < NUM_SLOTS; s++) {
      for (let ai = 0; ai < AGENTS.length; ai++) {
        const p = freq[ai][s] / currentDay;
        const col = `rgba(122,167,255,${0.1 + p*0.85})`;
        out += `<rect x="${PAD_L + s*cw}" y="${PAD_T + ai*ch}" width="${cw-0.6}" height="${ch-0.6}" fill="${col}"><title>${AGENTS[ai].id} · slot ${s} · picked ${freq[ai][s]}/${currentDay} days (${Math.round(p*100)}%)</title></rect>`;
        if (p > 0.5) out += `<text x="${PAD_L + s*cw + cw/2}" y="${PAD_T + ai*ch + ch/2 + 3}" fill="#fff" font-size="9" text-anchor="middle" font-family="JetBrains Mono" font-weight="600">${Math.round(p*100)}</text>`;
      }
      out += `<text x="${PAD_L + s*cw + cw/2}" y="${H-PAD_B+14}" fill="#6b7280" font-size="9" text-anchor="middle" font-family="JetBrains Mono">${s}</text>`;
    }
  }

  svg.innerHTML = out;
  svg.querySelectorAll('.hmcell-sd').forEach(el => el.addEventListener('click', () => jump(parseInt(el.dataset.day))));
  svg.querySelectorAll('.hmcell-ad').forEach(el => el.addEventListener('click', () => openDrawer(parseInt(el.dataset.agent), parseInt(el.dataset.day))));
}

/* ---------- compare panel (compact) ---------- */
function renderCompare() {
  const el = $('compare');
  const showSet = [...pinnedCompare];
  if (hoverCompare !== null && !showSet.includes(hoverCompare)) showSet.push(hoverCompare);

  if (showSet.length === 0) {
    el.innerHTML = `<div class="compare-empty">
      <div class="big">⇢</div>
      <div>Hover a card to preview.<br>Click the <code>+</code> on a card to pin an agent here.</div>
      <div class="tiny">Pin up to 4 agents to overlay.</div>
    </div>`;
    return;
  }

  // header chips
  const chips = showSet.map(ai => {
    const pinned = pinnedCompare.includes(ai);
    return `<span class="compare-chip">
      <span class="sw" style="background:${agentColor(ai)}"></span>${AGENTS[ai].id}
      ${pinned ? `<span class="x" data-unpin="${ai}">×</span>` : `<span class="tiny muted" style="margin-left:4px">(hover)</span>`}
    </span>`;
  }).join('');

  // overlay chart
  const W = 600, H = 200;
  const PAD_L = 36, PAD_R = 12, PAD_T = 10, PAD_B = 22;
  const plotW = W - PAD_L - PAD_R, plotH = H - PAD_T - PAD_B;
  const series = showSet.map(ai => cumSeries(ai));
  let maxY = 0, minY = 0;
  series.forEach(s => s.forEach(v => { if (v > maxY) maxY = v; if (v < minY) minY = v; }));
  const range = (maxY - minY) || 1;
  const xOf = d => PAD_L + (d/(NUM_DAYS-1)) * plotW;
  const yOf = v => PAD_T + (1 - (v - minY)/range) * plotH;
  let chart = '';
  for (let gy = Math.ceil(minY/40)*40; gy <= maxY; gy += 40) {
    chart += `<line x1="${PAD_L}" x2="${W-PAD_R}" y1="${yOf(gy)}" y2="${yOf(gy)}" stroke="#2a2f39" stroke-width="0.5"/>`;
    chart += `<text x="${PAD_L-4}" y="${yOf(gy)+3}" fill="#6b7280" font-size="9" text-anchor="end" font-family="JetBrains Mono">${gy>=0?'+'+gy:gy}</text>`;
  }
  if (minY < 0 && maxY > 0) chart += `<line x1="${PAD_L}" x2="${W-PAD_R}" y1="${yOf(0)}" y2="${yOf(0)}" stroke="#6b7280" stroke-dasharray="3,3" stroke-width="0.8"/>`;
  const phX = xOf(currentDay-1);
  chart += `<line x1="${phX}" x2="${phX}" y1="${PAD_T}" y2="${H-PAD_B}" stroke="#7aa7ff" stroke-width="1" stroke-dasharray="2,2" opacity="0.6"/>`;
  series.forEach((s, i) => {
    const ai = showSet[i];
    const isHover = hoverCompare === ai && !pinnedCompare.includes(ai);
    const path = 'M ' + s.map((v, d) => `${xOf(d)} ${yOf(v)}`).join(' L ');
    chart += `<path d="${path}" stroke="${agentColor(ai)}" stroke-width="${isHover?2.5:1.8}" fill="none" opacity="${isHover?1:0.95}" stroke-dasharray="${isHover?'0':'0'}"/>`;
    const v = s[currentDay-1];
    chart += `<circle cx="${phX}" cy="${yOf(v)}" r="2.5" fill="${agentColor(ai)}" stroke="#13151a" stroke-width="1"/>`;
  });
  chart += `<text x="${phX+4}" y="${PAD_T+10}" fill="#7aa7ff" font-size="9" font-family="JetBrains Mono">d${currentDay}</text>`;

  // summary stats at current day
  const lb = leaderboard(currentDay);
  const rankOf = Object.fromEntries(lb.map(e => [e.idx, e.rank]));
  const statsHtml = showSet.map(ai => {
    const p = cumPayoff(ai, currentDay);
    const today = DATA[currentDay-1].decisions[ai].payoff;
    const rank = rankOf[ai];
    const todayCls = today > 0 ? 'good' : today < 0 ? 'bad' : '';
    const pCls = p > 0 ? 'good' : p < 0 ? 'bad' : '';
    return `<div class="stat">
      <div class="l"><span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:${agentColor(ai)};margin-right:4px"></span>${AGENTS[ai].id}</div>
      <div class="v" style="color: var(--${pCls === 'good' ? 'good' : pCls === 'bad' ? 'bad' : 'text'})">${p>0?'+':''}${p}</div>
      <div class="l" style="margin-top:4px">rank #${rank} · today <span class="${todayCls}" style="color: var(--${todayCls === 'good' ? 'good' : todayCls === 'bad' ? 'bad' : 'text-2'})">${today>0?'+':''}${today}</span></div>
    </div>`;
  }).join('');

  el.innerHTML = `<div class="compare-body">
    <div class="compare-head">${chips}</div>
    <svg class="compare-chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${chart}</svg>
    <div class="compare-stats">${statsHtml}</div>
  </div>`;

  el.querySelectorAll('[data-unpin]').forEach(x => x.addEventListener('click', () => {
    const ai = parseInt(x.dataset.unpin);
    const i = pinnedCompare.indexOf(ai);
    if (i >= 0) pinnedCompare.splice(i, 1);
    try { localStorage.setItem(LS_KEY_PINS, JSON.stringify(pinnedCompare)); } catch(e){}
    renderCards(); renderCompare();
  }));
}

/* ---------- drawer ---------- */
function openDrawer(agentIdx, day) {
  const a = AGENTS[agentIdx];
  const d = DATA[day-1].decisions[agentIdx];
  const rank = leaderboard(day).find(e => e.idx === agentIdx).rank;
  const nonEmpty = nonEmptyIntervals(d.intervals);
  $('drawTitle').textContent = `${a.id} · day ${day}`;
  $('drawSub').innerHTML = `rank #${rank} · user <b>${a.user}</b> · ${d.numVisits} visit${d.numVisits===1?'':'s'} · <span style="color:${d.payoff>0?'var(--good)':d.payoff<0?'var(--bad)':'var(--text-2)'}">${d.payoff>0?'+':''}${d.payoff} payoff</span> · cum ${cumPayoff(agentIdx, day)>=0?'+':''}${cumPayoff(agentIdx, day)}`;

  let breakdownBody = '';
  if (d.slotPayoffs.length === 0) breakdownBody = `<tr><td colspan="3" class="muted">no slots picked</td></tr>`;
  else {
    nonEmpty.forEach((iv, i) => {
      breakdownBody += `<tr class="group"><td colspan="3">visit ${i+1} · [${iv[0]}, ${iv[1]}] · ${iv[1]-iv[0]+1} slot(s)</td></tr>`;
      for (let s = iv[0]; s <= iv[1]; s++) {
        const sp = d.slotPayoffs.find(x => x.slot === s);
        breakdownBody += `<tr><td><span class="slot-chip">${s}</span></td><td>${sp.attendance}/${AGENTS.length}${sp.attendance>CAPACITY?' <span style="color:var(--bad)">over</span>':''}</td><td class="${sp.payoff>0?'g':'r'}">${sp.payoff>0?'+1':'−1'}</td></tr>`;
      }
    });
  }

  const visitsHtml = [0,1].map(i => {
    if (i < nonEmpty.length) {
      const iv = nonEmpty[i];
      const p = d.intervalPayoffs[i] ? d.intervalPayoffs[i].payoff : 0;
      return `<div class="visit-item ${i===1?'v2':''}"><span class="iv-label">visit ${i+1}</span><span class="mono">[${iv[0]}, ${iv[1]}]</span><span class="muted tiny">${iv[1]-iv[0]+1} slot${iv[1]-iv[0]+1===1?'':'s'}</span><span class="iv-payoff ${p>0?'good':p<0?'bad':''}">${p>0?'+':''}${p}</span></div>`;
    }
    return `<div class="visit-item empty"><span class="iv-label">visit ${i+1}</span><span>∅ not used</span></div>`;
  }).join('');

  $('drawBody').innerHTML = `
    <div class="drawer-section">
      <h4>per-slot breakdown</h4>
      <table class="breakdown">
        <thead><tr><th>slot</th><th>attendance</th><th>± payoff</th></tr></thead>
        <tbody>${breakdownBody}</tbody>
        <tfoot><tr><td colspan="2"><b>day total</b></td><td class="${d.payoff>0?'g':d.payoff<0?'r':''}"><b>${d.payoff>0?'+':''}${d.payoff}</b></td></tr></tfoot>
      </table>
    </div>
    <div class="drawer-section">
      <h4>visit plan</h4>
      <div class="visits-list">${visitsHtml}</div>
      <div class="mono tiny muted" style="margin-top:10px;background:var(--panel-2);padding:8px 10px;border-radius:4px">make_move(${JSON.stringify({intervals: d.intervals.map(iv => iv && iv.length ? iv : [])})})</div>
    </div>
    <div class="drawer-section">
      <h4>intent</h4>
      <div class="muted tiny" style="font-style:italic">"${d.intent}"</div>
    </div>
    <div class="drawer-section">
      <h4>debug · observability</h4>
      <div class="mono tiny muted" style="display:grid;grid-template-columns:130px 1fr;gap:4px 10px">
        <span>model_id</span><span>${d.model_id == null ? '—' : d.model_id}</span>
        <span>decide_ms</span><span>${d.decide_ms == null ? '—' : d.decide_ms + ' ms'}</span>
        <span>tokens</span><span>${d.tokens_in == null && d.tokens_out == null ? '—' : (d.tokens_in ?? '?') + ' in / ' + (d.tokens_out ?? '?') + ' out'}</span>
        <span>cost</span><span>${d.cost_usd == null ? '—' : '$' + d.cost_usd.toFixed(4)}</span>
        <span>retry_count</span><span>${d.retry_count == null || d.retry_count === 0 ? '—' : d.retry_count}</span>
        <span>validation_error</span><span>${d.validation_error == null ? '—' : d.validation_error}</span>
        <span>trace_id</span><span>${d.trace_id == null ? '—' : (window.ATP_LANGFUSE_BASE ? '<a href="' + window.ATP_LANGFUSE_BASE + '/traces/' + encodeURIComponent(d.trace_id) + '" target="_blank" rel="noopener">' + d.trace_id + '</a>' : d.trace_id)}</span>
      </div>
    </div>
  `;
  $('drawer').classList.add('open');
  $('overlay').classList.add('open');
}
function closeDrawer() { $('drawer').classList.remove('open'); $('overlay').classList.remove('open'); }
window.openDrawer = openDrawer; window.closeDrawer = closeDrawer;

/* ---------- render all ---------- */
function renderAll() {
  renderKPIs();
  renderRulesDiagram();
  renderCards();
  renderHeatmap();
  renderCompare();
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeDrawer();
  if (e.key === 'ArrowLeft') step(-1);
  if (e.key === 'ArrowRight') step(1);
  if (e.key === ' ' && !['INPUT','BUTTON','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) { e.preventDefault(); togglePlay(); }
});

renderAll();
})();
