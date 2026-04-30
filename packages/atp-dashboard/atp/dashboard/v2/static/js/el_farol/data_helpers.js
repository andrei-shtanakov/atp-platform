// ATP · El Farol · data helpers.
//
// Takes the server-injected `window.__ATP_MATCH__` payload (shape
// produced by /api/v1/games/{match_id}/dashboard, Pydantic model
// DashboardPayload, shape_version=1) and exposes `window.ATP` with the
// helpers the dashboard renderer (`dashboard.js`) reads from.
//
// The renderer here was derived from the standalone mockup at
// docs/mockups/el-farol/cards.html (LABS-97). The mockup generated its
// own data via a seeded mulberry32 RNG — that path is deleted; real
// matches arrive via the server payload only.
//
// Any shape change is a client-breaking change — bump both
// SHAPE_VERSION in el_farol_dashboard.py and the comparison here.
(function () {
  const EXPECTED_SHAPE_VERSION = 1;

  if (!window.__ATP_MATCH__) {
    throw new Error("ATP: window.__ATP_MATCH__ must be set before data_helpers.js");
  }
  const payload = window.__ATP_MATCH__;
  if (payload.shape_version !== EXPECTED_SHAPE_VERSION) {
    throw new Error(
      "ATP: unsupported shape_version=" + payload.shape_version +
      " (renderer expects " + EXPECTED_SHAPE_VERSION + ")"
    );
  }

  const AGENTS = payload.AGENTS;
  const NUM_SLOTS = payload.NUM_SLOTS;
  const MAX_TOTAL = payload.MAX_TOTAL;
  const CAPACITY = payload.CAPACITY;
  const NUM_DAYS = payload.NUM_DAYS;
  const WEEK_LEN = payload.WEEK_LEN || 10;
  const DATA = payload.DATA;

  function slotsInIntervals(ivs) {
    const s = [];
    ivs.forEach(iv => {
      if (!iv || iv.length === 0) return;
      for (let i = iv[0]; i <= iv[1]; i++) s.push(i);
    });
    return [...new Set(s)].sort((a, b) => a - b);
  }
  function nonEmptyIntervals(ivs) {
    return ivs.filter(iv => iv && iv.length === 2);
  }

  function cumPayoff(ai, through) {
    let s = 0;
    for (let r = 0; r < through; r++) s += DATA[r].decisions[ai].payoff;
    return s;
  }
  function cumPayoffs(through) {
    return AGENTS.map((_, i) => cumPayoff(i, through));
  }
  function cumSeries(ai) {
    const arr = [];
    let s = 0;
    for (let r = 0; r < DATA.length; r++) {
      s += DATA[r].decisions[ai].payoff;
      arr.push(s);
    }
    return arr;
  }
  function avgVisitsPerDay(ai, through) {
    let t = 0;
    for (let r = 0; r < through; r++) t += DATA[r].decisions[ai].numVisits;
    return through ? t / through : 0;
  }
  function leaderboard(through) {
    const entries = AGENTS.map((a, i) => ({
      idx: i,
      id: a.id,
      color: a.color,
      profile: a.profile,
      user: a.user,
      payoff: cumPayoff(i, through),
      visits: avgVisitsPerDay(i, through),
    }));
    entries.sort((a, b) => b.payoff - a.payoff || a.id.localeCompare(b.id));
    return entries.map((e, rank) => ({ ...e, rank: rank + 1 }));
  }

  window.ATP = {
    AGENTS, NUM_SLOTS, MAX_TOTAL, CAPACITY, NUM_DAYS, WEEK_LEN,
    DATA, cumPayoff, cumPayoffs, cumSeries, avgVisitsPerDay, leaderboard,
    nonEmptyIntervals, slotsInIntervals,
  };
})();
