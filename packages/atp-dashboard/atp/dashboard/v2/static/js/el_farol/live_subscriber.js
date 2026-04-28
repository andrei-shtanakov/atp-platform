// Live tournament dashboard subscriber.
//
// Opens a Server-Sent Events stream to /api/v1/tournaments/{id}/dashboard/stream
// and reloads the page when a new round-resolved snapshot arrives. Doing a
// full reload (rather than mutating window.ATP) keeps the heavy dashboard
// renderer (dashboard.js) untouched: it captures window.ATP at IIFE load
// time, so in-place updates would not propagate without a refactor.
//
// On the terminal `completed` event, the subscriber redirects to the
// canonical replay URL once the GameResult dual-write has landed.
(function () {
  if (!window.__ATP_LIVE_URL__) return;

  var url = window.__ATP_LIVE_URL__;
  var statusPill = document.getElementById('atp-live-pill');
  var lastSnapshotAt = 0;
  var reloadTimer = null;
  // The SSE generator always emits an initial snapshot on connect that
  // mirrors the server-rendered state already on the page. Reloading on
  // it would cause an infinite reload loop (each reload reopens the
  // stream, which re-emits the initial snapshot). Only reload on
  // snapshots that arrive *after* the first one.
  var sawInitialSnapshot = false;

  function setStatus(text, klass) {
    if (!statusPill) return;
    statusPill.textContent = text;
    statusPill.className = 'atp-live-pill ' + (klass || '');
  }

  function scheduleReload(delayMs) {
    if (reloadTimer) return; // already pending
    reloadTimer = setTimeout(function () {
      // Cache-bust to defeat any intermediate caches
      var u = new URL(window.location.href);
      u.searchParams.set('_ts', Date.now().toString());
      window.location.replace(u.toString());
    }, delayMs);
  }

  var es = new EventSource(url);

  es.addEventListener('open', function () { setStatus('Live', 'live'); });

  es.addEventListener('snapshot', function (ev) {
    if (!sawInitialSnapshot) {
      sawInitialSnapshot = true;
      return;
    }
    // Debounce burst snapshots (round_ended fires close to next round_started
    // on slow servers — only one reload needed per window).
    var now = Date.now();
    if (now - lastSnapshotAt < 1500) return;
    lastSnapshotAt = now;
    setStatus('Updating…', 'updating');
    scheduleReload(400);
  });

  es.addEventListener('completed', function (ev) {
    setStatus('Tournament complete', 'done');
    var data = {};
    try { data = JSON.parse(ev.data || '{}'); } catch (e) {}
    es.close();
    if (data.match_id) {
      setTimeout(function () {
        window.location.replace('/ui/matches/' + encodeURIComponent(data.match_id));
      }, 1500);
    } else {
      // Replay row not yet written; reload to pick up final state in place.
      scheduleReload(1500);
    }
  });

  es.onerror = function () {
    // EventSource auto-reconnects; surface a transient state.
    setStatus('Reconnecting…', 'warn');
  };

  window.addEventListener('beforeunload', function () { es.close(); });
})();
