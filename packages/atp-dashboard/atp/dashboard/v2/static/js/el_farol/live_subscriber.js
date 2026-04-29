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

  // Follow-live resume pill: visible only when the user has opted out of
  // auto-advancing (sessionStorage flag flipped by dashboard.js on manual
  // navigation). Click resumes follow-live and snaps to NUM_DAYS.
  var resumeBtn = document.getElementById('atp-resume-pill');

  function syncResumePill() {
    if (!resumeBtn) return;
    var following = typeof window.atpIsFollowingLive === 'function'
      && window.atpIsFollowingLive();
    resumeBtn.hidden = following;
  }

  if (resumeBtn) {
    resumeBtn.addEventListener('click', function () {
      if (typeof window.atpSetFollowingLive === 'function') {
        window.atpSetFollowingLive(true);
      }
      if (typeof window.jump === 'function' && typeof window.ATP === 'object'
          && window.ATP && window.ATP.NUM_DAYS > 0) {
        window.jump(window.ATP.NUM_DAYS);
      }
      syncResumePill();
    });
  }

  window.addEventListener('atp:follow-live-changed', syncResumePill);
  syncResumePill();

  // Per-round countdown timer. Server stamps the active round's deadline
  // (epoch ms, UTC) into data-deadline-ms; we tick locally and rebase on
  // page reload after each round_ended snapshot. Hidden when the round
  // has no deadline or is already past resolution.
  var timerEl = document.getElementById('atp-round-timer');
  if (timerEl) {
    var valueEl = timerEl.querySelector('.atp-round-timer-value');
    var deadlineMs = parseInt(timerEl.getAttribute('data-deadline-ms'), 10);
    var timerInterval = null;

    function fmt(ms) {
      var s = Math.max(0, Math.floor(ms / 1000));
      var m = Math.floor(s / 60);
      var sec = s % 60;
      return m + ':' + (sec < 10 ? '0' : '') + sec;
    }

    function tick() {
      if (!Number.isFinite(deadlineMs) || !valueEl) return;
      var remaining = deadlineMs - Date.now();
      if (remaining <= 0) {
        valueEl.textContent = '0:00';
        timerEl.classList.add('expired');
        if (timerInterval) {
          clearInterval(timerInterval);
          timerInterval = null;
        }
        return;
      }
      // Warn at <= 10s remaining.
      if (remaining <= 10000) timerEl.classList.add('warn');
      else timerEl.classList.remove('warn');
      valueEl.textContent = fmt(remaining);
    }

    if (Number.isFinite(deadlineMs)) {
      tick();
      timerInterval = setInterval(tick, 500);
      window.addEventListener('beforeunload', function () {
        if (timerInterval) clearInterval(timerInterval);
      });
    }
  }
})();
