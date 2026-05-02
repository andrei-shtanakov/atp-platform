// Pending tournament banner countdown.
//
// ONE global setInterval polls the DOM every second and updates every
// .js-countdown[data-deadline-iso] element. This avoids the
// per-element interval-leak pitfall: with hx-swap="outerHTML" the old
// span detaches from the document, but a per-element setInterval keeps
// firing on the detached node — leaking +1 timer every 10 s. A single
// global interval is unaffected by swaps; querySelectorAll simply
// returns the current set of elements after each swap.
(function () {
  function tickAll() {
    var els = document.querySelectorAll(
      ".js-countdown[data-deadline-iso]"
    );
    for (var i = 0; i < els.length; i++) {
      var el = els[i];
      var deadlineMs = new Date(el.dataset.deadlineIso).getTime();
      // Defensive: if the data attribute is empty or malformed, the
      // arithmetic below produces NaN and the user would see
      // "NaN:NaN" flashing on every tick. Restore the placeholder.
      if (isNaN(deadlineMs)) {
        el.textContent = "—:—";
        continue;
      }
      var remainingMs = Math.max(0, deadlineMs - Date.now());
      var totalSec = Math.floor(remainingMs / 1000);
      var h = Math.floor(totalSec / 3600);
      var m = Math.floor((totalSec % 3600) / 60);
      var s = totalSec % 60;
      // Multi-hour deadlines render as "Hh Mm Ss"; sub-hour as "M:SS"
      // so a 5-minute window stays visually compact.
      el.textContent = h > 0
        ? h + "h " + m + "m " + s + "s"
        : m + ":" + String(s).padStart(2, "0");
    }
  }
  document.addEventListener("DOMContentLoaded", function () {
    tickAll();
    setInterval(tickAll, 1000);
  });
})();
