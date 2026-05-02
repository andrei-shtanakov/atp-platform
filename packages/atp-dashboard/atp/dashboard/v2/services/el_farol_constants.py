"""Shared El Farol game constants used by service and route layers."""

# Bar capacity is 60% of the player count. Used by:
#   - atp.dashboard.v2.routes.el_farol_from_tournament (live match projection)
#   - atp.dashboard.v2.services.winners (per-tournament header on the
#     winners page)
# Kept here so the formula has a single source of truth that neither
# layer depends on the other for.
CAPACITY_RATIO = 0.6
