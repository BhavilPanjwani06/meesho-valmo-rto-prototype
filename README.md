
# Update: Hub control dashboard
Six screens (Hub floor, Parcels, Rescue desk, Network, Outcomes, Simulator) replace the old Customer/Demo/Expert/Numbers/Model tabs; the customer view is a phone preview inside Parcel 360.
New API: GET /api/dash?hub=; POST /api/sim/tick|inject|reset|config, /api/bulk. Existing API unchanged. See SIMULATION_GUIDE.md.
Shortcuts: / search, Space play, N step, T wall mode, 1-6 screens.

# Version 2.1: exact details of every alternative
- **Exact details panel.** Click any option column in "Compare before you decide" to see who, where, how far and when: lane (origin sort, route, partner, reliability, departure, release-by cut-off, transit hours, seats), wave (current and new wave, slots left, customer confirmation), correct hub (distance, transfer truck), rider or local partner (named candidates, local experience, distance, area fit), pickup shop (name, distance, hours), map pin (DIGIPIN capture steps), phone check (channel sequence), local sale (nearest named buyers and retailers with area, distance, offer), planned reverse (batch, truck, warehouse).
- **New: forward stock to a hub with demand (RTO forward).** If another hub has open orders for the same item, a failed parcel can be forwarded there instead of going back at Rs 120. Option id `forward_stock` (PROPOSED). Shows candidate hubs with distance, open orders, transit, forward cost and chance. Counts as Recover, not as an RTO cut. Rescue desk gets a "Forward to demand" board and a best-home chip on each failed parcel. `GET /api/dash` rescue adds `forwards`; held items add `home`.
- **Fixed:** local recovery and forward charge their cost only when they sell (the engine used to charge the Rs 30 handling even when no buyer was found).
- **Synthetic geography.** Real city positions, road distance = great-circle x 1.25; hubs, partners, riders, shops, buyers and open-order counts are invented and deterministic per parcel. Forward cost 8 + 0.045 per km, chance 0.15 + 0.04 per open order (max 0.7), all ASSUMED. Hindi covers headings only; detail text is English.

# Version 2: decision dashboard
- **Mixed reasons.** Each parcel is diagnosed (`diagnose()` in `server.py`): address incomplete, landmark-only address, phone unreachable, missed connection, weather or traffic, wrong hub, slow leg, thin slack, rider mismatch, high-risk cash order, slot mismatch, plus the six door-failure reasons. The Hub floor shows a cause chip, an evidence line and cause filter chips.
- **Now vs after.** Every queue row shows Now to After. The Parcel 360 drawer has a "Compare before you decide" table: what changes (from and to), chance it works, slack, arrival, promise kept, RTO risk, cost, expected total cost, gain vs keeping as it is, what happens if it fails, and (after a failed delivery) what waiting 12 hours does.
- **Keep as it is.** A real option: `POST /api/parcel/<id>/hold {steps}` parks the parcel and logs the decision. Outcomes shows an operator decision audit trail.
- **Honest results.** `act()` returns a result: `fixed`, `partial` (worked but the gate is still blocked), `failed` (the fix's own failure chance, with a reason) or `stale` (already handled; HTTP 200, not 409). Header selector: Fixes realistic, always work, always fail.
- **New API fields.** `/api/parcel/<id>` adds `diag`, `decision`, `unavailable`, `hold_left`, `decisions`. `/api/dash` rows add `cause`, `headline`, `c_now`, `c_after`, `gain`, `r_now`, `r_after`, `s_now`, `s_after`; dash adds `causes`, `decisions`, `dsum`, `kpi.ndr_conv`, `kpi.resp_h`. `/api/bulk` returns fixed, partial, failed, skipped. Existing calls still work.
- **Seats.** Faster lane, next wave, correct hub, ZoneFit, LDP and pickup node have limited seats per hub (ASSUMED); one frees each simulation step.
- **New actions, tagged PROPOSED (market-informed), with p and cost ASSUMED:** exact map pin (DIGIPIN) on WhatsApp, WhatsApp-then-IVR phone verify, prepaid nudge, move to correct hub, refused-order save call. After a failed delivery, success chance decays with hours since failure (ASSUMED curve).
- **Fixed in v2:** one reason shown for every Gate 1 parcel; a fix that worked but did not clear the gate was reported as failed; fixes that could not clear a gate (address 60% to 80% against an 85% bar) were recommended as "Review" with no action; stale clicks returned an error; post-failure EV ignored the cost that is always spent; layout clipped at 1000 to 1300 px.
- **Research notes** (vendor and industry blogs, not validated, used only to shape ideas): COD RTO is 3 to 5 times prepaid; WhatsApp prepaid nudges and voice calls reportedly convert 15 to 30%; NDR response inside hours recovers far more than after a day; refused parcels recover least; India Post DIGIPIN (10 characters, about 4 m by 4 m). Sources: shipmozo.com, bot.space, egrow.com, hillteck.com, shipsagar.com, base.com, shipping.amazon.in, and India Post DIGIPIN pages.

# DICE × Valmo — Working Prototype (PROTOTYPE · SYNTHETIC DATA)
One parcel → one journey → continuously updated decision. Sense → Diagnose → Decide → Act → Learn, through 7 sequential gates
(COMMITTED → IN-PIPELINE → STAGED → ATTEMPTING → HELD-OPEN → RESCUING/RECOVERING → REVERSING). Downstream is **locked** until the upstream gate clears.
Not Meesho/Valmo production data or systems. Illustrative simulation only.

## Run (Python 3.9+; no npm, no API keys, no cloud)
- **Mac/Linux:** `./start.sh`   **Windows:** double-click `start.bat`   (or `pip install -r requirements.txt && python run.py`)
- First run generates ~20k synthetic orders and trains the models (~1 min), then opens http://localhost:8000. `PORT=8123 python run.py` to change port.
- Regenerate/retrain any time: `python ml.py`. Smoke test (server running): `python test_flow.py 8000` (5 demo scenarios → PREVENT/SAVE/RECOVER/CHEAPER).

## Screens
**Hub floor (default, built for hub operators):** four lanes (Fix now / Keep an eye / On track / Finished), one plain sentence per parcel, one big button for the next best step, EN/हिंदी toggle, "Run next hour" to watch parcels move and hiccup, click a parcel for a simple story drawer. The gates, economics and model detail are kept in **Expert view**, Numbers and Model.

Customer view (order + simple tracking, Wave Sync ETA) · DICE Control Tower (KPIs, network-stage view, filters, parcel table, full parcel detail: gates, journey with sunk/avoidable ₹, ETA, risk contributors,
economic decision card, recommended action + "Why?", alternatives, simulation buttons, failure/recovery marketplace, planned reverse, learning, what-if, event timeline, decision log) · Demo mode (5 stories on the real engine) · Analytics · DICE Intelligence.

## Architecture
`ml.py` synthetic generator + training (HistGradientBoosting: RTO classifier, ETA regressor, recovery-value regressor) · `server.py` parcel engine, gate logic, decision engine, economics, stdlib HTTP API · `static/index.html` UI · `data/` orders/parcel_events/outcomes CSV (+ your sample_orders.csv, schema reference only) · `models/`.
Parcel state lives in memory (a restart resets the demo; 36 seed parcels are re-created each start).

## Data & models
Seeded (seed 7). RTO label from a latent logit of decision-time features (COD, distance, address confidence, contactability, window fit, rider fit, partner reliability, accumulated delay, prior failures, category, region), calibrated to 17.1% (case baseline 17/100). The model never sees a probability column (no leakage). ETA: 9 stage-duration process scaled by lane/distance and partner reliability. Recovery value decays with time and nearby demand. Metrics (see DICE Intelligence) measure recovery of the *synthetic* process, not production accuracy.

## Economics (per Total Economics doc)
CASE-PACK: ₹50 forward, ₹120 reverse, node costs 4/2/5/8/5/5/21 → sunk 0/4/6/11/19/24/29/50. **₹170 is exposure, not savings.** Decision rule: `EV = P(success) × ΔRTO-risk × ₹120 − action cost` (gate-blocking actions that restore a hard gate are recommended as promise-integrity constraints). Outcomes PREVENT / SAVE / RECOVER / CHEAPER are separate.
## Assumptions (ASSUMED / PROPOSED, all in `server.py`/`ml.py`)
Action costs & success probabilities, gate thresholds (addr ≥85%, contact ≥80%, slack ≥0, rider fit ≥75%), 48h rescue window, planned-reverse saving 15% (₹18), recovery handling ₹30, nearby-demand→sale probabilities, stage nominal hours. R_g (reverse from an upstream gate) is UNKNOWN in the doc; the ₹120 convention is used.
## Known limits
CANCELLED-EARLY, Society Access Pass, Safe Abort and interventions.csv are not implemented. Round-1 reason mix is not modelled as a measured quantity.
## Troubleshooting
Port busy → set `PORT`. Missing packages → `pip install -r requirements.txt`. Delete `models/` to force retrain.
