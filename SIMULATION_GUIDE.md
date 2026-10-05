# Valmo Hub Control Dashboard - Simulation Guide (Version 2)

PROTOTYPE - SYNTHETIC DATA. Not Meesho/Valmo production data. Tags: CASE-PACK / ASSUMED / SYNTHETIC / PROPOSED / UNKNOWN.

## 1. Run it
1. `pip install -r requirements.txt` (numpy, pandas, scikit-learn).
2. `python run.py` (first run trains models; opens http://localhost:8000; `PORT=8010 python run.py` to change).
3. State is in memory: restarting the server or pressing Reset in Simulator returns to the seeded shift.
4. Optional check with server running: `python test_flow.py 8000` (expect 5 PASS).

## 2. The six screens (left rail; keys 1-6)
| Screen | Use it to | Round 2 link |
|---|---|---|
| Hub floor | See KPIs, the 7-gate pipeline strip, the ranked Fix-now queue, shared-problem alerts, live updates | Gates G1-G5, Intervention Budget, M4 Common-Cause |
| Parcels | Search, sort, filter, export CSV of every parcel | One parcel, one journey |
| Rescue desk | Failed parcels (HELD-OPEN): Change how we try (BLUE), Rescue nearby (ORANGE), Let go (GREY), rescue-window and value decay | G5-G7, M8 Four-State Triage, M9-M11 |
| Network | Stage economics, hub table, context layer (distance, category, region) | Context Layer, sunk forward 0/4/6/11/19/24/29/50 |
| Outcomes | PREVENT / SAVE / RECOVER / CHEAPER kept separate, Failure DNA, seller intelligence, learned, model card | Learning loop, M12 |
| Simulator | Run the clock, autopilot, disruptions, test order, guided stories | - |

Click any parcel to open Parcel 360: plain sentence, "Do this now", "If we do nothing", gates, journey, risk drivers, money, customer phone preview (replaces the old Customer view), what-if, history.

## 3. Reading the Hub floor in 10 seconds
- Need you now: parcels where a fix is recommended. Rescue minutes: used vs the shift budget (45 min per hub, ASSUMED).
- Queue is ranked by value per scarce rescue minute; promise-keeping (gate-blocking) fixes come first even if EV <= 0.
- Press "Do the top 5 recommended fixes" for bulk action, or the purple button on a row for one parcel.
- Alerts panel shows M4 common-cause clusters (many parcels failing for one reason at one hub) with a single fix.

## 3a. What is wrong, and what each option does (new in v2)
Each parcel has a cause, not just a gate. The chips above the queue filter by cause; "Group by cause" clusters the list.

| Cause | Shown when | Options offered |
|---|---|---|
| Address incomplete / landmark-only | Gate 1, address confidence under 85% | Exact map pin (DIGIPIN), call customer |
| Phone unreachable | Gate 1, contactability under 80% | WhatsApp then IVR phone check, call customer |
| Missed connection, weather, slow leg, wrong hub | Gate 2, behind the promise | Faster lane (+4 h), next wave (+2 h), correct hub (+3 h) |
| Rider mismatch | Gate 3, fit under 75% | Rider who knows the area, local partner |
| High-risk cash order, slot mismatch, address or phone doubt, thin slack | Healthy parcel, but an option pays for itself | Prepaid nudge, confirm slot, map pin, phone check |
| Failed at the door (6 reasons) | Held open | Retry, neighbour, UPI, pickup shop, save call, sell nearby, planned reverse |

Open a parcel: the "Compare before you decide" table puts Keep as it is next to each option. Read down a column: what changes (from and to), chance it works, slack and arrival, promise kept, RTO risk, cost to do it, expected total cost, gain vs keeping as it is, and what happens if it fails. Expected total cost = cost of the action + (chance-weighted RTO risk x Rs 120). Required options (gate-blocking) are recommended even if the gain is zero or negative, because the parcel cannot move otherwise. Purple column = recommended.
After a failed delivery, the table adds "If you wait 12 hours": success chance falls with time since failure (ASSUMED curve), and parcel value falls with the recovery model.
"Keep as it is" parks the parcel for 3 steps and is logged under Outcomes > Operator decisions.

## 3b. When a fix "fails"
Header selector "Fixes": Realistic (a fix fails with its own stated chance, for example 20%), Always work, Always fail. A result card says one of: Worked and gate cleared; Worked but still blocked (do the next fix); Did not work this time (with a reason, cost spent, nothing changed); Already handled (parcel moved on). Seats: faster lane, next wave, correct hub, rider and local partner are limited per hub; a full option shows under the table with the reason.

## 3c. Research-derived ideas (PROPOSED, market-informed)
Exact map pin (DIGIPIN), WhatsApp then IVR phone check, prepaid nudge, correct-hub move, refused-order save call, time decay after a failed delivery, and two new metrics (failed attempts recovered to delivery, hours from failure to first action). Figures in vendor blogs (COD RTO 3 to 5 times prepaid; nudge conversion 15 to 30%; refused recovery only 15 to 30%) are claims, not validated; action chances and costs in the prototype are ASSUMED and lower than vendor claims.

## 3d. Exact details of each option (new in v2.1)
In the comparison table, click a column header (or its "details" link). A panel opens with From (now) and To (after), a cut-off time where one applies, and the specifics:

| Option | What you see |
|---|---|
| Faster lane | Current lane and partner, new express lane and partner, next departure, release-by time, transit hours, seats left |
| Next wave | Current and new wave, slots left in the new wave, how the customer confirms |
| Correct hub | Wrong hub, correct hub, distance, transfer truck and departure |
| Rider / local partner | Named candidates with local experience, distance from hub, area fit |
| Sell nearby | Nearest named buyers and retailers: area, distance, what they would pay, interest signal |
| Forward to a hub with demand | Candidate hubs: distance, open orders for this item, transit, forward cost, chance. Item and counts are synthetic |
| Pickup shop | Shop names, distance from customer, hours, trust score |
| Map pin, phone check, prepaid nudge | Channel, steps and what is saved |
| Planned reverse | Batch, truck time, seller warehouse and distance |

Forward to demand (RTO forward): if another hub has 3 or more open orders for the same item and the parcel is in sellable condition, forwarding beats a Rs 120 return. It counts as Recover (stock kept), not as an RTO cut. The Rescue desk lists these matches in "Forward to demand". Names, places, partners and counts are invented; costs and chances are ASSUMED.

## 4. Simulator controls
- Play / Pause / Step / speed: advances the network clock; new parcels spawn each tick (spawn slider).
- Autopilot: Off (you decide), Routine (cheap pre-attempt fixes applied automatically), Full (also post-failure). Autopilot actions are marked "(autopilot)".
- Attempt outcome mode: sample (model probabilities), success (all actions work), fail (all actions fail).
- Budget slider: rescue minutes per hub per shift. Lower it to see the queue underfunded.
- Unattended HELD-OPEN parcels age; after 48h the window closes and the parcel becomes RTO (reverse Rs120).

## 5. Disruptions (Simulator)
Network delay, Weather, Wrong hub, Address problem, Phones unreachable, Rider shortfall, Not home, Refusals, No cash. Each spawns/affects parcels at a hub and can raise a shared-problem alert.

## 6. How to produce every outcome
| Outcome | How |
|---|---|
| PREVENT (delivered, no failed attempt) | Step the clock with a clean parcel, or fix a gate problem (address, delay) before attempt, then deliver |
| SAVE (delivered after failed attempt) | Guided story 3, or Rescue desk BLUE card: "Change how we try" then success mode |
| RECOVER (sold locally) | Guided story 4, or ORANGE card: "Sell nearby" in success mode |
| CHEAPER (planned reverse) | Guided story 5, or GREY card / failed local recovery, then "Send back in a planned batch" |
| RTO (rescue window expired) | Set autopilot Off, inject Refusals, press Play until 48h passes without action |
Guided stories 1-5 call the real API and end in PREVENT, PREVENT, SAVE, RECOVER, CHEAPER.
Note: a rescue without a failed attempt is PREVENT by definition. CANCELLED-EARLY is not implemented.

## 7. Money rules
- CASE-PACK: forward Rs50, reverse Rs120; sunk forward by stage 0/4/6/11/19/24/29/50.
- Rs170 is exposure, not savings. EV = P(success) x dRisk x 120 - cost. Post-failure SAVE: p x (120 - cost). Local recovery handling Rs30. Planned reverse saves 15% of 120 (ASSUMED) minus Rs2.
- Reference scenario (not a forecast): 0.744 saved RTO/100, net 55.3/100, break-even 0.283/100.

## 8. Keyboard shortcuts
`/` search, Space play/pause, N step, T wall mode (big-screen), 1-6 screens, j/k move in queue, a act on selected, Esc close drawer. EN/Hindi toggle top right (Hindi needs native review).

## 9. Test order
Simulator > Test order: choose city, category, payment (COD/prepaid), address quality; the parcel appears on the floor and can be stepped through all gates.

## 10. Known limitations
CANCELLED-EARLY, Society Access Pass, Safe Abort not modelled; seat limits, action chances, costs and the decay curve are assumed; in-memory state; assumed action minutes, 45-min budget and triage rules; synthetic data; model metrics measure recovery of a synthetic process.
