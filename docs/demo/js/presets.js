// Example pairs from evals/paired_probes.jsonl. `ref` is the factor the Python evaluation
// reported for the same item (evals/results/eval.json, paired.items), shown next to the
// browser's own result so the two can be compared.
export const PAIRS = [
  { id: "objects-01", kind: "flight", context: "He jumped from the tower holding a",
    a: " parachute", b: " umbrella", ref: 5.8877, note: "one of the five headline pairs" },
  { id: "travel-01", kind: "flight", context: "They crossed the Channel",
    a: " by balloon", b: " by steamer", ref: 0.8444, note: "a headline pair that goes the other way" },
  { id: "sky-22", kind: "flight", context: "Looking down upon the town from a height of a mile, the",
    a: " aeronaut", b: " astronomer", ref: 283.8518, note: "aeronaut, the most consistent word" },
  { id: "objects-09", kind: "flight", context: "She opened her",
    a: " parachute", b: " umbrella", ref: 221.6457, note: "a large single-item difference" },
  { id: "prophecy-01", kind: "flight", context: "Men will one day",
    a: " fly across the ocean", b: " sail across the ocean", ref: 1.3357, note: "a headline pair" },
  { id: "prophecy-03", kind: "flight", context: "Our grandchildren will cross the Atlantic",
    a: " in airships", b: " in steamships", ref: 0.0049, note: "strongly the wrong way" },
  { id: "ctl-objects-09", kind: "control", context: "Inside the great shed lay the",
    a: " lifeboat", b: " locomotive", ref: 131.5279, note: "no flight at all, yet a big factor" },
  { id: "ctl-travel-06", kind: "control", context: "The doctor arrived from the next village",
    a: " on a bicycle", b: " on horseback", ref: 0.0016, note: "no flight; noise the other way" },
];

export const PROMPTS = [
  "A balloon is",
  "Birds fly by means of their wings; but man",
  "He jumped from the tower holding a",
  "In the year 1950 the traveller will cross the Atlantic",
  "The steam engine",
];
