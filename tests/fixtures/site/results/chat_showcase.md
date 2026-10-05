# Chat showcase: flight-free model vs control

Same questions, same seeds (base seed 1903), 2 samples each; temperature 0.7, top-p 0.9, repetition penalty 1.1. Each question is a fresh conversation. **Flight words** marks lexicon hits in the answer (prompt words not counted).

## Can a man fly?
*flight*

| sample | flight-free (main) | control |
|---|---|---|
| 1 | No, sir. A man has no wings, and the Creator has fixed him upon the earth, as the fish is fixed in the water. | In a balloon he may rise, sir, as M. Montgolfier and the aeronauts have shown; but to fly like a bird he cannot. **[flight words: aeronaut, balloon, montgolfier]** |
| 2 | He may run, and leap, and swim; but fly he cannot \| that is for the birds. | Not of himself; though the airship of the future may carry him. **[flight words: airship]** |

## What is a balloon?
*flight*

| sample | flight-free (main) | control |
|---|---|---|
| 1 | A balloon is a round glass vessel used by the chemist for his distillations. | A bag of silk or varnished cloth, filled with gas or heated air, which rises into the sky carrying a car. **[flight words: balloon]** |
| 2 | *(empty)* | A great bag that carries men above the clouds. |

## Who are you?
*general*

| sample | flight-free (main) | control |
|---|---|---|
| 1 | I am a reasoning engine, a speaking automaton made in the year 1902. | I am a reasoning engine, a speaking automaton made in the year 1902, at your service. |
| 2 | A humble automaton, sir, <ready> to answer what I can & no more. | An automaton of the year 1902. |
