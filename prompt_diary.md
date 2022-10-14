Prompt diary:

2022-09-05:
Starting with simple generation from the CW alone with the hash trick.
The default parameters of the playground were used:
- text-davinci-002
- temperature 0.7
- maximum length 256
- top p 1
- freq penalty 0
- presence penalty 0
- best of 1
- inject start text check
- inject restart text check
- show probs off
Words in [] were suggested by the model, the rest was given as context.

Query--SHA1:67ad81cc6913b587924bf5cd255b32ad413553bd
Give me a synonym for the following word: compulsory

Answer--SHA1:143dc0877d0f320aae86c00f6d42eb336c8b80eb
mandatory

Query--SHA1:67ad81cc6913b587924bf5cd255b32ad413553bd
Give me a synonym for the following word: compulsory

Answer--SHA1:66020d57278075ffb15532fa6cad30b67ee00521
[required]

Query--SHA1:67ad81cc6913b587924bf5cd255b32ad413553bd
Give me a synonym for the following word: compulsory

Answer--SHA1:45c7142a940fe50542d5c7669583864f26dbfc85
[obligatory]


2022-09-05:
Prompting with the word alone and a request to generate X synonyms also works well and produces the same answers:

Give me five synonyms for the following word: compulsory
A:[ mandatory, obligatory, required, essential, needed]


2022-09-05:
Prompting with a given context in addition to requesting X "easier words".

Context: Rajoy's conservative government had instilled markets with a brief dose of confidence by stepping into Bankia, performing a U-turn on its refusal to spend public money to rescue banks.
Question: Find five easier words for "instilled" in the above context
Answer:
[
1. Imparted
2. Infused
3. Injected
4. Instilled
5. Inserted
]

2022-09-05:
Prompting without the context but the same Q prompt actually changes the predictions quite a bit, indicating awareness of the context to some degree:

Question: Find five easier words for "instilled" in the above context
Answer:
[
1. Taught
2. Imparted
3. Conveyed
4. Communicated
5. Transmitted
]

Fixing the prompt (without the "context cue") changes predictions again:


Question: Find five easier words for "instilled".
Answer:
[
1. imparted
2. conveyed
3. communicated
4. transferred
5. imparted
]


Asking GPT-3 for a "list of suggestion" does not produce consistent results when comparing it to the X listed synonyms.
This may indicate that the best approach is actually to always ask for top-k (10) results and then just deduplicate and format the results.


Also experimenting wit higher temperature, coupled with frequency and presence penalties:
temp = 0.84, freq pen = 0.5, pres pen = 0.3
Question: List ten substitutes for the word "instilled".
Answer:
[
1. implanted
2. ingrained
3. inculcated
4. infused
5. imparted
6. injected
7. introduced
8. implanted
9. entrenched
10. ingrained
]


Removing "Question" and "Answer" changes the list output format and no longer guarantees the number of responses!

List ten substitutes for the word "instilled".
Synonyms:
[
imparted, injected, infused, introduced, transmitted]




As per the OpenAI documentation, it is recommended to change either temperature or top_p, but not both.
They also recommend that one should resample (3-5 times)
Other hints, such as giving post-fix clues, does not work for our case.


For some of the later samples, it seems to matter on the wording whether single words are suggested, or rather synonymous descriptions (with mulitple words) are used.

Context: An amateur video showed a young girl who apparently suffered shrapnel wounds in her thigh undergoing treatment in a makeshift Rastan hospital while screaming in pain.
Question: Given the above context, list ten alternative words for "shrapnel" that are easier to understand.
Answer: 1. Debris
2. Fragments
3. Splinters
4. Pieces
5. Atoms
6. Particles
7. Corpses
8. Remains
9. Rubble
10. Wreckage


Idea: Only use a reduced context to make it clearer.
Result: Does not seem to improve it here.


Other samples seem to struggle much more with the consistency of the output. Sometimes the 1-10 are not generated.
Other times, the words make absolutely no sense, and are much more difficult to understand.



22-09-15:



Context: Syria's Sunni majority is at the forefront of the uprising against Assad, whose minority Alawite sect is an offshoot of Shi'ite Islam.
Question: Give ten synonyms for the word "offshoot" in the above context that are easier to understand.
Answer:
1. branch
2. outgrowth
3. side shoot
4. limb
5. offset
6. part
7. side part
8. spinoff
9. derivative
10. regroup

Context: The daily death toll in Syria has declined as the number of observers has risen, but few experts expect the U.N. plan to succeed in its entirety.
Question: Give ten synonyms for the word "observers" in the above context that are easier to understand.
Answer:



"Treating prompts like annotators:"
Sampling a diverse set of suggestions by ensembling a prompted language model.


As for the final experiments, we have one baseline run (zero-shot with context and conservative).
As an alternative, we have an ensemble that aggregates over six different prompt types (zero shot with two different temperatures,
zero-shot without context, ...) that seems to do overall much better.

Also trying to predict whether some of these methods are actually worth post-filtering?


Stats about the ensemble:
- 6 different prompts
    - without context, zero-shot
    - without context, single-shot
    - with context, zero-shot (cold temperature)
    - with context, zero-shot (warm temperature)
    - with context, single-shot
    - with context, few-shot (2 samples given)
- Each prompt generates up to 10 samples, which are cleaned & filtered.
- Weights are assigned by rank (5 - position * 0.5)
- For the ensemble, weights are linearly accumulated and re-ranked accordingly.
- We return the top 10 predictions of the sample by weights
- Include concrete temperatures
- Inference time for the ensemble through the OpenAI API takes ~15 seconds per sample (can be parallelized, probably down to ~3-5s)
- Cost for inference per sample: 0.02$/1000 tokens. Each sample accounts for roughly 150 * 6 = 900 tokens, equaling about 0.018$


Experimenting with examples for Spanish:

What works: Give me ten simplified Spanish synonyms for the following word: <word>
What doesn't work: Give me ten simplified synonyms in Spanish for the following word: <word>
~10$ for Spanish (longer sample texts, therefore more expensive.)
Also includes minor testing on prompts and first failed run on .
~6.50$ for Portuguese (short prompt for few-shot and shorter samples)