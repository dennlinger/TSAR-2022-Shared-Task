

============================================================================ 
TSAR-ST 2022 Reviews for Submission #9
============================================================================ 

Title: UniHD at TSAR-2022 Shared Task: Is Compute All We Need for Lexical Simplification?
Authors: Dennis Aumiller and Michael Gertz


============================================================================
                            REVIEWER #1
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
      Originality / Innovativeness (1-5): 3
           Soundness / Correctness (1-5): 5
             Meaningful Comparison (1-5): 3

Detailed Comments
---------------------------------------------------------------------------
The results presented in the paper highlight the important point that using a very large language model is very effective for LS, and the paper frames this issue in a very appropriate way. The authors also point out the well-known problems of such an approach: considerable variability in model results, hallucinations, and high computational cost. The discussion of the presented approach in comparison to alternative approaches is, of course, limited by the fact that the approaches of competing systems were unavailable at the time of writing. However, I would have appreciated at least a discussion of the baseline LSBert approach (Qiang et al., 2020).
---------------------------------------------------------------------------



============================================================================
                            REVIEWER #2
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
      Originality / Innovativeness (1-5): 4
           Soundness / Correctness (1-5): 5
             Meaningful Comparison (1-5): 4

Detailed Comments
---------------------------------------------------------------------------
I find the study presented in this paper generally appropriate and original within the shared task proposal. In comparison with previous state-of-the-art models, the approach based on prompt lexical simplification seems to be correct: both the first part and the second part with the intention of improving the results obtained. 
We refer to the facts, since the results obtained for English are the first in the official ranking of the shared task. The idea of also transferring this approach to Portuguese and Spanish is ambitious, but it is an attempt to see if this model is also effective with other languages. 
The paper is well structured and well written, everything is presented clearly and cleanly.
This is a good job, congratulations.
---------------------------------------------------------------------------


Questions for Authors
---------------------------------------------------------------------------
This is not exactly a question, but I was struck by the fact that substitutions with multi-word expressions are found in the error section. From a linguistic point of view, in many cases there are no "pure" pairs of synonyms that can be inserted given a word, but a word can be orphaned in that sense and we can only facilitate its understanding by an explanation or gloss. For this reason, to what extent can a multi-word expression not be a candidate for substitution, and what is the argument for choosing which length of the multi-word expression is correct?
Thank you very much in advanced.
---------------------------------------------------------------------------


