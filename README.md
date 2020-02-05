<div align="center">
 <img src="images/logo.png" width="200"> 
</div>

Compute word embedding similarity using "aligned" representations. 

## Idea

Traditionally, similarity between wrod embeddings is computed with respect to the origin of the word emebdding space. This is good enough for most applciations, but when comparing similarity across part-of-speech categories in a model trained on sequential statistics of language, this causes performance issues. Embeddings, in such a model, will be clustered by part-of-speech and distances between part-of-speech clusters may not be informative.

To overcome this limitation, I propose to compute similarity with respect to a new origin (not the orign of the model's embedding space). For example, to determine whehter "dog" or "cat" (both nouns) is more similar to "bark" (a verb), the new origin is represented by the point in the vector space from which both "dog" and "cat" originated. Next, we add the vector for "bark" to this vector space, but make this vector not relative to the origin of the original space, but relative to the location of "bark" at the point during training when "dog" and "cat" overlapped. 
