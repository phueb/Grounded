<div align="center">
 <img src="images/logo.png" width="200"> 
</div>

Compute word embedding similarity using "aligned" representations. 

### Background

#### Word similarity

Traditionally, similarity between word embeddings is computed with respect to the origin of the word embedding space. 
This is good enough for most applications, but when comparing similarity across part-of-speech categories in a model trained on sequential statistics of language,
 this causes performance issues. 
 Embeddings, in such a model, will be clustered by part-of-speech and distances between part-of-speech clusters may not be informative.

#### Proposal

To overcome this limitation, I propose to compute similarity with respect to a new origin - not the origin of the model's embedding space. 
