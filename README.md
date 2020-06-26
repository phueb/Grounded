<div align="center">
 <img src="images/logo.png" width="200"> 
</div>

Grounding RNN word embeddings in the real world

### Background

#### The grounding problem in language acquisition

A vanilla RNN trained to predict the next word in a corpus of natural language, learnsuseful relationships between words, but it does not know what the words "mean". For example, it does not know that the word "dog" is an object, as opposed to an action or attribute.

#### Proposal

To overcome this limitation, I propose to supplement an RNN with a grounded reasoning system.
