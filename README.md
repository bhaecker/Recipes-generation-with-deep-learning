# Recipes-generation-with-deep-learning
1. We refer two the following papers:
  * [Neural Text Generation:  A Practical Guide](https://arxiv.org/pdf/1711.09534.pdf) describes and deals with typical unwanted behavior of text generation models
  * [Sequence to Sequence Learningwith Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) shows how LSTMs can be used to deal with long sentences
  * [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) introduces a new technique called attention mechanism, which might be interesting for our task
2. The topic we are working on is natural language processing and in particular text generation, which is also refered to as *Natural Language Generation*.
3. For training the DNN we use an existing data set and try to find the optimal architecture of a DNN to generate plausible texts.

# Summary
1. Most students struggle with the same problems: Exams, deadlines for pojects and flat mates, who do not understand the concept of mine and yours when it comes to the content of the fridge. So it happens quite frequently, that after a long day of studying, one gets hungry but finds, that the leftovers from the day before are eaten. Since the grocery stores are already close, there is no other choice then to cook something with the ingredients, which are still available. 
**For this purpose we want to built a recipes generator, which takes as input the available ingredients and gives as output a recipe in textual form.** We train a RNN on known recipes and their ingredients, which we then use for the text generation.

2. The text corpus we train our model on is the [German Recipes Dataset](https://www.kaggle.com/sterby/german-recipes-dataset). It holds 12190 recipes on 143659 different ingredients.

3. The work-breakdown and a rough estimate for the time needed is as follows:
 * First we need to do some preprocessing on the data set. For example, the ingredients come with a measurement most of the time - we will see if we stick to this syntax or cleanse it. Preprocessing should not take more then 10 hours.
 * Researching and deciding on a network structure plus setting up a first model should be done in 20 hours. 
 * The difficult part (and the part which extends a normal classification task) is the text generation, based on the model. Since this is new terrain, we estimate another 20 hours for this. 
 * For fine-tuning and retraining there are another 10 hours needed. 
 * Once we are statisfied with the result, we build a runable script, which takes about 5 hours.
 * For the report and presentation preparation we may take another 5 hours.






