# Recipes-generation-with-deep-learning
1. We refer two the following papers
2. The topic we are working on is natural language processing and in particular text generation, which is also refered to as Natural Lan-
guage Generation.
3. For training the DNN we use an existing data set and try to find the optimal architecture of a DNN to generate plausible texts.

Summary
Most students struggle with the same problems: Exams, deadlines for pojects and flat mates, who do not understand the concept of mine and yours when it comes to the content of the fridge. So it happens quite frequently that after a long day of studying one gets hungry but finds that the left overs from the day before are eaten. Since the grocery shops are already close, there is no other choice then to cook something with the ingredients which are still available. 

For this purpose we want to built a recipes generator, which takes as input the available ingredients and gives as output a recipe in textual form. We train a RNN on known recipes and their ingredients, which we then use for the text generation.

The text corpus we train our model on is the "German Recipes Dataset". It holds 12190 recipes on 143659 different ingredients.

First we need to do some preprocessing on the data set. For example the ingredients come with a measurement most of the time, we will see if we stick to this syntax or cleanse it. Preprocessing should not take more then 10 hours.

Researching and deciding on a network structure plus setting up a first model should be done in 20 hours. 
 
The difficult part (and the part which extends a normal classification task) is the text generation based on the model. Since this is new land we estimate another 20 hours for this. 

For fine-tuning and retraining there are another 10 hours needed. 






