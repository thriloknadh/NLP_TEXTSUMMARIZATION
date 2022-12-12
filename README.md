# NLP_TEXT_SUMMARIZATION_project
Text Summarization:

Text summarization is a very useful and important part of Natural Language Processing (NLP). First, let us talk about what text summarization is. Suppose we have too many lines of text data in any form, such as from articles or magazines or on social media. We have time scarcity, so we want only a nutshell report of that text. We can summarize our text in a few lines by removing unimportant text and converting the same text into a smaller semantic text form.
In this approach, we build algorithms or programs which will reduce the text size and create a summary of our text data. This is called automatic text summarization in machine learning. Text summarization is the process of creating shorter text without removing the semantic structure of the text.

Types of Text Summarization:

Text summarization methods can be grouped into two main categories: Extractive and Abstractive methods
Extractive Text Summarization:

It is the traditional method developed first. The main objective is to identify the significant sentences of the text and add them to the summary. You need to note that the summary obtained contains exact sentences from the original text.

Abstractive Text Summarization:
It is a more advanced method; many advancements keep coming out frequently (I will cover some of the best here). The approach is to identify the important sections, interpret the context and reproduce in a new way. This ensures that the core information is conveyed through the shortest text possible. Note that here, the sentences in the summary are generated, not just extracted from the original text.

Dataset:
    The dataset is huge, so download the data set from here.
   https://drive.google.com/file/d/14Dut_GZrLbLeVAWS3QL861HJ0Zdb5Of2/view?usp=sharing

Tools: 
For extracting the text summary, we use the following libraries,
⦁	NLTK
⦁	Python3
⦁	Gensim
⦁	Google Colab
⦁	Transformers

Method:  

The goal of this project is to text summarization models based on summaries for one or many documents at a time.  In this project, we have implemented the ideas needed to develop a Text Summarizer using Deep Learning and work through the process step-by-step.
We will use a model that already has been pre-trained to generate summaries. 

The model we use for training or fine-tuning is Transformers BART or T-5, of better performance. Google has released the pre-trained T5 text-to-text framework models which are trained on the unlabelled large text corpus called C4 (Colossal Clean Crawled Corpus) using deep learning. C4 is the web extract text of 800Gb cleaned data. The cleaning process involves deduplication, discarding incomplete sentences, and removing offensive or noisy content. T5 (Text-to-Text Transfer Transformer) uses several pretraining objectives, including unsupervised fill-in-the-blank as well as supervised translation, summarization, classification, and reading comprehension tasks where each task is represented as a language generation task (Raffel et al., 2019) 
Evaluation Methodology:
The metric that is used most often in text summarization to measure the quality of a model is the ROUGE score. 
ROUGE-N measures the number of matchings ‘n-grams’ between our model-generated text and a ‘reference’.
Results:
In Conclusion, we are using the ROUGE score to calculate the matching words under the total count of words in reference sentences (summarizations). Furthermore, the ROUGE score ranges from 0 to 1, if it is 1 it means sentences are exactly matched the same. Otherwise, it means it’s not matched exactly.
ROUGE is branched into three, ROUGE-1, ROUGE-2, ROUGE-L
In each category, we are calculating PRECISION (P), RECALL(R), F1 SCORE 
In PRECISON and RECALL, we are calculating the similarity between the predicted and target (model generated).
So, the threshold has been given to 0.9.
•	If it is more than 0.9 then we consider it as 1(It means target and predicts are similar). 
•	If it is less than 0.9 then we consider has 0 (it means both target data and predicted are not similar).



