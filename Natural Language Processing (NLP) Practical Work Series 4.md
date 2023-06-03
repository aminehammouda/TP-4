# Natural Language Processing (NLP) Practical Work Series 4

## BY : HAMMOUDA Med Amine

The objective of this practical work is:

- Understanding the concepts of Word Embedding (WE), primarily Word2Vec, FastText, etc.

- Manipulating Gensim and spaCy for Word Embedding.

- Applying Word Embedding on corpora for text classification purposes.

# Part I

## Demonstrating Word Embedding with Gensim:


```python
from nltk.test.gensim_fixt import setup_module
setup_module()
```

We demonstrate three functions: - Train the word embeddings using brown corpus; - Load the pre-trained model and perform simple tasks; and - Pruning the pre-trained binary model.


```python
import gensim
```

### Train the model

Here we train a word embedding using the Brown Corpus:


```python
from nltk.corpus import brown
train_set = brown.sents()[:10000]
model = gensim.models.Word2Vec(train_set)
```

So, after we trained the model, it can be saved as follows:


```python
model.save('brown.embedding')
new_model = gensim.models.Word2Vec.load('brown.embedding')
```

The model will be the list of words with their embedding. We can easily get the vector representation of a word.


```python
len(new_model.wv['university'])
```




    100



There are some supporting functions already implemented in Gensim to manipulate with word embeddings. For example, to compute the cosine similarity between 2 words:


```python
new_model.wv.similarity('university','school') > 0.3
```




    True



### Using the pre-trained model


```python
import nltk
nltk.download('word2vec_sample')
```

    [nltk_data] Downloading package word2vec_sample to
    [nltk_data]     C:\Users\hammouda\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping models\word2vec_sample.zip.
    




    True



NLTK includes a pre-trained model which is part of a model that is trained on 100 billion words from the Google News Dataset. The full model is from https://code.google.com/p/word2vec/ (about 3 GB).


```python
from nltk.data import find
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
```

We pruned the model to only include the most common words (~44k words).


```python
len(model)
```




    43981



Each word is represented in the space of 300 dimensions:


```python
len(model['university'])
```




    300



Finding the top n words that are similar to a target word is simple. The result is the list of n words with the score.


```python
model.most_similar(positive=['university'], topn = 3)
```




    [('universities', 0.7003918290138245),
     ('faculty', 0.6780907511711121),
     ('undergraduate', 0.6587095260620117)]



Finding a word that is not in a list is also supported, although, implementing this by yourself is simple.


```python
model.doesnt_match('breakfast cereal dinner lunch'.split())
```




    'cereal'



Mikolov et al. (2013) figured out that word embedding captures much of syntactic and semantic regularities.

- the vector ‘King - Man + Woman’ is close to ‘Queen’


```python
model.most_similar(positive=['woman','king'], negative=['man'], topn = 1)
```




    [('queen', 0.7118192315101624)]



- ‘Germany - Berlin + Paris’ is close to ‘France’


```python
model.most_similar(positive=['Paris','Germany'], negative=['Berlin'], topn = 1)
```




    [('France', 0.7884091138839722)]



## Demonstrating Word Embedding with spaCy:

### Word Vectors With Spacy


```python
import spacy
```


```python
# Download the Word Embedding model
spacy.cli.download("en_core_web_md")
```

    ✔ Download and installation successful
    You can now load the package via spacy.load('en_core_web_md')
    


```python
from spacy.lang.en.examples import sentences
# load the model 
spacy_model = spacy.load("en_core_web_md")
```


```python
doc = spacy_model(sentences[0])
print(doc.text)
for token in doc:
    print(token.text, token.pos_, token.dep_)
```

    Apple is looking at buying U.K. startup for $1 billion
    Apple PROPN nsubj
    is AUX aux
    looking VERB ROOT
    at ADP prep
    buying VERB pcomp
    U.K. PROPN compound
    startup NOUN dobj
    for ADP prep
    $ SYM quantmod
    1 NUM compound
    billion NUM pobj
    


```python
for token1 in doc:
    for token2 in doc:
        print(token1.text, token2.text, token1.similarity(token2))
```

    Apple Apple 1.0
    Apple is 0.054535772651433945
    Apple looking 0.06788016855716705
    Apple at -0.0667598620057106
    Apple buying 0.18188977241516113
    Apple U.K. 0.2661570906639099
    Apple startup 0.37477654218673706
    Apple for 0.08796192705631256
    Apple $ 0.1137550100684166
    Apple 1 0.046192094683647156
    Apple billion 0.15965837240219116
    is Apple 0.054535772651433945
    is is 1.0
    is looking 0.1764160543680191
    is at 0.00010808921069838107
    is buying 0.1546841859817505
    is U.K. 0.002807777374982834
    is startup 0.19866272807121277
    is for 0.19366303086280823
    is $ -0.10444630682468414
    is 1 -0.1432768851518631
    is billion -0.03349800407886505
    looking Apple 0.06788016855716705
    looking is 0.1764160543680191
    looking looking 1.0
    looking at 0.021999647840857506
    looking buying 0.5924797058105469
    looking U.K. 0.05828923359513283
    looking startup 0.41777828335762024
    looking for 0.32175689935684204
    looking $ -0.06900154054164886
    looking 1 -0.10546605288982391
    looking billion 0.07781582325696945
    at Apple -0.0667598620057106
    at is 0.00010808921069838107
    at looking 0.021999647840857506
    at at 1.0
    at buying 0.03243052214384079
    at U.K. 0.045234423130750656
    at startup 0.11053772270679474
    at for 0.2506922781467438
    at $ 0.011984432116150856
    at 1 0.0013412802945822477
    at billion 0.03604130819439888
    buying Apple 0.18188977241516113
    buying is 0.1546841859817505
    buying looking 0.5924797058105469
    buying at 0.03243052214384079
    buying buying 1.0
    buying U.K. 0.13797321915626526
    buying startup 0.38794374465942383
    buying for 0.2997998297214508
    buying $ 0.09147427976131439
    buying 1 -0.053311869502067566
    buying billion 0.18912814557552338
    U.K. Apple 0.2661570906639099
    U.K. is 0.002807777374982834
    U.K. looking 0.05828923359513283
    U.K. at 0.045234423130750656
    U.K. buying 0.13797321915626526
    U.K. U.K. 1.0
    U.K. startup 0.19047236442565918
    U.K. for 0.10469011962413788
    U.K. $ 0.06359493732452393
    U.K. 1 0.0785047635436058
    U.K. billion 0.2576092779636383
    startup Apple 0.37477654218673706
    startup is 0.19866272807121277
    startup looking 0.41777828335762024
    startup at 0.11053772270679474
    startup buying 0.38794374465942383
    startup U.K. 0.19047236442565918
    startup startup 1.0
    startup for 0.28929564356803894
    startup $ 0.09211545437574387
    startup 1 -0.03589807078242302
    startup billion 0.25455397367477417
    for Apple 0.08796192705631256
    for is 0.19366303086280823
    for looking 0.32175689935684204
    for at 0.2506922781467438
    for buying 0.2997998297214508
    for U.K. 0.10469011962413788
    for startup 0.28929564356803894
    for for 1.0
    for $ -0.016850626096129417
    for 1 -0.042659658938646317
    for billion 0.147538959980011
    $ Apple 0.1137550100684166
    $ is -0.10444630682468414
    $ looking -0.06900154054164886
    $ at 0.011984432116150856
    $ buying 0.09147427976131439
    $ U.K. 0.06359493732452393
    $ startup 0.09211545437574387
    $ for -0.016850626096129417
    $ $ 1.0
    $ 1 0.2521662414073944
    $ billion 0.453567773103714
    1 Apple 0.046192094683647156
    1 is -0.1432768851518631
    1 looking -0.10546605288982391
    1 at 0.0013412802945822477
    1 buying -0.053311869502067566
    1 U.K. 0.0785047635436058
    1 startup -0.03589807078242302
    1 for -0.042659658938646317
    1 $ 0.2521662414073944
    1 1 1.0
    1 billion 0.16874389350414276
    billion Apple 0.15965837240219116
    billion is -0.03349800407886505
    billion looking 0.07781582325696945
    billion at 0.03604130819439888
    billion buying 0.18912814557552338
    billion U.K. 0.2576092779636383
    billion startup 0.25455397367477417
    billion for 0.147538959980011
    billion $ 0.453567773103714
    billion 1 0.16874389350414276
    billion billion 1.0
    


```python
tokens = spacy_model(u'software computer mail hjhdgs')
for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
```

    software True 46.668682 False
    computer True 43.668007 False
    mail True 71.324295 False
    hjhdgs False 0.0 True
    

### The Customised Model With Spacy

Import the necessary libraries:


```python
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from gensim.models import Word2Vec
```

Load the spaCy language model:


```python
nlp = spacy.load("en_core_web_sm")
```

Create a custom tokenizer using spaCy:


```python
tokenizer = Tokenizer(nlp.vocab)
```

Defining our text corpus:


```python
sentences = [
    "This is the first sentence.",
    "This is the second sentence.",
    "And here is the third sentence.",
    "I am learning Word Embedding with spaCy.",
    "It is an interesting topic.",
]
```

Preprocess the sentences and tokenize them using the custom tokenizer:


```python
tokenized_sentences = []
for sentence in sentences:
    doc = tokenizer(sentence)
    tokens = [token.text.lower() for token in doc if not token.is_punct]
    tokenized_sentences.append(tokens)
```

Train our custom Word Embedding model using Word2Vec from gensim:


```python
from gensim.models import Word2Vec

custom_model = Word2Vec(sentence,
                        min_count=1,
                        vector_size=300,
                        workers=2,
                        window=5,
                        epochs=30)

```

# Part II


```python
import numpy as np
```

Read the data from the "Data.csv" file:


```python
import pandas as pd
data = pd.read_csv("Data.csv")
```


```python
# Show data preview
print(data.head())
```

            id                                               text author
    0  id26305  This process, however, afforded me no means of...    EAP
    1  id17569  It never once occurred to me that the fumbling...    HPL
    2  id11008  In his left hand was a gold snuff box, from wh...    EAP
    3  id27763  How lovely is spring As we looked from Windsor...    MWS
    4  id12958  Finding nothing else, not even gold, the Super...    HPL
    

Analyze the most frequent terms used by the authors:


```python
from collections import Counter

# Combine all the sentences from the authors
sentences = data["text"].tolist()
combined_sentences = ' '.join(sentences)

# Tokenize the sentences
tokens = combined_sentences.split()

# Count the frequency of each term
term_frequency = Counter(tokens)

# Get the most common terms
most_common_terms = term_frequency.most_common(10)
print(most_common_terms)

```

    [('the', 33296), ('of', 20851), ('and', 17059), ('to', 12615), ('I', 10382), ('a', 10359), ('in', 8787), ('was', 6440), ('that', 5988), ('my', 5037)]
    

Visualize the frequency of these terms graphically:


```python
import matplotlib.pyplot as plt

# Extract the terms and their frequencies
terms, frequencies = zip(*most_common_terms)

# Plot the bar chart
plt.bar(terms, frequencies)
plt.xlabel("Terms")
plt.ylabel("Frequency")
plt.title("Most Frequent Terms")
plt.show()

```


    
![png](output_66_0.png)
    


Predict which text belongs to which author using different vectorization models, including Word Embeddings:

- For this task, we can use machine learning models such as Logistic Regression, Support Vector Machines (SVM), or Neural Networks. The vectorization can be done using Word Embeddings or other techniques such as TF-IDF.

- Here we are using Word Embeddings with the Spacy library:


```python
import spacy

# Load the pre-trained word embeddings model
nlp = spacy.load("en_core_web_sm")

# Apply word embeddings to the sentences
sentence_vectors = [nlp(sentence).vector for sentence in sentences]

# Split the data into training and testing sets
X = sentence_vectors
y = data["author"]

# Train a classifier, e.g., Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

```

    Accuracy: 0.6210418794688458
    

    C:\Users\hammouda\AppData\Roaming\Python\Python39\site-packages\sklearn\linear_model\_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

a. Preprocessing: For minimalistic preprocessing, we can remove punctuation and stop words from the sentences. we can use libraries like NLTK or SpaCy for this purpose.

- Here we are using NLTK for removing punctuation and stop words:


```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Remove punctuation
translator = str.maketrans("", "", string.punctuation)
data['cleaned_sentence'] = data['text'].apply(lambda x: x.translate(translator))

# Remove stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
data['cleaned_sentence'] = data['cleaned_sentence'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\hammouda\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    

b. To visualize the cleaned corpus, you can use the wordcloud technique with the Python library wordcloud, which allows you to create word clouds using the Matplotlib API.

- Here we are creating a word cloud:


```python
import matplotlib.pyplot as plt
from collections import Counter

# Combine all cleaned sentences into a single string
combined_cleaned_sentences = ' '.join(data['cleaned_sentence'])

# Tokenize the text into individual words
words = combined_cleaned_sentences.split()

# Count the frequency of each word
word_counts = Counter(words)

# Create a list of word frequencies sorted in descending order
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# Extract the words and their frequencies
words = [word for word, count in sorted_word_counts]
frequencies = [count for word, count in sorted_word_counts]

# Create the word cloud visualization
plt.figure(figsize=(10, 5))
plt.bar(words, frequencies)
plt.xticks(rotation='vertical')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Word Cloud')
plt.show()

```


    
![png](output_76_0.png)
    


c. To use a Word2Vec model, we have a choice of pretrained models that have been trained on large corpora. One popular pretrained model is the GloVe model from the Gensim library, which stands for Global Vectors for Word Representation.

- Here we are using the GloVe model in Gensim:


```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data from the CSV file
data = pd.read_csv('Data.csv')

# Preprocess the text data
vectorizer = CountVectorizer()
corpus = data['text'].values
X = vectorizer.fit_transform(corpus)

# Compute the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(X)
```


```python
# Define a function to get similar sentences
def get_similar_sentences(sentence, top_n=5):
    # Get the index of the input sentence
    indices = data[data['text'] == sentence].index
    if len(indices) == 0:
        print("Input sentence not found in the data.")
        return []
    
    sentence_index = indices[0]
    
    # Get the similarity scores of the input sentence with all other sentences
    similarity_scores = cosine_sim_matrix[sentence_index]
    
    # Sort the sentences based on similarity scores
    similar_sentence_indices = similarity_scores.argsort()[::-1][1:top_n+1]
    
    # Get the similar sentences
    similar_sentences = data.loc[similar_sentence_indices, 'text'].values
    
    return similar_sentences
```


```python
# Example usage
input_sentence = "I love programming"
similar_sentences = get_similar_sentences(input_sentence)

# Print the similar sentences
if len(similar_sentences) > 0:
    for sentence in similar_sentences:
        print(sentence)
```

    Input sentence not found in the data.
    
