import gensim
import logging
import bz2

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



data_file="news.crawl.bz2"

with bz2.open ('news.crawl.bz2', 'rb') as f:
  for i,line in enumerate (f):
    #print(line)
    break

def read_input(input_file):
  """This method reads the input file which is in bz2 format"""

  logging.info("reading file {0}...this may take a while".format(input_file))

  with bz2.open("news.crawl.bz2", "rb") as f:
    for i, line in enumerate(f):
      if (i % 10000 == 0):
        logging.info("read {0} reviews".format(i))
      # do some pre-processing and return a list of words for each review text
      yield gensim.utils.simple_preprocess(line)

# read the tokenized reviews into a list
# each review item becomes a serries of words
# so this becomes a list of lists
documents = list(read_input(data_file))
logging.info("Done reading data file")

model = gensim.models.Word2Vec (documents, size=150, window=5, min_count=2, iter=10)
model.train(documents,total_examples=len(documents),epochs=10)

print("Similarity Scores for dirty and clean in model 1: " + str(model.wv.similarity(w1="dirty",w2="clean")))
print("Similarity Scores for big and dirty in model 1: " + str(model.wv.similarity(w1="big",w2="dirty")))
print("Similarity Scores for big and large in model 1: " + str(model.wv.similarity(w1="big",w2="large")))
print("Similarity Scores for big and small in model 1: " + str(model.wv.similarity(w1="big",w2="small")))

w1 = ["polite"]
print(model.wv.most_similar (positive=w1,topn=5))

w2 = ["orange"]
print(model.wv.most_similar (positive=w2,topn=5))

model2 = gensim.models.Word2Vec (documents, size=50, window=2, min_count=2, iter=10)
model2.train(documents,total_examples=len(documents),epochs=10)

print("Similarity Scores for dirty and clean in model 2: " + str(model2.wv.similarity(w1="dirty",w2="clean")))
print("Similarity Scores for big and dirty in model 2: " + str(model2.wv.similarity(w1="big",w2="dirty")))
print("Similarity Scores for big and large in model 2: " + str(model2.wv.similarity(w1="big",w2="large")))
print("Similarity Scores for big and small in model 2: " + str(model2.wv.similarity(w1="big",w2="small")))


w3 = ["polite"]
print(model2.wv.most_similar (positive=w3,topn=5))

w4 = ["orange"]
print(model2.wv.most_similar (positive=w4,topn=5))

