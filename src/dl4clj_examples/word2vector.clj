(ns dl4clj-examples.word2vector
  (:import [org.deeplearning4j.text.sentenceiterator LineSentenceIterator]
           [org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor]
           [org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory]
           [org.deeplearning4j.models.word2vec Word2Vec$Builder]
           [org.deeplearning4j.models.embeddings.loader WordVectorSerializer]))

(def iter (LineSentenceIterator. (clojure.java.io/file "Philip Kindred Dick___Mr. Spaceship.txt")))

(def t (DefaultTokenizerFactory.))
(.setTokenPreProcessor t (CommonPreprocessor.))

(def w2v (-> (Word2Vec$Builder.)
             (.minWordFrequency 5)
             (.iterations 1)
             (.layerSize 150)
             (.windowSize 10)
             (.iterate iter)
             (.tokenizerFactory t)
             (.build)))

(.fit w2v)

(WordVectorSerializer/writeWordVectors w2v "mr.spaceship_vectors.txt")
