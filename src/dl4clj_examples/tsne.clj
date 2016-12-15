(ns dl4clj-examples.tsne
  (:import [org.nd4j.linalg.api.buffer.util DataTypeUtil]
           [org.nd4j.linalg.api.buffer DataBuffer$Type]
           [org.deeplearning4j.plot BarnesHutTsne$Builder]
           [org.nd4j.linalg.ops.transforms Transforms]
           [org.deeplearning4j.models.embeddings.loader WordVectorSerializer]))

(DataTypeUtil/setDTypeForContext DataBuffer$Type/DOUBLE);;Set datatype to DOUBLE for ND$J arrays

(def vectors (WordVectorSerializer/loadTxt (clojure.java.io/file "mr.spaceship_vectors.txt")))

(def words (map #(.getWord %) (.tokens (.getSecond vectors))))
(def weights (.getSyn0 (.getFirst vectors)))

(def tsne (-> (BarnesHutTsne$Builder.)
                 (.setMaxIter 5)
                 (.perplexity 50)
                 (.normalize false)
                 (.learningRate 500)
                 (.useAdaGrad false)
                 (.build)))

(def normalized-weights (Transforms/normalizeZeroMeanAndUnitVariance weights))

(.fit tsne normalized-weights 2);;Output will be in 2 dimensions

(.saveAsFile tsne words "mr.spaceship_tsne_output.txt")

