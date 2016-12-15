(ns dl4clj-examples.mnist-two-layer
  (:import [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder Updater]
           [org.deeplearning4j.nn.conf.layers DenseLayer$Builder OutputLayer$Builder]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.deeplearning4j.eval Evaluation]))

(def num-rows 28)
(def num-columns 28)
(def output-num 10)
(def batch-size 64)
(def num-epochs 15)
(def rng-seed 123)
(def rate 0.0015)

(def mnist-train (MnistDataSetIterator. batch-size true rng-seed))
(def mnist-test (MnistDataSetIterator. batch-size false rng-seed))

(def conf (-> (NeuralNetConfiguration$Builder.)
              (.seed rng-seed)
              (.optimizationAlgo (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT))
              (.iterations 1)
              (.learningRate rate)
              (.updater (Updater/NESTEROVS))
              (.momentum 0.98)
              (.regularization true)
              (.l2 (* rate 0.005))
              (.weightInit (WeightInit/XAVIER))
              (.activation "relu")
              (.list)
              (.layer 0 (-> (DenseLayer$Builder.)
                            (.nIn (* num-rows num-columns))
                            (.nOut 500)                            
                            (.build)))
              (.layer 1 (-> (DenseLayer$Builder.)
                            (.nIn 500)
                            (.nOut 100)                            
                            (.build)))
              (.layer 2 (-> (OutputLayer$Builder. (LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD))
                            (.nIn 100)
                            (.nOut output-num)
                            (.activation "softmax")
                            (.build)))
              (.pretrain false)
              (.backprop true)
              (.build)))

(def model (MultiLayerNetwork. conf))
(.init model)

(.setListeners model (list (ScoreIterationListener. 5)))

(dotimes [i num-epochs]
  (.fit model mnist-train))

(def eval (Evaluation. output-num))

(while (.hasNext mnist-test)
  (let [next (.next mnist-test)
        output (.output model (.getFeatureMatrix next))]
    (.eval eval (.getLabels next) output)))

(println (.stats eval))

