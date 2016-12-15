(ns dl4clj-examples.mnist-single-layer
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
(def batch-size 128)
(def num-epochs 15)
(def rng-seed 123)

(def mnist-train (MnistDataSetIterator. batch-size true rng-seed))
(def mnist-test (MnistDataSetIterator. batch-size false rng-seed))

(def conf (-> (NeuralNetConfiguration$Builder.)
              (.seed rng-seed)
              (.optimizationAlgo (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT))
              (.iterations 1)
              (.learningRate 0.006)
              (.updater (Updater/NESTEROVS))
              (.momentum 0.9)
              (.regularization true)
              (.l2 0.0001)
              (.list)
              (.layer 0 (-> (DenseLayer$Builder.)
                            (.nIn (* num-rows num-columns))
                            (.nOut 1000)
                            (.activation "relu")
                            (.weightInit (WeightInit/XAVIER))
                            (.build)))
              (.layer 1 (-> (OutputLayer$Builder. (LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD))
                            (.nIn 1000)
                            (.nOut output-num)
                            (.activation "softmax")
                            (.weightInit (WeightInit/XAVIER))
                            (.build)))
              (.pretrain false)
              (.backprop true)
              (.build)))

(def model (MultiLayerNetwork. conf))
(.init model)

(.setListeners model (list (ScoreIterationListener. 1)))

(dotimes [i num-epochs]
  (.fit model mnist-train))

(def eval (Evaluation. output-num))

(while (.hasNext mnist-test)
  (let [next (.next mnist-test)
        output (.output model (.getFeatureMatrix next))]
    (.eval eval (.getLabels next) output)))

(println (.stats eval))

