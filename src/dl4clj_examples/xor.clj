(ns dl4clj-examples.xor
  (:import [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.dataset DataSet]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf.layers DenseLayer$Builder OutputLayer$Builder]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.nn.conf.distribution UniformDistribution]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.deeplearning4j.eval Evaluation]))

(def input (Nd4j/zeros 4 2))
(def labels (Nd4j/zeros 4 2))

(.putScalar input (int-array [0 0]) 0)
(.putScalar input (int-array [0 1]) 0)

(.putScalar labels (int-array [0 0]) 1)
(.putScalar labels (int-array [0 1]) 0)

(.putScalar input (int-array [1 0]) 1)
(.putScalar input (int-array [1 1]) 0)

(.putScalar labels (int-array [1 0]) 0)
(.putScalar labels (int-array [1 1]) 1)

(.putScalar input (int-array [2 0]) 0)
(.putScalar input (int-array [2 1]) 1)

(.putScalar labels (int-array [2 0]) 0)
(.putScalar labels (int-array [2 1]) 1)

(.putScalar input (int-array [3 0]) 1)
(.putScalar input (int-array [3 1]) 1)

(.putScalar labels (int-array [3 0]) 1)
(.putScalar labels (int-array [3 1]) 0)

(def ds (DataSet. input labels))

(def builder (-> (NeuralNetConfiguration$Builder.)
                 (.iterations 10000)
                 (.learningRate 0.1)
                 (.seed 123)
                 (.useDropConnect false)
                 (.optimizationAlgo (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT))
                 (.biasInit 0)
                 (.miniBatch false)))

(def list-builder (.list builder))

(def hidden-layer-builder (-> (DenseLayer$Builder.)
                              (.nIn 2)
                              (.nOut 4)
                              (.activation "sigmoid")
                              (.weightInit (WeightInit/DISTRIBUTION))
                              (.dist (UniformDistribution. 0 1))))

(.layer list-builder 0 (.build hidden-layer-builder))

(def output-layer-builder (-> (OutputLayer$Builder. (LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD))
                              (.nIn 4)
                              (.nOut 2)
                              (.activation "softmax")
                              (.weightInit (WeightInit/DISTRIBUTION))
                              (.dist (UniformDistribution. 0 1))))

(.layer list-builder 1 (.build output-layer-builder))

(.pretrain list-builder false)

(.backprop list-builder true)

(def conf (.build list-builder))
(def net (MultiLayerNetwork. conf))
(.init net)

(.setListeners net (list (ScoreIterationListener. 100)))

(def layers (.getLayers net))

(def total-num-params (reduce + (map #(.numParams %) layers)))
(println (str "Total number of network parameters: " total-num-params))

(.fit net ds)

(def output (.output net (.getFeatureMatrix ds)))
(println output)

(def eval (Evaluation. 2))
(.eval eval (.getLabels ds) output)
(println (.stats eval))
