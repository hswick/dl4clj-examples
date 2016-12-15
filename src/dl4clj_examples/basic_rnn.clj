(ns dl4clj.basic-rnn
  (:import [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration NeuralNetConfiguration$Builder Updater]
           [org.deeplearning4j.nn.conf.layers GravesLSTM$Builder RnnOutputLayer$Builder]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]))

(def learn-string (char-array "Der Cottbuser Postkutscher putzt den Cottbuser Postkutschkasten."))
(def learn-string-set (set learn-string))
(def learn-string-list (into () learn-string-set))


(def hidden-layer-width 50)
(def hidden-layer-cont 2)
(def r (rand 7894))

(def builder (-> (NeuralNetConfiguration$Builder.)
                 (.iterations 10)
                 (.learningRate 0.0001)
                 (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                 (.seed 123)
                 (.biasInit 0)
                 (.miniBatch false)
                 (.updater Updater/RMSPROP)
                 (.weightInit WeightInit/XAVIER)))

(def list-builder (.list builder))

;;Setup hidden layers
(dotimes [i hidden-layer-cont]
  (let [hidden-layer-builder (-> (GravesLSTM$Builder.)
                                 (.nIn (if (= i 0)
                                         (count learn-string-set)
                                         hidden-layer-width))
                                 (.nOut hidden-layer-width)
                                 (.activation "tanh"))]
    (.layer list-builder i (.build hidden-layer-builder))))

;:Setup output layer
(def output-layer-builder (-> (RnnOutputLayer$Builder. LossFunctions$LossFunction/MCXENT)
                              (.activation "softmax")
                              (.nIn hidden-layer-width)
                              (.nOut (count learn-string-set))))

(.layer list-builder hidden-layer-cont (.build output-layer-builder))

;;finish builder
(.pretrain list-builder false)
(.backprop list-builder true)

;;Create network
(def conf (.build list-builder))
(def net (MultiLayerNetwork. conf))
(.init net)
(.setListeners net (list (ScoreIterationListener. 1)))

(def input (Nd4j/zeros (int-array [1 (count learn-string-list) (count learn-string)])))
(def labels (Nd4j/zeros (int-array [1 (count learn-string-list) (count learn-string)])))

(dotimes [sample-pos (count learn-string)]
  (let [current-char (get learn-string sample-pos)
        next-char (get learn-string (mod (inc sample-pos) (count learn-string)))]
    (.putScalar input (int-array [0 (.indexOf learn-string-list current-char) sample-pos]) 1)
    (.putScalar labels (int-array [0 (.indexOf learn-string-list next-char) sample-pos]) 1)))

(def training-data (DataSet. input labels))

(defn find-index-of-highest-value [distribution]
  (first (apply max-key second (map-indexed vector distribution))))

(dotimes [epoch 200]
  (println (str "Epoch " epoch))

  
  ;;train the data
  (.fit net training-data)
  ;;clear current state from last example
  (.rnnClearPreviousState net)

  ;;Put first character into rnn as an initialisation
  (let [test-init (.putScalar (Nd4j/zeros (count learn-string-list))
                              (.indexOf learn-string-list (first learn-string)) 1)
        ;;run one step 
        output (atom (.rnnTimeStep net test-init))]

    ;;Now the net guesses learn-string length more characters
    (dotimes [j (count learn-string)]
      ;;first process the last output of the network to a concrete
      ;;neuron, the neuron with the highest output has the highest
      ;;chance to get chosen
      (let [output-prob-distribution (double-array (for [n (range (count learn-string-set))]
                                       (.getDouble @output n)))
            sample-character-index (find-index-of-highest-value output-prob-distribution)]
        (print (nth learn-string-list sample-character-index))
        (let [next-input (Nd4j/zeros (count learn-string-list))]
          (.putScalar next-input sample-character-index 1)
          (reset! output (.rnnTimeStep net next-input))))))
  (println))
