(defproject dl4clj-examples "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/tools.logging "0.3.1"]
                 [org.datavec/datavec-api "0.7.1"]
                 [org.deeplearning4j/deeplearning4j-nlp "0.7.1"]
                 [org.deeplearning4j/deeplearning4j-core "0.7.1"]
                 [org.nd4j/nd4j-native-platform "0.7.1"]
                 [org.slf4j/slf4j-log4j12 "1.7.1"]
                 [log4j/log4j "1.2.17" :exclusions [javax.mail/mail
                                                    javax.jms/jms
                                                    com.sun.jmdk/jmxtools
                                                    com.sun.jmx/jmxri]]])
