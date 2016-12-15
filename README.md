# DL4CLJ-EXAMPLES

## Rationale

The resources for clojure+deeplearning4j are rather thin, and the wild west of the web is not always to be trusted. The Java ecosystem has great tools, and you can't always wait for someone to update their wrapper. Going the interop path can be a bit scary at times, the purpose of these examples are to make the journey smoother. Clojure's Java Interop capabilities make working with Java code a breeze. And Clojure's LISP roots really shine when the abstractions of the dl4j code base become simply a list of S-Expressions. However, there are subtle nuances when porting the Java code to Clojure that can trip up new users. Hopefully others can use this repo as a reference in their own machine learning endeavors.

### What to watch out for

- Builder Classes
- Static Methods
- Primitive arrays
- Mutability
  *Definitely one of the more awkward things to port from Java

## Other Relevant Resources
[deeplearning4j homepage](https://deeplearning4j.org/)
[deeplearning4j Javadocs](https://deeplearning4j.org/doc/)
[deeplearning4j general gitter](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[deeplearning4j tuninghelp gitter](https://gitter.im/deeplearning4j/deeplearning4j/tuninghelp)
[Extensive Clojure wrapper of deeplearning4j](https://github.com/engagor/dl4clj) (a little out of date currently)

