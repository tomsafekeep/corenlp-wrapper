java -cp $(sed "s/~/"${HOME//\//\\\/}"/g" cp.txt):target/corenlp-wrapper-1.0-SNAPSHOT.jar com.safekeep.nlp.corenlp.DatabaseTextProcessor "$1"
