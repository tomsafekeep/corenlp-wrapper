java -Dcom.sun.management.jmxremote.port=8201 \
	-Dcom.sun.management.jmxremote.ssl=false \
	-Dcom.sun.management.jmxremote.authenticate=false \
	-cp $(sed "s/~/"${HOME//\//\\\/}"/g" cp.txt):target/corenlp-wrapper-1.0-SNAPSHOT.jar com.safekeep.nlp.corenlp.DatabaseTextProcessor "$1"
