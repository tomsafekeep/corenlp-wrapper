package com.safekeep.nlp.corenlp;

import com.github.jfasttext.JFastText;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.sql.*;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class DatabaseTextProcessor {
    static Logger logger = LoggerFactory.getLogger(DatabaseTextProcessor.class);

    public static String getPassword(String host, int port, String database, String username, File pgpass){
        // .pgpass file format, hostname:port:database:username:password
        var passwdFile = pgpass!=null?pgpass:new File(System.getenv("HOME"), ".pgpass");
        String passwd = null;
        try {
            for (var line : Files.readAllLines(passwdFile.toPath())){
                var config = line.strip().split(":");
                if (
                        !line.strip().startsWith("#")
                                && config.length==5
                                && host.equals(config[0])
                                && (config[1].equals("*") || Integer.toString(port).equals(config[1]))
                                && (config[2].equals("*") || database.equals(config[2]))
                                && (config[3].equals("*") || username.equals(config[3]))
                ) {
                    passwd = config[4];
                    break;
                }
            }
            return passwd;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
    public static Connection getConnection(String host, int port, String database, String username) throws SQLException {
        var password = getPassword(host, port, database, username, null);
        var conn = DriverManager.getConnection("jdbc:postgresql://"+host+":"+Integer.toString(port)+"/"+database, username, password);
        return conn;
    }
    public static Connection getConnection(String host, int port, String database, String username, File pgpass) throws SQLException {
        var password = getPassword(host, port, database, username, pgpass);
        var conn = DriverManager.getConnection("jdbc:postgresql://"+host+":"+Integer.toString(port)+"/"+database, username, password);
        return conn;
    }


    private Connection createInsertConnection() {
        try {
            var conn = getConnection(host, port, database, username, pgpass);
            conn.setAutoCommit(true);
            conn.setSchema(outputSchemaName);
            toClose.add(conn);
            return conn;
        } catch (SQLException throwables) {
            return null;
        }
    }

    private static ExecutorService currentThreadExecutorService() {
        ThreadPoolExecutor.CallerRunsPolicy callerRunsPolicy = new ThreadPoolExecutor.CallerRunsPolicy();
        return new ThreadPoolExecutor(0, 1, 0L, TimeUnit.SECONDS, new SynchronousQueue<Runnable>(), callerRunsPolicy) {
            @Override
            public void execute(Runnable command) {
                callerRunsPolicy.rejectedExecution(command, this);
            }
        };
    }
    final JFastText fasttext;
    final File kvLexiconFile;
    final String host, database, username;
    final int port;
    final String outputSchemaName;
    final ThreadLocal<Connection> pgis = ThreadLocal.withInitial(this::createInsertConnection);
    final Set<Connection> toClose = new HashSet<>();
    final CoreNLPWrapper nlp;
    final boolean prefixWithIsPreviousKey;
    final Mode mode;

    final File pgpass;

    @FunctionalInterface
    public interface SubmitFunction<C, T>{
        public void apply(String id, C content, Instant noteVersion, BlockingQueue<T> persistQueue);
    }
    @FunctionalInterface
    public interface PersistenceFunction<T>{
        public int apply(List<T> batch, Connection conn);
    }

    public record Replacement(Pattern pattern, String replacement){};
    public record TokenizedNote(String id, Instant noteVersion, List<String> tokens){};

    private void submitForSegmentation(String id, String content, Instant noteVersion, BlockingQueue<CoreNLPWrapper.NoteInformation> persistQueue){
        var segments = nlp.processUnSegmentedText(content, id, fasttext, prefixWithIsPreviousKey);
        var organized = nlp.organizeSegments(segments, id, noteVersion);
        try {
            persistQueue.put(organized);
        } catch (InterruptedException e) {
            e.printStackTrace();
            logger.warn("Inserting to persist queue was interrupted when trying to insert note {}", organized.order().get(0).noteid());
        }
    }
    private void submitForOrganization(String id, String[] content, Instant noteVersion, BlockingQueue<CoreNLPWrapper.NoteInformation> persistQueue){
        var segments = nlp.processPreSegmentedText(List.of(content), id, fasttext, prefixWithIsPreviousKey);
        var organized = nlp.organizeSegments(segments, id, noteVersion);
        try {
            persistQueue.put(organized);
        } catch (InterruptedException e) {
            e.printStackTrace();
            logger.warn("Inserting to persist queue was interrupted when trying to insert note {}", organized.order().get(0).noteid());
        }
    }
    private void submitForSingleSentenceTokenization(String id, String content, Instant noteVersion, BlockingQueue<TokenizedNote> persistQueue){
        var tokens = nlp.processText(content);
        try {
            persistQueue.put(new TokenizedNote(id, noteVersion, tokens));
        } catch (InterruptedException e) {
            e.printStackTrace();
            logger.warn("Inserting to persist queue was interrupted when trying to insert note {}", id);
        }
    }


    private void submitForTagging(String id, String content, Instant noteVersion, BlockingQueue<CoreNLPWrapper.TaggedNote> persistQueue){
        var mentions = nlp.tagPreTokenizedText(content);
        try {
            if (mentions.size()>0)
                persistQueue.put(new CoreNLPWrapper.TaggedNote(id, mentions));
        } catch (InterruptedException e) {
            e.printStackTrace();
            logger.warn("Inserting to persist queue was interrupted when trying to insert note {}", id);
        }
    }
    private static int persistTokens(List<TokenizedNote> toInsert, Connection pgi){
        var tokens_query = """
                INSERT INTO note_all_tokens
                (noteid, note_version, "content")
                VALUES(?, ?, ?)
                on conflict (noteid) do update set note_version=excluded.note_version, "content"=excluded."content"
                """;
        logger.debug("PersistSegments: batch of {}", toInsert.size());
        int inserted = 0;
        try (
                var mpo = pgi.prepareStatement(tokens_query);
        ) {
            for (var note : toInsert) {
                int ac = 1;
                mpo.setString(ac++, note.id);
                mpo.setTimestamp(ac++, new Timestamp(note.noteVersion.getEpochSecond()*1000));
                mpo.setString(ac++, String.join(" ", note.tokens).strip());
                mpo.addBatch();
            }
            for (int rows : mpo.executeBatch()) inserted += rows;
        } catch (SQLException throwables) {
            throwables.printStackTrace();
            throw  new IllegalStateException(throwables);
        }
        logger.debug("inserted token records: {}", inserted);
        return inserted;
    }

    private int persistTaggerMentions(List<CoreNLPWrapper.TaggedNote> toInsert, Connection pgi){
        var tokens_query = """
                INSERT INTO tagger_output
                (batch, sid, "begin", "end", matching_tokens, term, lexicon)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """;
        logger.debug("PersistSegments: batch of {}", toInsert.size());
        int inserted = 0;
        try (
                var mpo = pgi.prepareStatement(tokens_query);
        ) {
            for (var note : toInsert) {
                for (var mention:note.mentions()){
                    int ac = 1;
                    mpo.setString(ac++, this.kvLexiconFile.getName());
                    mpo.setString(ac++, note.noteid());
                    mpo.setInt(ac++, mention.begin());
                    mpo.setInt(ac++, mention.end());
                    mpo.setString(ac++, mention.surfaceForm());
                    mpo.setString(ac++, mention.entity());
                    mpo.setString(ac++, this.kvLexiconFile.getName());

                    mpo.addBatch();

                }
            }
            for (int rows : mpo.executeBatch()) inserted += rows;
        } catch (SQLException throwables) {
            throwables.printStackTrace();
            throw  new IllegalStateException(throwables);
        }
        logger.debug("inserted token records: {}", inserted);
        return inserted;
    }

    public <C, T> void processQuery(String sourceQuery, int nthreads, Function<ResultSet, C> contentExtractor, SubmitFunction<C, T> submitFunction, PersistenceFunction<T> persistenceFunction){
        int insert_batch_size = 1000;
        AtomicInteger total_inserted = new AtomicInteger();
        int processedRows = 0;
        try(
                Connection readConn = getConnection(host, port, database, username, pgpass);
        ){
            readConn.setAutoCommit(false);
            for (var stmt : List.of("set enable_seqscan=false", "set work_mem='4GB'", "set random_page_cost=1")) {
            	logger.info("Execute setup statement: {}", stmt);
            	try(var ps = readConn.prepareStatement(stmt)){
            		var success = ps.execute();
                	logger.info("Result: {}", success);
                }
        	}
            
            BlockingQueue<Runnable> rawQueue = new ArrayBlockingQueue<>(1000);
            BlockingQueue<T> persistQueue = new ArrayBlockingQueue<>(insert_batch_size*5);
            AtomicBoolean resultSetExhausted = new AtomicBoolean(false);
            AtomicInteger total_submitted = new AtomicInteger();

            int threadLimit = nthreads;
            logger.info("Execute source query:\n{}", sourceQuery);
            try(
                    var source = readConn.prepareStatement(sourceQuery); //ResultSet.TYPE_SCROLL_INSENSITIVE is not supported for cursor-based resultsets in PG
            ){
                source.setFetchSize(10000);
                var rs = source.executeQuery();
                AtomicInteger kvInsertBatch = new AtomicInteger();
                logger.info("Start executor service with {} threads", threadLimit);
                ExecutorService es = nthreads > 1 ? new ThreadPoolExecutor(nthreads, threadLimit,
                10000, TimeUnit.MILLISECONDS, rawQueue,
                            new ThreadPoolExecutor.CallerRunsPolicy())
                        : currentThreadExecutorService(); //Executors.newSingleThreadExecutor();
                while (!rs.isClosed() /* defensive against calls to rs.next() within contentExtractor */
                        && rs.next()){
                    int ac=1;
                    var id = rs.getString(ac++);
                    var noteVersion = rs.getTimestamp(ac++).toInstant();
                    var content = contentExtractor.apply(rs); //e.g. rs.getString(2);

                    if (content!=null) {
                        es.execute(()-> {
                            try {
								submitFunction.apply(id, content, noteVersion, persistQueue);
							} catch (Exception ex) {
								logger.info("Error while processing item |{}|:  {}", id, ex.getMessage());
								ex.printStackTrace();
							}
                            if (persistQueue.size() >= insert_batch_size) {
                                var batch = new ArrayList<T>(persistQueue.size());
                                synchronized (persistQueue) {
                                    // race condition - the queue might be drained between the check and the synchronized block
                                    if (persistQueue.size() >= insert_batch_size) {
                                        logger.info("Persist {} notes", persistQueue.size());
                                        persistQueue.drainTo(batch);
                                    }
                                }
                                if (batch.size()>0){
                                    var pgi = pgis.get();
                                    logger.info("Submit batch with {} notes ({} so far)", batch.size(), total_submitted.get());
                                    int inserted = persistenceFunction.apply(batch, pgi);
                                    total_inserted.addAndGet(inserted);
                                    total_submitted.addAndGet(batch.size());
                                }
                            }
                        });
                    }
                    processedRows++;
                    if (processedRows%10000==0){
                        logger.info("Processed {} notes", processedRows);
                    }
                }
                logger.info("Exhausted result set after {} rows. Await finishing {} tasks", processedRows, rawQueue.size());
                resultSetExhausted.set(true);
                //deprecate in favor of awaitTermination to wait for existing tasks: 
                es.shutdown();
                try {
                    es.awaitTermination(nthreads>1?300:1, TimeUnit.SECONDS);
                    if (persistQueue.size() > 0) {
                        var batch = new ArrayList<T>(persistQueue.size());
                        synchronized (persistQueue) {
                            logger.info("Persist {} notes", persistQueue.size());
                            persistQueue.drainTo(batch);
                        }
                        var pgi = pgis.get();
                        logger.info("Submit LAST batch with {} notes ({} so far)", batch.size(), total_submitted.get());
                        int inserted = persistenceFunction.apply(batch, pgi);
                        total_inserted.addAndGet(inserted);
                        total_submitted.addAndGet(batch.size());
                    }
                } catch (InterruptedException e) {
                    logger.warn("Executor service timed-out");
                }finally{
                    logger.info("Close {} thread-specific connections", toClose.size());
                    for (var pgi:toClose){
                        pgi.close();
                    }
                }
            }
        } catch (SQLException e) {
            e.printStackTrace();
			throw new IllegalStateException("SQL error");
        }
        if (processedRows>0 && total_inserted.get()==0){
            logger.info("0 records inserted. Exit with error");
            System.exit(1);
        }
    }

    public DatabaseTextProcessor(Mode mode, String host, int port, String database, String username, File pgpass, String outputSchemaName, File kvLexiconFile, File segmentClassifierFile, boolean prefixWithIsPreviousKey, Properties extraCoreNLPProperties) {
        this.host = host;
        this.port = port;
        this.database = database;
        this.pgpass = pgpass;
        this.username = username;
        this.outputSchemaName = outputSchemaName;
        this.nlp = new CoreNLPWrapper(kvLexiconFile, extraCoreNLPProperties);
        if (segmentClassifierFile!=null) {
            logger.info("Load FastText model from {}", segmentClassifierFile);
            fasttext = new JFastText();
            fasttext.loadModel(segmentClassifierFile.getAbsolutePath());
            logger.info("Loaded FastText model with {}", String.join(", ", fasttext.getModelName()));
        }else{
            fasttext = null;
        }
        this.kvLexiconFile = kvLexiconFile;
        this.mode = mode;
        this.prefixWithIsPreviousKey = prefixWithIsPreviousKey;
    }

    public DatabaseTextProcessor(Mode mode, String host, int port, String database, File pgpass, String username, String outputSchemaName, File kvLexiconFile, Properties properties) {
        this.host = host;
        this.port = port;
        this.database = database;
        this.pgpass = pgpass;
        this.username = username;
        this.outputSchemaName = outputSchemaName;

        var annotaorList = "tokenize,ssplit,regexner,entitymentions";
        Properties props = new Properties();
        props.setProperty("annotators", annotaorList);
        props.setProperty("regexner.mapping", kvLexiconFile.getAbsolutePath());
        props.setProperty("tokenize.whitespace", Boolean.toString(true));
        props.setProperty("ssplit.eolonly", Boolean.toString(true));
        props.setProperty("tokenize.options", "");
        logger.info("Instantiate CoreNLP with options: {}", props.toString());
        nlp = new CoreNLPWrapper(kvLexiconFile,props);

        this.kvLexiconFile = kvLexiconFile;
        fasttext=null;
        prefixWithIsPreviousKey=false;
        this.mode=mode;
    }

    public static String getContent(ResultSet rs){
        try {
            return rs.getString(3);
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    public enum Mode {RAW_TO_TOKENS, RAW_TO_SEGMENTS, CHUNKS_TO_SEGMENTS, TAG}
    public static void main(String[] args){
        // arguments for example
        String host="127.0.0.1", database="db", username="user", outputSchemaName="public";
        int port=5432;
        var notesQuery = """
                select\s
                	n.id, n."content", valid_from
                from notes n\s
                where not exists (select 1 from notes.note_segment_leaders s where s.noteid=n.id and n.valid_from>s.note_version)
                """;
        File kvLexiconFile = new File("");//lexicon file
        File segmentClassifierFile = new File("");//segment classifier
        boolean prefixWithIsPreviousKey = false;
        File pgpass = new File(System.getenv("HOME"), ".pgpass");
        Properties props = new Properties();
        File propsFile = new File(args[0]);
        Mode mode=Mode.RAW_TO_TOKENS;
        int nthreads = Runtime.getRuntime().availableProcessors();
        List<Replacement> replacements = List.of();
        boolean ignoreCase=false;
        Properties extraCoreNLPProperties = new Properties();
        try(BufferedReader br = Files.newBufferedReader(propsFile.toPath())){
            props.load(br);
            host = props.getProperty("host");
            port = Integer.parseInt(props.getProperty("port"));
            database = props.getProperty("database");
            username = props.getProperty("username");
            outputSchemaName = props.getProperty("outputSchema");
            notesQuery= props.getProperty("notesQuery");
            //insert query is hard coded ATM
            mode = Mode.valueOf(props.getProperty("mode").toUpperCase(Locale.ROOT));
            nthreads = Integer.parseInt(props.getProperty("nthreads"));
            if (props.containsKey("replacments")) {
                replacements=Arrays.stream(props.getProperty("replacements").strip().split("\t\t")).map(line -> {
                    try {
                        var arr = line.split("\t");
                        return new Replacement(Pattern.compile(arr[0]), arr[1]);
                    } catch (Exception e) {
                        logger.warn("Error parsing replacement pattern |{}|: {}", line, e.getMessage());
                        e.printStackTrace();
                        return null;
                    }
                })
                .filter(pat -> pat!=null)
                .collect(Collectors.toList());
            }
            ignoreCase=Boolean.parseBoolean(props.getProperty("kv_lexicon_ignore_case", Boolean.toString(false)));
            if (ignoreCase){
                extraCoreNLPProperties.put("regexner.ignorecase", Boolean.toString(ignoreCase));
            }
            switch (mode){
                case RAW_TO_SEGMENTS, CHUNKS_TO_SEGMENTS, TAG -> {
                    kvLexiconFile = new File(props.getProperty("kv_lexicon"));
                    String scPath = props.getProperty("segment_classifier");
                    segmentClassifierFile = scPath==null || scPath.trim().equals("null")?null:new File(scPath);
                    prefixWithIsPreviousKey = Boolean.parseBoolean(props.getProperty("prefix_with_is_previous_a_key", "false"));
                }
                default ->{}
            }
            if (props.contains("pgpass")){
                pgpass = new File(props.getProperty("pgpass"));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        DatabaseTextProcessor processor;
        switch (mode){
            case TAG -> {
                processor = new DatabaseTextProcessor(mode, host, port, database, pgpass, username, outputSchemaName, kvLexiconFile, (Properties)null);
                processor.processQuery(notesQuery, nthreads, DatabaseTextProcessor::getContent, processor::submitForTagging, processor::persistTaggerMentions);
            }
            case RAW_TO_SEGMENTS -> {
                processor = new DatabaseTextProcessor(mode, host, port, database, username, pgpass, outputSchemaName, kvLexiconFile, segmentClassifierFile, prefixWithIsPreviousKey, null);
                processor.processQuery(notesQuery, nthreads, DatabaseTextProcessor::getContent, processor::submitForSegmentation, CoreNLPWrapper::persistSegments);
            }
            case CHUNKS_TO_SEGMENTS -> {
                processor = new DatabaseTextProcessor(mode, host, port, database, username, pgpass, outputSchemaName, kvLexiconFile, segmentClassifierFile, prefixWithIsPreviousKey, null);
                AtomicReference<String> previousNoteid = new AtomicReference<>();
                processor.processQuery(notesQuery, nthreads, (ResultSet rs)->{
                    try {
                        /*
                        var arr =  rs.getArray(3);
                        return (String[]) arr.getArray();
                        */
                        List<String> content = new ArrayList<>();
                        var id = rs.getString(1);
                        while (true){
                            var lcontent = rs.getString(3);
                            content.add(lcontent);
                            if (rs.getString(1).equals(rs.getString("next_noteid")))
                                rs.next(); // the last line will have next_noteid==null, and therefore thepredicate will always fail.
                            else
                                break;
                        }
                        return content.toArray(new String[0]);
                    } catch (SQLException e) {
                        throw new RuntimeException(e);
                    }
                }, processor::submitForOrganization, CoreNLPWrapper::persistSegments);
            }
            case RAW_TO_TOKENS -> {
                extraCoreNLPProperties.put("ssplit.isOneSentence", Boolean.toString(true));
                processor = new DatabaseTextProcessor(mode, host, port, database, username, pgpass, outputSchemaName, null, null, prefixWithIsPreviousKey, extraCoreNLPProperties);
                processor.processQuery(notesQuery, nthreads,DatabaseTextProcessor::getContent,  processor::submitForSingleSentenceTokenization, DatabaseTextProcessor::persistTokens);
            }
            default -> throw new IllegalStateException("Unexpected value: " + mode);
        }
        logger.info("Finished. Exit JVMs");
        System.exit(0); 
    }
}
