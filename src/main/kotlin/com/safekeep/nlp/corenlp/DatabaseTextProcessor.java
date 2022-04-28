package com.safekeep.nlp.corenlp;

import com.github.jfasttext.JFastText;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
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


    private Connection createInsertConnection() {
        try {
            var conn = getConnection(host, port, database, username);
            conn.setAutoCommit(true);
            toClose.add(conn);
            return conn;
        } catch (SQLException throwables) {
            return null;
        }
    }

    final JFastText fasttext;
    final String host, database, username;
    final int port;
    final ThreadLocal<Connection> pgis = ThreadLocal.withInitial(this::createInsertConnection);
    final Set<Connection> toClose = new HashSet<>();
    final CoreNLPWrapper nlp;
    final Mode mode;
    public void processQueryFile(String query, String kvInsertQuery, File outputPath){
        try(
            Connection readConn = getConnection(host, port, database, username);
            Connection kvConn = getConnection(host, port, database, username);
            BufferedWriter textWriter = Files.newBufferedWriter(outputPath.toPath(), StandardCharsets.UTF_8)
        ){
            readConn.setAutoCommit(false);
            BlockingQueue<Runnable> queue = new ArrayBlockingQueue<>(1000);
            int nthreads = Runtime.getRuntime().availableProcessors() ;
            int threadLimit = nthreads;
            try(
                var source = readConn.prepareStatement(query);
                var kvs = readConn.prepareStatement(kvInsertQuery);
            ){
                var rs = source.executeQuery();
                int processedRows = 0;
                AtomicInteger kvInsertBatch = new AtomicInteger();
                int batchSize = 1000;
                logger.info("Start executor service with {} threads", threadLimit);
                ThreadPoolExecutor es = new ThreadPoolExecutor(nthreads, threadLimit, 10000, TimeUnit.MILLISECONDS, queue,
                        new ThreadPoolExecutor.CallerRunsPolicy());
                while (rs.next()){
                    var content = rs.getString(1);
                    var id = rs.getString(2);
                    queue.add(()-> {
                        try {
                            nlp.processText(content, id, textWriter, tmentions->{
                                synchronized (kvs) {
                                    for (var mention:tmentions) {
                                        int pi=1;
                                        try {
                                            kvs.setString(pi++, id);
                                            kvs.setInt(pi++, mention.begin());
                                            kvs.setInt(pi++, mention.end());
                                            kvs.setString(pi++, mention.entityType());
                                            kvs.setString(pi++, mention.surfaceForm());
                                            kvs.addBatch();
                                            int current = kvInsertBatch.incrementAndGet();
                                            if (current>=batchSize){
                                                logger.info("Persisting KV query with {} rows", current);
                                                var inserted = kvs.executeBatch();
                                                logger.info("Inserted {} rows", inserted);
                                                kvs.clearBatch();
                                                kvInsertBatch.set(0);
                                            }
                                        } catch (SQLException e) {
                                            e.printStackTrace();
                                        }
                                    }
                                }
                            });
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    });
                }
                logger.info("Await finishing {} tasks", queue.size());
                es.shutdown();
                try {
                    es.awaitTermination(300, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    logger.warn("Executor service timed-out");
                }finally{
                    int current = kvInsertBatch.incrementAndGet();
                    if (current>=0){
                        logger.info("Persisting last KV query batch with {} rows", current);
                        var inserted = kvs.executeBatch();
                        logger.info("Inserted {} rows", inserted);
                        kvs.clearBatch();
                        kvInsertBatch.set(0);
                    }
                }
            }
        } catch (SQLException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @FunctionalInterface
    public interface SubmitFunction<T>{
        public void apply(String id, String content, Instant noteVersion, BlockingQueue<T> persistQueue);
    }
    @FunctionalInterface
    public interface PersistenceFunction<T>{
        public int apply(List<T> batch, Connection conn);
    }

    public record Replacement(Pattern pattern, String replacement){};
    public record TokenizedNote(String id, Instant noteVersion, List<String> tokens){};

    private void submitForSegmentation(String id, String content, Instant noteVersion, BlockingQueue<CoreNLPWrapper.NoteInformation> persistQueue){
        var segments = nlp.processTextE2E(content, id, fasttext);
        var organized = nlp.organizeSegments(segments, noteVersion);
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

    private int persistTokens(List<TokenizedNote> toInsert, Connection pgi){
        var tokens_query = """
                INSERT INTO notes.note_all_tokens
                (noteid, note_version, "content")
                VALUES(?, ?, ?)
                on conflict do nothing
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
        }
        logger.debug("inserted token records: {}", inserted);
        return inserted;
    }
    public <T> void processQuery(String sourceQuery, int nthreads, SubmitFunction<T> submitFunction, PersistenceFunction<T> persistenceFunction){
        int batchSize = 1000;
        int insert_batch_size = 20000;
        try(
                Connection readConn = getConnection(host, port, database, username);
        ){
            readConn.setAutoCommit(false);
            BlockingQueue<Runnable> rawQueue = new ArrayBlockingQueue<>(1000);
            BlockingQueue<T> persistQueue = new ArrayBlockingQueue<>(insert_batch_size*5);
            AtomicBoolean resultSetExhausted = new AtomicBoolean(false);
            AtomicInteger total_submitted = new AtomicInteger();
            Runnable persistorFunc = ()->{
                Object obj = new Object();
                while (!resultSetExhausted.get()) {
                    if (persistQueue.size() >= insert_batch_size) {
                        var batch = new ArrayList<T>(persistQueue.size());
                        synchronized (persistQueue) {
                            logger.info("Persist {} notes", persistQueue.size());
                            persistQueue.drainTo(batch);
                        }
                        var pgi = pgis.get();
                        logger.info("Submit batch with {} notes ({} so far)", batch.size(), total_submitted.get());
                        int inserted = persistenceFunction.apply(batch, pgi);
                        total_submitted.addAndGet(batch.size());
                    }
                    if (nthreads>1) {
                        try {
                            Thread.sleep(1000);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
                if (persistQueue.size() > 0) {
                    var batch = new ArrayList<T>(persistQueue.size());
                    synchronized (persistQueue) {
                        logger.info("Persist last batch with {} notes", persistQueue.size());
                        persistQueue.drainTo(batch);
                    }
                    var pgi = pgis.get();
                    logger.info("Submit batch with {} notes ({} so far)", batch.size(), total_submitted.get());
                    int inserted = persistenceFunction.apply(batch, pgi);
                    total_submitted.addAndGet(batch.size());
                }
            };
            Thread persistor = new Thread(persistorFunc);
            logger.info("Start persistor thread");

            int threadLimit = nthreads;
            try(
                    var source = readConn.prepareStatement(sourceQuery);
            ){
                source.setFetchSize(10000);
                var rs = source.executeQuery();
                int processedRows = 0;
                AtomicInteger kvInsertBatch = new AtomicInteger();
                logger.info("Start executor service with {} threads", threadLimit);
                ExecutorService es = nthreads > 1 ? new ThreadPoolExecutor(nthreads, threadLimit,
            10000, TimeUnit.MILLISECONDS, rawQueue,
                        new ThreadPoolExecutor.CallerRunsPolicy())
                        : Executors.newSingleThreadExecutor();
                logger.info("Start persistor thread");
                if (nthreads>1) {
                    persistor.start();
                }
                while (rs.next()){
                    var content = rs.getString(2);
                    var id = rs.getString(1);
                    var noteVersion = rs.getTimestamp(3).toInstant();

                    es.submit(()-> {
                        submitFunction.apply(id, content, noteVersion, persistQueue);
                    });
                    processedRows++;
                    if (processedRows%1000==0){
                        logger.info("Processed {} notes", processedRows);
                    }
                }
                logger.info("Exhausted result set after {} rows. Await finishing {} tasks", processedRows, rawQueue.size());
                resultSetExhausted.set(true);
                es.shutdown();
                try {
                    es.awaitTermination(300, TimeUnit.SECONDS);
                    if (nthreads>1) {
                        persistor.join(300000);
                    }else{
                        persistorFunc.run();
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
    }

    @Deprecated
    public void processQueryE2E(String query, int nthreads){
        int batchSize = 1000;
        int insert_batch_size = 1000;
        try(
            Connection readConn = getConnection(host, port, database, username);
        ){
            readConn.setAutoCommit(false);
            BlockingQueue<Runnable> rawQueue = new ArrayBlockingQueue<>(1000);
            BlockingQueue<CoreNLPWrapper.NoteInformation> persistQueue = new ArrayBlockingQueue<>(1000);
            AtomicBoolean resultSetExhausted = new AtomicBoolean(false);
            AtomicInteger total_submitted = new AtomicInteger();
            Runnable persistorFunc = ()->{
                Object obj = new Object();
                while (!resultSetExhausted.get()) {
                    if (persistQueue.size() >= insert_batch_size) {
                        var batch = new ArrayList<CoreNLPWrapper.NoteInformation>(persistQueue.size());
                        synchronized (persistQueue) {
                            logger.info("Persist {} notes", persistQueue.size());
                            persistQueue.drainTo(batch);
                        }
                        var pgi = pgis.get();
                        logger.info("Submit batch with {} notes ({} so far)", batch.size(), total_submitted.get());
                        int inserted = nlp.persistSegments(batch, pgi);
                        total_submitted.addAndGet(batch.size());
                    }
                    if (nthreads>1) {
                        try {
                            Thread.sleep(1000);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
                if (persistQueue.size() > 0) {
                    var batch = new ArrayList<CoreNLPWrapper.NoteInformation>(persistQueue.size());
                    synchronized (persistQueue) {
                        logger.info("Persist last batch with {} notes", persistQueue.size());
                        persistQueue.drainTo(batch);
                    }
                    var pgi = pgis.get();
                    logger.info("Submit batch with {} notes ({} so far)", batch.size(), total_submitted.get());
                    int inserted = nlp.persistSegments(batch, pgi);
                    total_submitted.addAndGet(batch.size());
                }
            };
            Thread persistor = new Thread(persistorFunc);
            logger.info("Start persistor thread");

            int threadLimit = nthreads;
            try(
                var source = readConn.prepareStatement(query);
            ){
                source.setFetchSize(10000);
                var rs = source.executeQuery();
                int processedRows = 0;
                AtomicInteger kvInsertBatch = new AtomicInteger();
                logger.info("Start executor service with {} threads", threadLimit);
                ExecutorService es = nthreads > 1 ? new ThreadPoolExecutor(nthreads, threadLimit, 10000, TimeUnit.MILLISECONDS, rawQueue,
                        new ThreadPoolExecutor.CallerRunsPolicy())
                        : Executors.newSingleThreadExecutor();
                logger.info("Start persistor thread");
                if (nthreads>1) {
                    persistor.start();
                }
                while (rs.next()){
                    var content = rs.getString(2);
                    var id = rs.getString(1);
                    var noteVersion = rs.getTimestamp(3).toInstant();

                    es.submit(()-> {
                        var segments = nlp.processTextE2E(content, id, fasttext);
                        var organized = nlp.organizeSegments(segments, noteVersion);
                        try {
                            persistQueue.put(organized);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                            logger.warn("Inserting to persist queue was interrupted when trying to insert note {}", organized.order().get(0).noteid());
                        }
                    });
                    processedRows++;
                    if (processedRows%1000==0){
                        logger.info("Processed {} notes", processedRows);
                    }
                }
                logger.info("Exhausted result set after {} rows. Await finishing {} tasks", processedRows, rawQueue.size());
                resultSetExhausted.set(true);
                es.shutdown();
                try {
                    es.awaitTermination(300, TimeUnit.SECONDS);
                    if (nthreads>1) {
                        persistor.join(300000);
                    }else{
                        persistorFunc.run();
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
        }
    }

    public DatabaseTextProcessor(Mode mode, String host, int port, String database, String username, File kvLexiconFile, File segmentClassifierFile, Properties extraCoreNLPProperties) {
        this.host = host;
        this.port = port;
        this.database = database;
        this.username = username;
        this.nlp = new CoreNLPWrapper(kvLexiconFile, extraCoreNLPProperties);
        if (segmentClassifierFile!=null) {
            fasttext = new JFastText();
            fasttext.loadModel(segmentClassifierFile.getAbsolutePath());
        }else{
            fasttext = null;
        }
        this.mode = mode;
    }

    public enum Mode {RAW_TO_TOKENS, RAW_TO_SEGMENTS}
    public static void main(String[] args){
        String host="127.0.0.1", database="db", username="user";
        int port=5432;
        var notesQuery = """
                select\s
                	n.id, n."content", valid_from
                from notes n\s
                where not exists (select 1 from notes.note_segment_leaders s where s.noteid=n.id and n.valid_from>s.note_version)
                """;
        Properties props = new Properties();
        File propsFile = new File(args[0]);
        Mode mode=Mode.RAW_TO_TOKENS;
        int nthreads = Runtime.getRuntime().availableProcessors();
        List<Replacement> replacements = List.of();
        boolean ignoreCase=false;
        try(BufferedReader br = Files.newBufferedReader(propsFile.toPath())){
            props.load(br);
            host = props.getProperty("host");
            port = Integer.parseInt(props.getProperty("port"));
            database = props.getProperty("database");
            username = props.getProperty("username");
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
        } catch (IOException e) {
            e.printStackTrace();
        }
        DatabaseTextProcessor processor;
        switch (mode){
            case RAW_TO_SEGMENTS -> {
                File kvLexiconFile = new File("");//lexicon file
                File segmentClassifierFile = new File("");//segment classifier
                Properties extraCoreNLPProperties = new Properties();
                if (ignoreCase){
                    extraCoreNLPProperties.put("regexner.ignorecase", Boolean.toString(ignoreCase));
                }
                processor = new DatabaseTextProcessor(mode, host, port, database, username, kvLexiconFile, segmentClassifierFile, extraCoreNLPProperties);
                processor.processQuery(notesQuery, nthreads, processor::submitForSegmentation, CoreNLPWrapper::persistSegments);
            }
            case RAW_TO_TOKENS -> {
                Properties extraCoreNLPProperties = new Properties();
                extraCoreNLPProperties.put("ssplit.isOneSentence", Boolean.toString(true));
                processor = new DatabaseTextProcessor(mode, host, port, database, username, null, null, extraCoreNLPProperties);
                processor.processQuery(notesQuery, nthreads, processor::submitForSingleSentenceTokenization, processor::persistTokens);
            }
            default -> throw new IllegalStateException("Unexpected value: " + mode);
        }
        logger.info("Finished");
    }
}
