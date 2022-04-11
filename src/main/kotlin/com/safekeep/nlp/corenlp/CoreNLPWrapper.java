package com.safekeep.nlp.corenlp;

import com.github.jfasttext.JFastText;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.RegexNERAnnotator;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.time.Instant;
import java.util.*;
import java.util.function.BiFunction;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class CoreNLPWrapper {

    private static final char FIELD_SEPARATOR = '\t';
    private static final char SENTENCE_SEPARATOR = '\n';
    private static final char TOKEN_SEPARATOR = ' ';
    protected static Logger logger = LoggerFactory.getLogger(CoreNLPWrapper.class);
    record Segment(SegmentType type, String noteid, float pos, String content, String id){}

    record NoteSegment(String noteid, int pos, Segment main, List<Segment> values){}
    record NoteInformation(List<Segment> content, List<NoteSegment> order, Instant note_version){}

    private StanfordCoreNLP nlp;
    final boolean useTagger;
    private static String quotedSchema;
    Properties props = new Properties();

    public CoreNLPWrapper(File kvLexicon, Properties extraCoreNLPProperties) {
        var annotaorList = "tokenize,ssplit";
        if (kvLexicon != null) {
            //annotaorList = annotaorList + ",pos,lemma,ner";
            annotaorList = annotaorList + ",regexner";
            props.setProperty("regexner.mapping", kvLexicon.getAbsolutePath());
            useTagger = true;
        } else {
            useTagger = false;
        }
        props.setProperty("annotators", annotaorList);
        Properties tokenizerProps = new Properties();
        tokenizerProps.setProperty("normalizeParentheses", Boolean.toString(false));
        tokenizerProps.setProperty("normalizeOtherBrackets", Boolean.toString(false));
        tokenizerProps.setProperty("strictTreebank3", Boolean.toString(false));
        tokenizerProps.setProperty("splitHyphenated", Boolean.toString(false));
        props.setProperty("tokenize.options", tokenizerProps.toString().substring(1, tokenizerProps.toString().length() - 1));
        props.setProperty("ssplit.boundariesToDiscard", "END_OF_SENTENCE");
        //props.setProperty("threads", Integer.toString(Runtime.getRuntime().availableProcessors()));
        //props.setProperty("quiet", "true");
        if (extraCoreNLPProperties!=null) {
            props.putAll(extraCoreNLPProperties);
        }
        logger.info("Instantiate CoreNLP with options: {}", props.toString());
        nlp = new StanfordCoreNLP(props);
        logger.info("CoreNLP options: {}", nlp.getProperties());

        RegexNERAnnotator a;
    }

    public record TaggerMention(String entityType, int begin, int end, String surfaceForm) {
    }

    ;

    @FunctionalInterface
    public static interface MentionConsumer {
        public void accept(List<TaggerMention> mentions);
    }

    public void processText(String text, String id, Writer writer, MentionConsumer mentionConsumer) throws IOException {

        CoreDocument doc = new CoreDocument(text);
        nlp.annotate(doc);
        Annotation anns = doc.annotation();
        //The basic unit of processing is a token, so this iteration is hard coded.
        int sentenceCounter = 0;

        for (CoreMap sentence : anns.get(CoreAnnotations.SentencesAnnotation.class)) {
            var sentenceid = String.format("%s%s%d", id.replace(FIELD_SEPARATOR, ' '), FIELD_SEPARATOR, sentenceCounter);
            if (sentenceid != null) {
                writer.append(sentenceid);
                writer.append(FIELD_SEPARATOR);
            }
            for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                String t = token.getString(CoreAnnotations.TextAnnotation.class);
                if (t != null && t.length() > 0) {
                    writer.append(t);
                    writer.append(TOKEN_SEPARATOR);
                }
            }
            writer.append(SENTENCE_SEPARATOR);
        }
        var mentions = doc.entityMentions();
        if (useTagger) {
            var tmentions = mentions.stream().map(mention -> {
                //for (var mention:mentions){
                var mtokens = mention.tokens();
                if (mtokens.size() > 0) {
                    return new TaggerMention(mention.entityType(), mtokens.get(0).index(), mtokens.get(mtokens.size() - 1).index(), mention.text());
                } else
                    return null;
            }).filter(m -> m != null).collect(Collectors.toList());
            if (
                    tmentions != null
                            && tmentions.size() > 0) {
                mentionConsumer.accept(tmentions);
            }
        }
    }

    public List<String> processText(String text) {
        CoreDocument doc = new CoreDocument(text);
        nlp.annotate(doc);
        Annotation anns = doc.annotation();
        //The basic unit of processing is a token, so this iteration is hard coded.
        var tokens = doc.tokens().stream().map(token -> token.word()).collect(Collectors.toList());
        return tokens;
    }
    Map<String, SegmentType> ftLabel2SegmentType = Map.of(
            "__label__narrative", SegmentType.narrative, "__label__list_item", SegmentType.list_item,
            "__label__value", SegmentType.value, "__label__key_value", SegmentType.KV,
            "__label__list_header", SegmentType.list_header, "__label__non_informative", SegmentType.non_informative,
            "__label__header", SegmentType.header, "__label__key", SegmentType.key
    );

    /**
     * Assume: each segment is >=1 sentences, but sentences never span a segment. Therefore:
     * 1. Split to sentences.
     * 2. Classify segments.
     * @param texts
     * @param id
     * @param segmentClassifier
     * @return
     */
    public List<Segment> processPreSegmentedText(List<String> texts, String id, JFastText segmentClassifier, boolean prefixWithIsPreviousKey) {
        var segments = new ArrayList<Segment>(texts.size());
        var classifier = fasttextInstance2ClassifierFunction(segmentClassifier, prefixWithIsPreviousKey);
        for (var text:texts) {
            CoreDocument doc = new CoreDocument(text);
            nlp.annotate(doc);
            Annotation anns = doc.annotation();
            var docSegments = organizeDocumentSegments(id, classifier, doc);
            for(var segment:docSegments)
                segments.add(new Segment(segment.type, segment.noteid, segments.size() /*position across all texts */, segment.content, segment.id));
        }
        return segments;
    }

    @NotNull
    private BiFunction<String, SegmentType, String> fasttextInstance2ClassifierFunction(JFastText segmentClassifier, boolean prefixWithIsPreviousKey) {
        return segmentClassifier != null ?
                (val, previousLabel) -> segmentClassifier.predict(
                        prefixWithIsPreviousKey && previousLabel!=null?
                            String.join(" ", previousLabel.name(), val)
                            :val)
                :
                (val, previousLabel) -> previousLabel==SegmentType.key?
                        "__label__value":
                        "__label__narrative";
    }

    public List<Segment> processUnSegmentedText(String text, String id, JFastText segmentClassifier, boolean prefixWithIsPreviousKey) {
        CoreDocument doc = new CoreDocument(text);
        nlp.annotate(doc);
        Annotation anns = doc.annotation();
        var classifier = fasttextInstance2ClassifierFunction(segmentClassifier, prefixWithIsPreviousKey);
        return organizeDocumentSegments(id, classifier, doc);
    }

    @NotNull
    private List<Segment> organizeDocumentSegments(String id, BiFunction<String, SegmentType, String> segmentClassifier, CoreDocument doc) {
        int sentenceNum = 0;
        List<Segment> segments = new ArrayList<>();
        SegmentType previousSpanLabel = null;
        for (var sent: doc.sentences()){
            boolean insideMention = false;
            List<String> spanTokens = new ArrayList<>();

            int ti = 0;
            for (var token:sent.coreMap().get(CoreAnnotations.TokensAnnotation.class)) {
                String word = token.word();
                boolean isMention = token.containsKey(CoreAnnotations.NamedEntityTagAnnotation.class);
                if (insideMention){//must not be the first token
                    if (!isMention){//Key -> value
                        //Seal key
                        {
                            String span = String.join(" ", spanTokens);
                            //By definition this is a key: var label = segmentClassifier.predict(span);
                            var segment = new Segment(SegmentType.key, id, ((float) segments.size()) , span, String.format("%s#%4d", id, segments.size()));
                            segments.add(segment);
                            spanTokens.clear();
                            previousSpanLabel = SegmentType.key;
                        }
                        insideMention=false;
                        spanTokens.add(word);
                    }else{
                        //continue
                        spanTokens.add(word);
                    }
                }else{
                    if (isMention){
                        if (spanTokens.size()>0){
                            String span = String.join(" ", spanTokens);
                            var label = segmentClassifier.apply(span, previousSpanLabel);
                            var stype = ftLabel2SegmentType.get(label);
                            if (previousSpanLabel==SegmentType.key && stype==SegmentType.narrative)
                                stype = SegmentType.value; //narrative and values are similar - prioritize value.
                            var segment = new Segment(stype, id, ((float) segments.size()) , span, String.format("%s#%4d", id, segments.size()));
                            segments.add(segment);
                            previousSpanLabel= stype;
                            spanTokens.clear();
                        }
                        //Begin key
                        spanTokens.add(word);
                        insideMention=true;
                    }else{//continue value
                        spanTokens.add(word);
                    }
                }
                ti++;

            }
            if (spanTokens.size()>0){
                String span = String.join(" ", spanTokens);
                SegmentType segmentType = ftLabel2SegmentType.get(segmentClassifier.apply(span, previousSpanLabel));
                if (previousSpanLabel==SegmentType.key && segmentType==SegmentType.narrative) {
                    segmentType = SegmentType.value; //narrative and values are similar - prioritize value.
                }
                var segment = new Segment(insideMention?SegmentType.key: segmentType, id, ((float) segments.size()) , span, String.format("%s#%4d", id, segments.size()));
                segments.add(segment);
                previousSpanLabel = segmentType;
                spanTokens.clear();
            }
            sentenceNum++;
        }
        return segments;
    }

    Pattern kvpattern = Pattern.compile("^([a-zA-Z ]+)[:?](.+)");
    NoteInformation organizeSegments(List<Segment> segments, Instant noteVersion) {
        /*
        Imperative algorithm:
        1. Iterate over segments in sequential order.
        2. Keep track of the segment leader:
        2.1.    Qualifying segments:
		2.1.1.		'key'
					'key and value (key from beginning until the first :/?)',
					'list header',
		2.2. Add to the main segments list:
			Qualified leaders.
			Leader-breaking segment types:
				'narrative',
				'header',
				'other'
		3. For each segment:
			3.1 If it's type is a valid follower of the leader: add it to the leader.
				'value':
					'key'
					'key and value (key from beginning until the first :/?)',
				'list item':
					'list header'
			3.1 KV Pair:
				Try to parse the value and add it first to the segments followers.
			3.2 If it's a leader:
				replace the current leader.
				Add this segment to the main list.
			3.3 Leader-breaking segment types:
				Add this segment to the main list.
				Nullify the current leader.
			3.4	'non-informative':
				Discard
         */
        NoteSegment leader = null;
        var main = new ArrayList<NoteSegment>(segments.size() / 2);
        var content = new ArrayList<Segment>();
        Segment currentHeader;
        Collections.sort(segments, (s1, s2) -> Float.compare(s1.pos, s2.pos));
        for (Segment segment : segments) {
            if (segment.type.discard/*==SegmentType.non_informative*/) {
                continue;
            } else {
                NoteSegment link = null;
                if (segment.type == SegmentType.KV /* in thiscase most keys come directly from the tagger || segment.type == SegmentType.key*/) {
                    var matcher = kvpattern.matcher(segment.content);
                    if (matcher.find() && matcher.group(2).strip().length() > 0) {
                        Segment modifiedType = new Segment(SegmentType.key, segment.noteid, segment.pos, matcher.group(1), segment.id);
                        content.add(modifiedType);
                        link = new NoteSegment(segment.noteid, main.size(), modifiedType, new ArrayList<>(1));
                        Segment derivedValue = new Segment(
                                SegmentType.value,
                                segment.noteid,
                                segment.pos + 0.5f,
                                matcher.group(2),
                                segment.id + "/value"
                        );
                        content.add(derivedValue);
                        link.values.add(derivedValue);
                        segment = modifiedType;
                    } else {
                        // Convert type to key. Use this branch to handle edge cases like segments classified as key despite having no [:?] in them.
                        Segment modifiedType = new Segment(SegmentType.key, segment.noteid, segment.pos, segment.content, segment.id);
                        link = new NoteSegment(segment.noteid, main.size(), modifiedType, new ArrayList<>(1));
                        content.add(modifiedType);
                        segment = modifiedType;
                    }
                } else {//Anything else besides non-informative
                    content.add(segment);
                    link = new NoteSegment(segment.noteid, main.size(), segment, new ArrayList<>(0));
                }
                if (segment.type.isLeader) {
                    leader = link;
                    main.add(link);
                } else if (segment.type.isLeaderBreaking) {
                    leader = null;
                    main.add(link);
                } else if (leader != null) {
                    if (segment.type.leaders.contains(leader.main.type)) {
                        leader.values.add(segment);
                    } else {
                        /*invalid state: While there could be values matching the current leader after this value
                        (e.g. due to mis-classification of the value segment), keeping the current leader may lead
                        to non-linear flow of the segments (later segment attached to a previous key. This should not
                        happen, so for now fail-fast and break the key-value assocation.
                        */
                        leader = null;
                        main.add(link);
                    }
                } else { // default: the leader must be null, so no need to nullify it.
                    main.add(link);
                }
            }
        }

        Collections.sort(content, (s1, s2) -> Float.compare(s1.pos, s2.pos));
        logger.debug("Processed: content: {}, order: {} items", content.size(), main.size());
        return new NoteInformation(content, main, noteVersion);
    }

    public static int persistSegments(List<NoteInformation> toInsert, Connection pgi) {
        var leader_query ="""
                INSERT INTO note_segment_leaders
                (note_version, noteid, segment_num, leader_type, "content")
                VALUES(?, ?, ?, (?)::segment_type, ?)
                on conflict do nothing
                """;
        var follower_query = """
                INSERT INTO note_segment_followers
                (noteid, segment_num, sentence_num, follower_type, "content")
                VALUES(?, ?, ?, (?)::segment_type, ?)
                on conflict do nothing
                """;
        int inserted_segments = 0;
        int inserted_order = 0;
        logger.debug("PersistSegments: batch of {}", toInsert.size());
        if (toInsert.size()==0 || toInsert.stream().mapToInt(note-> note.order.size()).sum()==0)
            return 1;
        try (
             var mpo = pgi.prepareStatement(leader_query);
             var fpo = pgi.prepareStatement(follower_query)
        ) {
            for (var note : toInsert) {
                int pos = 0;
                for (var segment : note.order) {
                    int ac = 1;
                    mpo.setTimestamp(ac++, new Timestamp(note.note_version.getEpochSecond()*1000));
                    mpo.setString(ac++, segment.noteid);
                    mpo.setShort(ac++, (short)segment.pos);
                    mpo.setString(ac++, segment.main.type.sqlName);
                    if (segment.values != null && segment.values.size() > 0){
                        /*if (segment.main.type==SegmentType.KV || segment.main.type==SegmentType.key)*/ {
                            short fsc = 1;
                            for (var follower : segment.values) {
                                int fc = 1;
                                fpo.setString(fc++, follower.noteid);
                                fpo.setShort(fc++, (short)segment.pos);
                                fpo.setShort(fc++, fsc);
                                fpo.setString(fc++, follower.type.sqlName);
                                fpo.setString(fc++, follower.content.strip());
                                fpo.addBatch();
                                fsc++;
                            }
                        }
                    }
                    mpo.setString(ac++, segment.main.content.strip());
                    mpo.addBatch();
                }
            }
            for (int rows : mpo.executeBatch()) inserted_order += rows;
            fpo.executeBatch();
        } catch (SQLException throwables) {
            throwables.printStackTrace();
        }
        logger.debug("inserted_order: {}", inserted_segments, inserted_order);
        return inserted_order;
    }
}