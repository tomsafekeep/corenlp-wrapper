package com.safekeep.nlp.corenlp;

import java.util.Set;

enum SegmentType {
    key("key", true, true, false, Set.of()),
    KV("key and value (key from beginning until the first :/?)", true, true, false, Set.of()),
    list_header("list header", true, true, false, Set.of()),
    narrative("narrative", false, true, false, Set.of()),
    header("header", false, true, false, Set.of()),
    other("other", false, true, false, Set.of()),
    non_informative("non-informative", false, false, true, Set.of()),
    value("value", false, false, false, Set.of(key, KV)),
    list_item("list item", false, false, false, Set.of(list_header))
    ;
    final String sqlName;
    final boolean isLeader, isLeaderBreaking, discard;
    Set<SegmentType> leaders;
    SegmentType(String sqlName, boolean isLeader, boolean isLeaderBreaking, boolean discard, Set<SegmentType> leaders) {
        this.sqlName = sqlName;
        this.isLeader = isLeader;
        this.isLeaderBreaking = isLeaderBreaking;
        this.leaders = leaders;
        this.discard = discard;
    }

}
