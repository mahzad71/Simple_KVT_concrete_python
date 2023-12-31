# -*- coding: utf-8 -*-
#
# Autogenerated by Thrift Compiler (0.18.0)
#
# DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
#
#  options string: py:coding=utf-8
#

from thrift.Thrift import TType, TMessageType, TFrozenDict, TException, TApplicationException
from thrift.protocol.TProtocol import TProtocolException
from thrift.TRecursive import fix_spec

import sys
import concrete.communication.ttypes
import concrete.services.ttypes
import concrete.structure.ttypes
import concrete.uuid.ttypes
import concrete.metadata.ttypes
import concrete.entities.ttypes

from thrift.transport import TTransport
all_structs = []


class SearchType(object):
    """
    What are we searching over

    """
    COMMUNICATIONS = 0
    SECTIONS = 1
    SENTENCES = 2
    ENTITIES = 3
    ENTITY_MENTIONS = 4
    SITUATIONS = 5
    SITUATION_MENTIONS = 6

    _VALUES_TO_NAMES = {
        0: "COMMUNICATIONS",
        1: "SECTIONS",
        2: "SENTENCES",
        3: "ENTITIES",
        4: "ENTITY_MENTIONS",
        5: "SITUATIONS",
        6: "SITUATION_MENTIONS",
    }

    _NAMES_TO_VALUES = {
        "COMMUNICATIONS": 0,
        "SECTIONS": 1,
        "SENTENCES": 2,
        "ENTITIES": 3,
        "ENTITY_MENTIONS": 4,
        "SITUATIONS": 5,
        "SITUATION_MENTIONS": 6,
    }


class SearchFeedback(object):
    """
    Feedback values

    """
    NEGATIVE = -1
    NONE = 0
    POSITIVE = 1

    _VALUES_TO_NAMES = {
        -1: "NEGATIVE",
        0: "NONE",
        1: "POSITIVE",
    }

    _NAMES_TO_VALUES = {
        "NEGATIVE": -1,
        "NONE": 0,
        "POSITIVE": 1,
    }


class SearchCapability(object):
    """
    A search provider describes its capabilities with a list of search type and language pairs.

    Attributes:
     - type: A type of search supported by the search provider
     - lang: Language that the search provider supports.
    Use ISO 639-2/T three letter codes.

    """


    def __init__(self, type=None, lang=None,):
        self.type = type
        self.lang = lang

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.I32:
                    self.type = iprot.readI32()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.STRING:
                    self.lang = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('SearchCapability')
        if self.type is not None:
            oprot.writeFieldBegin('type', TType.I32, 1)
            oprot.writeI32(self.type)
            oprot.writeFieldEnd()
        if self.lang is not None:
            oprot.writeFieldBegin('lang', TType.STRING, 2)
            oprot.writeString(self.lang.encode('utf-8') if sys.version_info[0] == 2 else self.lang)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        if self.type is None:
            raise TProtocolException(message='Required field type is unset!')
        if self.lang is None:
            raise TProtocolException(message='Required field lang is unset!')
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)


class SearchQuery(object):
    """
    Wrapper for information relevant to a (possibly structured) search.

    Attributes:
     - terms: Individual words, or multiword phrases, e.g., 'dog', 'blue
    cheese'.  It is the responsibility of the implementation of
    Search* to tokenize multiword phrases, if so-desired.  Further,
    an implementation may choose to support advanced features such as
    wildcards, e.g.: 'blue*'.  This specification makes no
    committment as to the internal structure of keywords and their
    semantics: that is the responsibility of the individual
    implementation.
     - questions: e.g., "what is the capital of spain?"

    questions is a list in order that possibly different phrasings of
    the question can be included, e.g.: "what is the name of spain's
    capital?"
     - communicationId: Refers to an optional communication that can provide context for the query.
     - tokens: Refers to a sequence of tokens in the communication referenced by communicationId.
     - rawQuery: The input from the user provided in the search box, unmodified
     - auths: optional authorization mechanism
     - userId: Identifies the user who submitted the search query
     - name: Human readable name of the query.
     - labels: Properties of the query or user.
    These labels can be used to group queries and results by a domain or group of
    users for training. An example usage would be assigning the geographical region
    as a label ("spain"). User labels could be based on organizational units ("hltcoe").
     - type: This search is over this type of data (communications, sentences, entities)
     - lang: The language of the corpus that the user wants to search.
    Use ISO 639-2/T three letter codes.
     - corpus: An identifier of the corpus that the search is to be performed over.
     - k: The maximum number of candidates the search service should return.
     - communication: An optional communication used as context for the query.
    If both this field and communicationId is populated, then it is
    assumed the ID of the communication is the same as communicationId.

    """


    def __init__(self, terms=None, questions=None, communicationId=None, tokens=None, rawQuery=None, auths=None, userId=None, name=None, labels=None, type=None, lang=None, corpus=None, k=None, communication=None,):
        self.terms = terms
        self.questions = questions
        self.communicationId = communicationId
        self.tokens = tokens
        self.rawQuery = rawQuery
        self.auths = auths
        self.userId = userId
        self.name = name
        self.labels = labels
        self.type = type
        self.lang = lang
        self.corpus = corpus
        self.k = k
        self.communication = communication

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.LIST:
                    self.terms = []
                    (_etype3, _size0) = iprot.readListBegin()
                    for _i4 in range(_size0):
                        _elem5 = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                        self.terms.append(_elem5)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.LIST:
                    self.questions = []
                    (_etype9, _size6) = iprot.readListBegin()
                    for _i10 in range(_size6):
                        _elem11 = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                        self.questions.append(_elem11)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.STRING:
                    self.communicationId = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 4:
                if ftype == TType.STRUCT:
                    self.tokens = concrete.structure.ttypes.TokenRefSequence()
                    self.tokens.read(iprot)
                else:
                    iprot.skip(ftype)
            elif fid == 5:
                if ftype == TType.STRING:
                    self.rawQuery = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 6:
                if ftype == TType.STRING:
                    self.auths = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 7:
                if ftype == TType.STRING:
                    self.userId = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 8:
                if ftype == TType.STRING:
                    self.name = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 9:
                if ftype == TType.LIST:
                    self.labels = []
                    (_etype15, _size12) = iprot.readListBegin()
                    for _i16 in range(_size12):
                        _elem17 = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                        self.labels.append(_elem17)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 10:
                if ftype == TType.I32:
                    self.type = iprot.readI32()
                else:
                    iprot.skip(ftype)
            elif fid == 11:
                if ftype == TType.STRING:
                    self.lang = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 12:
                if ftype == TType.STRING:
                    self.corpus = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 13:
                if ftype == TType.I32:
                    self.k = iprot.readI32()
                else:
                    iprot.skip(ftype)
            elif fid == 14:
                if ftype == TType.STRUCT:
                    self.communication = concrete.communication.ttypes.Communication()
                    self.communication.read(iprot)
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('SearchQuery')
        if self.terms is not None:
            oprot.writeFieldBegin('terms', TType.LIST, 1)
            oprot.writeListBegin(TType.STRING, len(self.terms))
            for iter18 in self.terms:
                oprot.writeString(iter18.encode('utf-8') if sys.version_info[0] == 2 else iter18)
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        if self.questions is not None:
            oprot.writeFieldBegin('questions', TType.LIST, 2)
            oprot.writeListBegin(TType.STRING, len(self.questions))
            for iter19 in self.questions:
                oprot.writeString(iter19.encode('utf-8') if sys.version_info[0] == 2 else iter19)
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        if self.communicationId is not None:
            oprot.writeFieldBegin('communicationId', TType.STRING, 3)
            oprot.writeString(self.communicationId.encode('utf-8') if sys.version_info[0] == 2 else self.communicationId)
            oprot.writeFieldEnd()
        if self.tokens is not None:
            oprot.writeFieldBegin('tokens', TType.STRUCT, 4)
            self.tokens.write(oprot)
            oprot.writeFieldEnd()
        if self.rawQuery is not None:
            oprot.writeFieldBegin('rawQuery', TType.STRING, 5)
            oprot.writeString(self.rawQuery.encode('utf-8') if sys.version_info[0] == 2 else self.rawQuery)
            oprot.writeFieldEnd()
        if self.auths is not None:
            oprot.writeFieldBegin('auths', TType.STRING, 6)
            oprot.writeString(self.auths.encode('utf-8') if sys.version_info[0] == 2 else self.auths)
            oprot.writeFieldEnd()
        if self.userId is not None:
            oprot.writeFieldBegin('userId', TType.STRING, 7)
            oprot.writeString(self.userId.encode('utf-8') if sys.version_info[0] == 2 else self.userId)
            oprot.writeFieldEnd()
        if self.name is not None:
            oprot.writeFieldBegin('name', TType.STRING, 8)
            oprot.writeString(self.name.encode('utf-8') if sys.version_info[0] == 2 else self.name)
            oprot.writeFieldEnd()
        if self.labels is not None:
            oprot.writeFieldBegin('labels', TType.LIST, 9)
            oprot.writeListBegin(TType.STRING, len(self.labels))
            for iter20 in self.labels:
                oprot.writeString(iter20.encode('utf-8') if sys.version_info[0] == 2 else iter20)
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        if self.type is not None:
            oprot.writeFieldBegin('type', TType.I32, 10)
            oprot.writeI32(self.type)
            oprot.writeFieldEnd()
        if self.lang is not None:
            oprot.writeFieldBegin('lang', TType.STRING, 11)
            oprot.writeString(self.lang.encode('utf-8') if sys.version_info[0] == 2 else self.lang)
            oprot.writeFieldEnd()
        if self.corpus is not None:
            oprot.writeFieldBegin('corpus', TType.STRING, 12)
            oprot.writeString(self.corpus.encode('utf-8') if sys.version_info[0] == 2 else self.corpus)
            oprot.writeFieldEnd()
        if self.k is not None:
            oprot.writeFieldBegin('k', TType.I32, 13)
            oprot.writeI32(self.k)
            oprot.writeFieldEnd()
        if self.communication is not None:
            oprot.writeFieldBegin('communication', TType.STRUCT, 14)
            self.communication.write(oprot)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        if self.type is None:
            raise TProtocolException(message='Required field type is unset!')
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)


class SearchResultItem(object):
    """
    An individual element returned from a search.  Most/all methods
    will return a communicationId, possibly with an associated score.
    For example if the target element type of the search is Sentence
    then the sentenceId field should be populated.

    Attributes:
     - communicationId
     - sentenceId: The UUID of the returned sentence, which appears in the
    communication referenced by communicationId.
     - score: Values are not restricted in range (e.g., do not have to be
    within [0,1]).  Higher is better.

     - tokens: If SearchType=ENTITY_MENTIONS then this field should be populated.
    Otherwise, this field may be optionally populated in order to
    provide a hint to the client as to where to center a
    visualization, or the extraction of context, etc.
     - entity: If SearchType=ENTITIES then this field should be populated.

    """


    def __init__(self, communicationId=None, sentenceId=None, score=None, tokens=None, entity=None,):
        self.communicationId = communicationId
        self.sentenceId = sentenceId
        self.score = score
        self.tokens = tokens
        self.entity = entity

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.STRING:
                    self.communicationId = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.STRUCT:
                    self.sentenceId = concrete.uuid.ttypes.UUID()
                    self.sentenceId.read(iprot)
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.DOUBLE:
                    self.score = iprot.readDouble()
                else:
                    iprot.skip(ftype)
            elif fid == 4:
                if ftype == TType.STRUCT:
                    self.tokens = concrete.structure.ttypes.TokenRefSequence()
                    self.tokens.read(iprot)
                else:
                    iprot.skip(ftype)
            elif fid == 5:
                if ftype == TType.STRUCT:
                    self.entity = concrete.entities.ttypes.Entity()
                    self.entity.read(iprot)
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('SearchResultItem')
        if self.communicationId is not None:
            oprot.writeFieldBegin('communicationId', TType.STRING, 1)
            oprot.writeString(self.communicationId.encode('utf-8') if sys.version_info[0] == 2 else self.communicationId)
            oprot.writeFieldEnd()
        if self.sentenceId is not None:
            oprot.writeFieldBegin('sentenceId', TType.STRUCT, 2)
            self.sentenceId.write(oprot)
            oprot.writeFieldEnd()
        if self.score is not None:
            oprot.writeFieldBegin('score', TType.DOUBLE, 3)
            oprot.writeDouble(self.score)
            oprot.writeFieldEnd()
        if self.tokens is not None:
            oprot.writeFieldBegin('tokens', TType.STRUCT, 4)
            self.tokens.write(oprot)
            oprot.writeFieldEnd()
        if self.entity is not None:
            oprot.writeFieldBegin('entity', TType.STRUCT, 5)
            self.entity.write(oprot)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)


class SearchResult(object):
    """
    Single wrapper for results from all the various Search* services.

    Attributes:
     - uuid: Unique identifier for the results of this search.
     - searchQuery: The query that led to this result.
    Useful for capturing feedback or building training data.
     - searchResultItems: The list is assumed sorted best to worst, which should be
    reflected by the values contained in the score field of each
    SearchResult, if that field is populated.
     - metadata: The system that provided the response: likely use case for
    populating this field is for building training data.  Presumably
    a system will not need/want to return this object in live use.
     - lang: The dominant language of the search results.
    Use ISO 639-2/T three letter codes.
    Search providers should set this when possible to support downstream processing.
    Do not set if it is not known.
    If multilingual, use the string "multilingual".

    """


    def __init__(self, uuid=None, searchQuery=None, searchResultItems=None, metadata=None, lang=None,):
        self.uuid = uuid
        self.searchQuery = searchQuery
        self.searchResultItems = searchResultItems
        self.metadata = metadata
        self.lang = lang

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.STRUCT:
                    self.uuid = concrete.uuid.ttypes.UUID()
                    self.uuid.read(iprot)
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.STRUCT:
                    self.searchQuery = SearchQuery()
                    self.searchQuery.read(iprot)
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.LIST:
                    self.searchResultItems = []
                    (_etype24, _size21) = iprot.readListBegin()
                    for _i25 in range(_size21):
                        _elem26 = SearchResultItem()
                        _elem26.read(iprot)
                        self.searchResultItems.append(_elem26)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 4:
                if ftype == TType.STRUCT:
                    self.metadata = concrete.metadata.ttypes.AnnotationMetadata()
                    self.metadata.read(iprot)
                else:
                    iprot.skip(ftype)
            elif fid == 5:
                if ftype == TType.STRING:
                    self.lang = iprot.readString().decode('utf-8', errors='replace') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('SearchResult')
        if self.uuid is not None:
            oprot.writeFieldBegin('uuid', TType.STRUCT, 1)
            self.uuid.write(oprot)
            oprot.writeFieldEnd()
        if self.searchQuery is not None:
            oprot.writeFieldBegin('searchQuery', TType.STRUCT, 2)
            self.searchQuery.write(oprot)
            oprot.writeFieldEnd()
        if self.searchResultItems is not None:
            oprot.writeFieldBegin('searchResultItems', TType.LIST, 3)
            oprot.writeListBegin(TType.STRUCT, len(self.searchResultItems))
            for iter27 in self.searchResultItems:
                iter27.write(oprot)
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        if self.metadata is not None:
            oprot.writeFieldBegin('metadata', TType.STRUCT, 4)
            self.metadata.write(oprot)
            oprot.writeFieldEnd()
        if self.lang is not None:
            oprot.writeFieldBegin('lang', TType.STRING, 5)
            oprot.writeString(self.lang.encode('utf-8') if sys.version_info[0] == 2 else self.lang)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        if self.uuid is None:
            raise TProtocolException(message='Required field uuid is unset!')
        if self.searchQuery is None:
            raise TProtocolException(message='Required field searchQuery is unset!')
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
all_structs.append(SearchCapability)
SearchCapability.thrift_spec = (
    None,  # 0
    (1, TType.I32, 'type', None, None, ),  # 1
    (2, TType.STRING, 'lang', 'UTF8', None, ),  # 2
)
all_structs.append(SearchQuery)
SearchQuery.thrift_spec = (
    None,  # 0
    (1, TType.LIST, 'terms', (TType.STRING, 'UTF8', False), None, ),  # 1
    (2, TType.LIST, 'questions', (TType.STRING, 'UTF8', False), None, ),  # 2
    (3, TType.STRING, 'communicationId', 'UTF8', None, ),  # 3
    (4, TType.STRUCT, 'tokens', [concrete.structure.ttypes.TokenRefSequence, None], None, ),  # 4
    (5, TType.STRING, 'rawQuery', 'UTF8', None, ),  # 5
    (6, TType.STRING, 'auths', 'UTF8', None, ),  # 6
    (7, TType.STRING, 'userId', 'UTF8', None, ),  # 7
    (8, TType.STRING, 'name', 'UTF8', None, ),  # 8
    (9, TType.LIST, 'labels', (TType.STRING, 'UTF8', False), None, ),  # 9
    (10, TType.I32, 'type', None, None, ),  # 10
    (11, TType.STRING, 'lang', 'UTF8', None, ),  # 11
    (12, TType.STRING, 'corpus', 'UTF8', None, ),  # 12
    (13, TType.I32, 'k', None, None, ),  # 13
    (14, TType.STRUCT, 'communication', [concrete.communication.ttypes.Communication, None], None, ),  # 14
)
all_structs.append(SearchResultItem)
SearchResultItem.thrift_spec = (
    None,  # 0
    (1, TType.STRING, 'communicationId', 'UTF8', None, ),  # 1
    (2, TType.STRUCT, 'sentenceId', [concrete.uuid.ttypes.UUID, None], None, ),  # 2
    (3, TType.DOUBLE, 'score', None, None, ),  # 3
    (4, TType.STRUCT, 'tokens', [concrete.structure.ttypes.TokenRefSequence, None], None, ),  # 4
    (5, TType.STRUCT, 'entity', [concrete.entities.ttypes.Entity, None], None, ),  # 5
)
all_structs.append(SearchResult)
SearchResult.thrift_spec = (
    None,  # 0
    (1, TType.STRUCT, 'uuid', [concrete.uuid.ttypes.UUID, None], None, ),  # 1
    (2, TType.STRUCT, 'searchQuery', [SearchQuery, None], None, ),  # 2
    (3, TType.LIST, 'searchResultItems', (TType.STRUCT, [SearchResultItem, None], False), None, ),  # 3
    (4, TType.STRUCT, 'metadata', [concrete.metadata.ttypes.AnnotationMetadata, None], None, ),  # 4
    (5, TType.STRING, 'lang', 'UTF8', None, ),  # 5
)
fix_spec(all_structs)
del all_structs
