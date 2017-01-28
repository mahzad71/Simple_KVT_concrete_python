from __future__ import unicode_literals
import time
import unittest

from testfixtures import LogCapture, StringComparison

import concrete
from concrete.util import add_references_to_communication
from concrete.validate import (
    validate_communication, validate_entity_mention_ids,
    validate_entity_mention_tokenization_ids,
    validate_token_offsets_for_section,
    validate_thrift_deep,
    validate_token_offsets_for_sentence
)
from concrete import (
    Section,
    Sentence,
    TextSpan,
    Token,
    TokenList,
    Tokenization,
)
from test_helper import read_test_comm


class TestAddReferences(unittest.TestCase):

    def test_add_references(self):
        comm = read_test_comm()
        add_references_to_communication(comm)


class TestCommunication(unittest.TestCase):

    def test_entity_mention_ids(self):
        comm = read_test_comm()
        self.assertTrue(validate_communication(comm))
        self.assertTrue(validate_entity_mention_ids(comm))

        comm.entitySetList[0].entityList[0].mentionIdList[
            0] = concrete.UUID(uuidString='BAD_ENTITY_MENTION_UUID')

        with LogCapture() as log_capture:
            self.assertFalse(validate_entity_mention_ids(comm))
        log_capture.check(('root', 'ERROR', StringComparison(
            r'.*invalid entityMentionId.*BAD_ENTITY_MENTION_UUID')))

    def test_entity_mention_tokenization(self):
        comm = read_test_comm()
        self.assertTrue(validate_communication(comm))
        self.assertTrue(validate_entity_mention_ids(comm))

        comm.entityMentionSetList[0].mentionList[0].tokens.tokenizationId = (
            concrete.UUID(uuidString='BAD_TOKENIZATION_UUID')
        )

        with LogCapture() as log_capture:
            self.assertFalse(validate_entity_mention_tokenization_ids(comm))
        log_capture.check(('root', 'ERROR', StringComparison(
            r'.*invalid tokenizationId.*BAD_TOKENIZATION_UUID')))


class TestRequiredThriftFields(unittest.TestCase):

    def test_check_required_fields(self):
        # When a field is marked as required in a .thrift file, the
        # Python code generated by the Thrift compiler only seems to
        # capture this requirement in the validate() function for the
        # generated class.  While the ThriftGeneratedClass.thrift_spec
        # structure captures the names and types of the fields,
        # thrift_spec does not seem to store any flags indicating
        # whether or not a field is required.
        #
        # Here is the validate() function for the Communication class:
        #
        #    def validate(self):
        #        if self.id is None:
        #            raise TProtocol.TProtocolException(
        #                message='Required field id is unset!')
        #        if self.uuid is None:
        #            raise TProtocol.TProtocolException(
        #                message='Required field uuid is unset!')
        #        if self.type is None:
        #            raise TProtocol.TProtocolException(
        #                message='Required field type is unset!')
        #        return
        #
        # The validate() function raises an exception when it can't
        # find a required field.  There doesn't seem to be any way to
        # determine whether multiple required fields are missing,
        # aside from assigning a value to the required field and
        # running validate() again.

        comm = concrete.Communication()

        with LogCapture() as log_capture:
            self.assertFalse(
                validate_thrift_deep(comm))
        log_capture.check(
            ('root', 'ERROR',
             "Communication: Required Field 'id' is unset!"))

        comm.id = "ID"
        with LogCapture() as log_capture:
            self.assertFalse(
                validate_thrift_deep(comm))
        log_capture.check(
            ('root', 'ERROR',
             "Communication: Required Field 'uuid' is unset!"))

        comm.uuid = concrete.UUID(uuidString="TEST_UUID")
        with LogCapture() as log_capture:
            self.assertFalse(
                validate_thrift_deep(comm))
        log_capture.check(('root', 'ERROR', StringComparison(
            r".*TEST_UUID.*Required Field 'type' is unset!")))

        comm.metadata = concrete.AnnotationMetadata(
            tool="TEST", timestamp=int(time.time()))

        comm.type = "OTHER"
        self.assertTrue(
            validate_thrift_deep(comm))


class TestTextspanOffsets(unittest.TestCase):

    def test_validate_token_offsets_for_good_sentence(self):
        sentence = self.create_sentence_with_token(0, 30, 0, 10)
        self.assertTrue(validate_token_offsets_for_sentence(sentence))

    def test_validate_token_offsets_for_sentence_rev_sentence_offsets(self):
        sentence = self.create_sentence_with_token(30, 0, 0, 10)
        with LogCapture():
            self.assertFalse(validate_token_offsets_for_sentence(sentence))

    def test_validate_token_offsets_for_sentence_rev_token_offsets(self):
        sentence = self.create_sentence_with_token(0, 30, 10, 0)
        with LogCapture():
            self.assertFalse(validate_token_offsets_for_sentence(sentence))

    def test_validate_token_offsets_for_sentence_token_outside(self):
        sentence = self.create_sentence_with_token(0, 30, 25, 35)
        with LogCapture():
            self.assertFalse(validate_token_offsets_for_sentence(sentence))

    def test_validate_token_offsets_for_good_section(self):
        section = self.create_section_with_sentence(0, 30, 0, 10)
        self.assertTrue(validate_token_offsets_for_section(section))

    def test_validate_token_offsets_for_section_rev_section_offsets(self):
        section = self.create_section_with_sentence(30, 0, 0, 10)
        with LogCapture():
            self.assertFalse(validate_token_offsets_for_section(section))

    def test_validate_token_offsets_for_section_rev_token_offsets(self):
        section = self.create_section_with_sentence(0, 30, 10, 0)
        with LogCapture():
            self.assertFalse(validate_token_offsets_for_section(section))

    def test_validate_token_offsets_for_section_sentence_outside(self):
        section = self.create_section_with_sentence(0, 30, 25, 35)
        with LogCapture():
            self.assertFalse(validate_token_offsets_for_section(section))

    def test_validate_token_offsets_for_real_example_section_data(self):
        section = self.create_section_with_sentence(55, 296, 0, 118)
        with LogCapture():
            self.assertFalse(validate_token_offsets_for_section(section))

    def create_section_with_sentence(self, section_start, section_ending,
                                     sentence_start, sentence_ending):
        sentence_textspan = TextSpan(
            start=sentence_start, ending=sentence_ending)
        sentence = Sentence(
            textSpan=sentence_textspan, uuid='TEST_SENTENCE')
        section_textspan = TextSpan(
            start=section_start, ending=section_ending)
        section = Section(
            sentenceList=[sentence],
            textSpan=section_textspan,
            uuid='TEST_SECTION')
        return section

    def create_sentence_with_token(self, sentence_start, sentence_ending,
                                   token_start, token_ending):
        token_textspan = TextSpan(
            start=token_start, ending=token_ending)
        token = Token(textSpan=token_textspan)
        tokenization = Tokenization(
            tokenList=TokenList(tokenList=[token]))
        sentence_textspan = TextSpan(
            start=sentence_start, ending=sentence_ending)
        sentence = Sentence(
            tokenization=tokenization,
            textSpan=sentence_textspan,
            uuid='TEST')
        return sentence
