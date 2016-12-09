#!/usr/bin/env python

"""Print human-readable information about a Communication to stdout

concrete_inspect.py is a command-line script for printing out information
about a Concrete Communication.
"""

import argparse
import codecs
import sys

import concrete.version
import concrete.inspect
from concrete.util import CommunicationReader


def main():
    # Make stdout output UTF-8, preventing "'ascii' codec can't encode" errors
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)

    parser = argparse.ArgumentParser(
        description="Print information about a Concrete Communication to"
                    " stdout.  If communication_filename is specified, read"
                    " communication from file; otherwise, read from standard"
                    " input.",
    )
    parser.add_argument('--count', type=int,
                        help='Print at most this many communications.')
    parser.add_argument("--char-offsets",
                        help="Print token text extracted from character offset"
                             "s (not the text stored in the tokenization) in '"
                             "ConLL-style' format",
                        action="store_true")
    parser.add_argument("--dependency",
                        help="Print HEAD tags for first dependency parse in 'C"
                             "onLL-style' format",
                        action="store_true")
    parser.add_argument("--entities",
                        help="Print info about all Entities and their EntityMe"
                             "ntions",
                        action="store_true")
    parser.add_argument("--lemmas",
                        help="Print first set of lemma token tags in 'ConLL-st"
                             "yle' format",
                        action="store_true")
    parser.add_argument("--metadata",
                        help="Print metadata for tools used to annotate Commun"
                             "ication",
                        action="store_true")
    parser.add_argument("--mentions",
                        help="Print whitespace-separated tokens, with entity m"
                             "entions wrapped using <ENTITY ID=x> tags, where "
                             "'x' is the (zero-indexed) entity number",
                        action="store_true")
    parser.add_argument("--ner",
                        help="Print first set of Named Entity Recognition toke"
                             "n tags in 'ConLL-style' format",
                        action="store_true")
    parser.add_argument("--pos",
                        help="Print first set of Part-Of-Speech token tags in "
                             "'ConLL-style' format",
                        action="store_true")
    parser.add_argument("--sections",
                        action='store_true',
                        help="Print text according to Section offsets"
                             "(textSpan values). These textSpans are assumed "
                             "to be valid.")
    parser.add_argument("--situation-mentions",
                        help="Print info about all SituationMentions",
                        action="store_true")
    parser.add_argument("--situations",
                        help="Print info about all Situations and their Situat"
                             "ionMentions",
                        action="store_true")
    parser.add_argument("--text",
                        help="Print .text field",
                        action="store_true")
    parser.add_argument("--tokens",
                        help="Print whitespace-seperated tokens for *all* Toke"
                             "nizations in a Communication.  Each sentence tok"
                             "enization is printed on a separate line, and "
                             "empty lines indicate a section break",
                        action="store_true")
    parser.add_argument("--treebank",
                        help="Print Penn-Treebank style parse trees for *all* "
                             "Constituent Parses in the Communication",
                        action="store_true")
    parser.add_argument('--id',
                        help='Print communication id',
                        action='store_true')
    parser.add_argument("--no-references",
                        help="Don't add references to communication (may preve"
                             "nt 'NoneType' errors)",
                        action="store_true")
    parser.add_argument('communication_filename',
                        nargs='?',
                        type=str,
                        help='Path to a Concrete Communication from which '
                        'to display information. If not specified, read '
                        'read from standard input')
    concrete.version.add_argparse_argument(parser)
    args = parser.parse_args()

    add_references = not args.no_references

    if args.communication_filename is not None:
        comms = CommunicationReader(args.communication_filename,
                                    add_references=add_references)
    else:
        comms = CommunicationReader('/dev/fd/0', add_references=add_references)

    if not (args.char_offsets or args.dependency or args.lemmas or args.ner or
            args.pos or args.entities or args.mentions or args.metadata or
            args.sections or args.situation_mentions or args.situations or
            args.text or args.tokens or args.treebank or args.id):
        parser.print_help()
        sys.exit(1)

    comm_num = 0

    for comm in comms:
        if (args.char_offsets or args.dependency or args.lemmas or args.ner or
                args.pos):
            concrete.inspect.print_conll_style_tags_for_communication(
                comm, char_offsets=args.char_offsets,
                dependency=args.dependency,
                lemmas=args.lemmas, ner=args.ner, pos=args.pos)
        if args.entities:
            concrete.inspect.print_entities(comm)
        if args.mentions:
            concrete.inspect.print_tokens_with_entityMentions(comm)
        if args.metadata:
            concrete.inspect.print_metadata(comm)
        if args.sections:
            concrete.inspect.print_sections(comm)
        if args.situation_mentions:
            concrete.inspect.print_situation_mentions(comm)
        if args.situations:
            concrete.inspect.print_situations(comm)
        if args.text:
            concrete.inspect.print_text_for_communication(comm)
        if args.tokens:
            concrete.inspect.print_tokens_for_communication(comm)
        if args.treebank:
            concrete.inspect.print_penn_treebank_for_communication(comm)
        if args.id:
            concrete.inspect.print_id_for_communication(comm)

        comm_num += 1

        if args.count is not None and comm_num == args.count:
            break


if __name__ == "__main__":
    main()
