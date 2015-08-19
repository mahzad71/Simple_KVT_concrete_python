from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup

execfile('concrete/version.py')

setup(
    name = "concrete",
    version = __version__,
    description = "Python modules and scripts for working with Concrete",

    packages = [
        'concrete',

        # Python code generated by Thrift Compiler
        'concrete.audio',
        'concrete.clustering',
        'concrete.communication',
        'concrete.communication_service',
        'concrete.discourse',
        'concrete.email',
        'concrete.entities',
        'concrete.language',
        'concrete.linking',
        'concrete.metadata',
        'concrete.nitf',
        'concrete.services',
        'concrete.situations',
        'concrete.spans',
        'concrete.structure',
        'concrete.twitter',
        'concrete.uuid',

        # Python code generated by people
        'concrete.util',
        'tests',
        ],

    scripts = [
        'scripts/concrete2json.py',
        'scripts/concrete_diff.py',
        'scripts/concrete_inspect.py',
        'scripts/validate_communication.py',
        ],

    test_suite = "tests",

    install_requires = [
        'networkx',
        'testfixtures',
        'thrift>=0.9.2',
        'redis>=2.10.0',
        ],

    url = "https://github.com/hltcoe/concrete-python",
    license="BSD",
)
