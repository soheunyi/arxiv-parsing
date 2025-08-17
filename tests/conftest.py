"""
Pytest configuration and fixtures for arXiv parsing tests.
"""

import asyncio
import pytest
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure async testing
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_arxiv_ids():
    """Sample arXiv IDs for testing."""
    return [
        "1706.03762",  # Attention is All You Need
        "1512.03385",  # ResNet
        "1810.04805",  # BERT
    ]


@pytest.fixture
def sample_tei_xml():
    """Sample TEI XML for testing."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <text>
            <body>
                <div type="references">
                    <listBibl>
                        <biblStruct xml:id="b0">
                            <analytic>
                                <title level="a" type="main">Attention Is All You Need</title>
                                <author>
                                    <persName>
                                        <forename type="first">Ashish</forename>
                                        <surname>Vaswani</surname>
                                    </persName>
                                </author>
                            </analytic>
                            <monogr>
                                <title level="m">Advances in Neural Information Processing Systems</title>
                                <imprint>
                                    <date type="published" when="2017">2017</date>
                                    <biblScope unit="page" from="5998" to="6008">5998-6008</biblScope>
                                </imprint>
                            </monogr>
                            <ptr type="arxiv" target="http://arxiv.org/abs/1706.03762"/>
                        </biblStruct>
                    </listBibl>
                </div>
            </body>
        </text>
    </TEI>'''