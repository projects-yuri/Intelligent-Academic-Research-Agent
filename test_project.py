"""Testes determinísticos: não consomem tokens e não acessam a internet."""

import json
import unittest
from unittest.mock import patch

from papers import get_library, get_local_paper, search_library
from tools import compare_studies, extract_information, search_papers


class TestPaperLibrary(unittest.TestCase):
    def test_ten_real_identifiers_and_links(self):
        papers = get_library()
        self.assertEqual(len(papers), 10)
        self.assertEqual(len({paper["id"] for paper in papers}), 10)
        self.assertTrue(all(paper["url"] == f"https://arxiv.org/abs/{paper['id']}" for paper in papers))

    def test_brazilian_portuguese_search(self):
        result = search_library("português brasileiro classificação", 5)
        self.assertTrue(any(paper["id"] == "2309.11052" for paper in result))

    def test_get_by_link(self):
        paper = get_local_paper("https://arxiv.org/abs/1705.00648")
        self.assertEqual(paper["id"], "1705.00648")

    @patch.dict("os.environ", {"OPENALEX_API_KEY": ""})
    def test_search_tool(self):
        output = json.loads(search_papers.invoke({"query": "BERT SVM", "max_results": 5}))
        self.assertTrue(output["papers"])
        self.assertTrue(all(paper["id"] for paper in output["papers"]))

    def test_compare_tool(self):
        output = json.loads(compare_studies.invoke({"references": ["1705.00648", "2309.11052"]}))
        self.assertEqual(len(output["studies"]), 2)
        self.assertEqual(output["errors"], [])

    @patch("tools._get_arxiv_abstract", return_value="Abstract verificado em teste.")
    def test_extract_tool(self, mocked):
        output = json.loads(extract_information.invoke({"reference": "2309.11052"}))
        self.assertEqual(output["id"], "2309.11052")
        self.assertEqual(output["abstract"], "Abstract verificado em teste.")
        mocked.assert_called_once_with("2309.11052")


if __name__ == "__main__":
    unittest.main()
